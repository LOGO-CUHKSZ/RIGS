"""Reusable rooted product-graph inputs for protocol-bound generation.

The root-choice protocol acts on a canonical product-atom ordering.  This
module turns a product SMILES plus one frozen atom permutation into the dense
categorical GraphDiT input contract without relying on a dataset-specific
endpoint cache.  It deliberately owns no model, RC predictor, or scoring
logic, so inference, API generation, and round-trip candidate generation can
share the same frozen rooted inputs.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor

from retro_guided_generation.models.graphdit.config import GraphDiTConfig

from .io import EvaluationError


class CategoricalGraphVocabulary(Protocol):
    """The vocabulary surface required by :class:`RootedProductGraphBuilder`."""

    vocabulary_id: str
    node_symbols: tuple[str, ...]
    edge_names: tuple[str | None, ...]

    @property
    def node_classes(self) -> int: ...

    @property
    def edge_classes(self) -> int: ...


def _rdkit():
    try:
        from rdkit import Chem
    except ImportError as error:  # pragma: no cover - dependency-specific
        raise RuntimeError("RDKit is required to build rooted product graphs") from error
    return Chem


def canonical_product_smiles(product_smiles: str) -> str:
    """Canonicalize one product exactly in the Pista RC/GraphDiT convention."""

    if not isinstance(product_smiles, str) or not product_smiles.strip():
        raise EvaluationError("product_smiles must be a non-empty string")
    Chem = _rdkit()
    molecule = Chem.MolFromSmiles(product_smiles.strip())
    if molecule is None:
        raise EvaluationError("product_smiles cannot be parsed")
    for atom in molecule.GetAtoms():
        if atom.HasProp("molAtomMapNumber"):
            atom.ClearProp("molAtomMapNumber")
    result = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    if not result:
        raise EvaluationError("product_smiles canonicalization returned an empty value")
    return result


def product_atom_count(canonical_smiles: str) -> int:
    """Return a checked product atom count for a canonical product SMILES."""

    Chem = _rdkit()
    molecule = Chem.MolFromSmiles(canonical_smiles)
    if molecule is None:
        raise EvaluationError("canonical product_smiles cannot be parsed")
    count = int(molecule.GetNumAtoms())
    if count < 2:
        raise EvaluationError("product_smiles must contain at least two atoms")
    if molecule.GetNumBonds() < 1:
        raise EvaluationError("product_smiles must contain at least one bond")
    return count


@dataclass(frozen=True, slots=True)
class RootedProductGraphBatch:
    """One dense, product-only GraphDiT batch with frozen root orderings."""

    p_X: Tensor
    p_E: Tensor
    node_mask: Tensor
    p_mask: Tensor
    cond: Tensor
    canonical_product_smiles: tuple[str, ...]
    product_atom_counts: tuple[int, ...]
    applied_permutations: tuple[tuple[int, ...], ...]

    def to_device(self, device: torch.device | str) -> "RootedProductGraphBatch":
        """Move the completed CPU batch in one transfer per tensor field."""

        target = torch.device(device)

        def move(value: Tensor) -> Tensor:
            if value.device == target:
                return value
            if target.type == "cuda" and value.device.type == "cpu" and value.numel() > 0:
                if not value.is_pinned():
                    value = value.pin_memory()
                return value.to(device=target, non_blocking=True)
            return value.to(device=target)

        return RootedProductGraphBatch(
            p_X=move(self.p_X),
            p_E=move(self.p_E),
            node_mask=move(self.node_mask),
            p_mask=move(self.p_mask),
            cond=move(self.cond),
            canonical_product_smiles=self.canonical_product_smiles,
            product_atom_counts=self.product_atom_counts,
            applied_permutations=self.applied_permutations,
        )


class RootedProductGraphBuilder:
    """Build checkpoint-compatible GraphDiT inputs from frozen rooted products."""

    def __init__(
        self,
        *,
        vocabulary: CategoricalGraphVocabulary,
        graphdit: GraphDiTConfig,
        template_cache_capacity: int = 4_096,
    ) -> None:
        if graphdit.node_classes != vocabulary.node_classes:
            raise EvaluationError("graph vocabulary node width differs from GraphDiTConfig")
        if graphdit.edge_classes != vocabulary.edge_classes:
            raise EvaluationError("graph vocabulary edge width differs from GraphDiTConfig")
        if not vocabulary.node_symbols or vocabulary.node_symbols[-1] != "*":
            raise EvaluationError("rooted product vocabulary must reserve the final node class for '*'")
        if not vocabulary.edge_names or vocabulary.edge_names[0] is not None:
            raise EvaluationError("rooted product vocabulary must reserve edge class zero for no bond")
        bond_names = tuple(name for name in vocabulary.edge_names[1:] if name is not None)
        if len(set(bond_names)) != len(bond_names):
            raise EvaluationError("rooted product vocabulary contains duplicate bond names")
        if (
            isinstance(template_cache_capacity, bool)
            or not isinstance(template_cache_capacity, int)
            or template_cache_capacity < 1
        ):
            raise EvaluationError("template_cache_capacity must be a positive integer")
        self.vocabulary = vocabulary
        self.graphdit = graphdit
        self.template_cache_capacity = template_cache_capacity
        self._node_index = {symbol: index for index, symbol in enumerate(vocabulary.node_symbols)}
        self._bond_index = {
            str(name): index
            for index, name in enumerate(vocabulary.edge_names)
            if index > 0 and name is not None
        }
        # RC sampling reuses a small set of rooted permutations for the same
        # products.  Cache only the canonical unpermuted dense template so
        # every draw still receives its explicit frozen permutation.  The
        # cache changes neither the tensor contract nor ordering; it removes
        # repeated RDKit parsing and one-hot construction from the hot path.
        self._template_cache: OrderedDict[str, tuple[Tensor, Tensor, int]] = OrderedDict()
        self._template_cache_hits = 0
        self._template_cache_misses = 0

    @property
    def template_cache_statistics(self) -> dict[str, int]:
        """Return bounded runtime-cache telemetry without content hashing."""

        return {
            "capacity": self.template_cache_capacity,
            "entries": len(self._template_cache),
            "hits": self._template_cache_hits,
            "misses": self._template_cache_misses,
        }

    @staticmethod
    def _validate_permutation(
        permutation: Sequence[int] | None,
        *,
        atom_count: int,
        index: int,
    ) -> tuple[int, ...]:
        if permutation is None:
            return tuple(range(atom_count))
        if isinstance(permutation, (str, bytes)) or not isinstance(permutation, Sequence):
            raise EvaluationError(f"permutations[{index}] must be an integer sequence")
        values = tuple(permutation)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise EvaluationError(f"permutations[{index}] contains a non-integer")
        if len(values) != atom_count:
            raise EvaluationError(
                f"permutations[{index}] length differs from product atom count"
            )
        if sorted(values) != list(range(atom_count)):
            raise EvaluationError(f"permutations[{index}] is not a bijection")
        return values

    def _unpermuted_template(
        self,
        canonical: str,
        *,
        index: int,
    ) -> tuple[Tensor, Tensor, int]:
        cached = self._template_cache.pop(canonical, None)
        if cached is not None:
            self._template_cache[canonical] = cached
            self._template_cache_hits += 1
            return cached
        self._template_cache_misses += 1
        Chem = _rdkit()
        molecule = Chem.MolFromSmiles(canonical)
        if molecule is None:  # Defensive: canonical_product_smiles already checked this.
            raise EvaluationError(f"product_smiles[{index}] cannot be parsed after canonicalization")
        atom_count = product_atom_count(canonical)
        if atom_count > self.graphdit.max_n_len:
            raise EvaluationError(f"product_smiles[{index}] exceeds GraphDiT max_n_len")
        symbols = {atom.GetSymbol() for atom in molecule.GetAtoms()}
        unknown = sorted(symbols - set(self._node_index))
        if unknown or "*" in symbols:
            raise EvaluationError(f"product_smiles[{index}] has unsupported atom symbol(s): {unknown}")
        nodes = atom_count + self.graphdit.n_dummy
        dummy_index = self.graphdit.node_classes - 1
        x = torch.zeros((nodes, self.graphdit.node_classes), dtype=torch.float32)
        x[:, dummy_index] = 1.0
        for atom in molecule.GetAtoms():
            atom_index = int(atom.GetIdx())
            symbol = atom.GetSymbol()
            node_index = self._node_index.get(symbol)
            if node_index is None or node_index == dummy_index:
                raise EvaluationError(f"product_smiles[{index}] has unsupported atom {symbol!r}")
            x[atom_index].zero_()
            x[atom_index, node_index] = 1.0

        e = torch.zeros(
            (nodes, nodes, self.graphdit.edge_classes),
            dtype=torch.float32,
        )
        e[..., 0] = 1.0
        diagonal = torch.arange(nodes)
        e[diagonal, diagonal] = 0.0
        for bond in molecule.GetBonds():
            name = str(bond.GetBondType())
            edge_index = self._bond_index.get(name)
            if edge_index is None:
                raise EvaluationError(
                    f"product_smiles[{index}] has unsupported bond type {name!r}"
                )
            left = int(bond.GetBeginAtomIdx())
            right = int(bond.GetEndAtomIdx())
            if left == right:
                raise EvaluationError(f"product_smiles[{index}] has a self-loop bond")
            for first, second in ((left, right), (right, left)):
                e[first, second].zero_()
                e[first, second, edge_index] = 1.0

        template = (x.contiguous(), e.contiguous(), atom_count)
        self._template_cache[canonical] = template
        while len(self._template_cache) > self.template_cache_capacity:
            self._template_cache.popitem(last=False)
        return template

    def _build_one(
        self,
        product_smiles: str,
        *,
        index: int,
        permutation: Sequence[int] | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, str, int, tuple[int, ...]]:
        canonical = canonical_product_smiles(product_smiles)
        base_x, base_e, atom_count = self._unpermuted_template(canonical, index=index)
        atom_permutation = self._validate_permutation(
            permutation,
            atom_count=atom_count,
            index=index,
        )
        nodes = atom_count + self.graphdit.n_dummy
        full_permutation = torch.tensor(
            atom_permutation + tuple(range(atom_count, nodes)),
            dtype=torch.long,
        )
        x = base_x.index_select(0, full_permutation)
        e = base_e.index_select(0, full_permutation).index_select(1, full_permutation)
        node_mask = torch.ones(nodes, dtype=torch.bool)
        dummy_index = self.graphdit.node_classes - 1
        p_mask = ~x[:, dummy_index].bool()
        if not torch.equal(e, e.transpose(0, 1)):
            raise EvaluationError("rooted product edge tensor is not symmetric")
        if not bool(torch.all(~p_mask <= node_mask)):
            raise EvaluationError("rooted product dummy node escaped the node mask")
        return x, e, node_mask, p_mask, canonical, atom_count, atom_permutation

    def build(
        self,
        product_smiles: Sequence[str],
        *,
        permutations: Sequence[Sequence[int] | None] | None = None,
    ) -> RootedProductGraphBatch:
        if isinstance(product_smiles, (str, bytes)):
            raise EvaluationError("product_smiles must be a sequence, not a string")
        products = list(product_smiles)
        if not products:
            raise EvaluationError("product_smiles must not be empty")
        if permutations is None:
            rootings: list[Sequence[int] | None] = [None] * len(products)
        else:
            if isinstance(permutations, (str, bytes)):
                raise EvaluationError("permutations must be a sequence")
            rootings = list(permutations)
            if len(rootings) != len(products):
                raise EvaluationError("permutations length must equal product_smiles length")
        built = [
            self._build_one(product, index=index, permutation=rootings[index])
            for index, product in enumerate(products)
        ]
        max_nodes = max(item[0].shape[0] for item in built)
        batch_size = len(built)
        p_x = torch.zeros(
            (batch_size, max_nodes, self.graphdit.node_classes),
            dtype=torch.float32,
        )
        p_e = torch.zeros(
            (batch_size, max_nodes, max_nodes, self.graphdit.edge_classes),
            dtype=torch.float32,
        )
        node_mask = torch.zeros((batch_size, max_nodes), dtype=torch.bool)
        p_mask = torch.zeros_like(node_mask)
        for row_index, (x, e, mask, product_mask, *_rest) in enumerate(built):
            nodes = x.shape[0]
            p_x[row_index, :nodes] = x
            p_e[row_index, :nodes, :nodes] = e
            node_mask[row_index, :nodes] = mask
            p_mask[row_index, :nodes] = product_mask
        cond = torch.zeros(
            (batch_size, self.graphdit.condition_dim),
            dtype=torch.float32,
        )
        return RootedProductGraphBatch(
            p_X=p_x,
            p_E=p_e,
            node_mask=node_mask,
            p_mask=p_mask,
            cond=cond,
            canonical_product_smiles=tuple(item[4] for item in built),
            product_atom_counts=tuple(int(item[5]) for item in built),
            applied_permutations=tuple(item[6] for item in built),
        )


__all__ = [
    "CategoricalGraphVocabulary",
    "RootedProductGraphBatch",
    "RootedProductGraphBuilder",
    "canonical_product_smiles",
    "product_atom_count",
]
