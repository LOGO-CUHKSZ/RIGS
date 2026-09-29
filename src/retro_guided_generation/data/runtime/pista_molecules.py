"""Pista graph-vocabulary decoding for evaluation adapters.

Training operates only on dense categorical graph tensors.  Evaluation needs
to compare generated endpoints against known route molecules, so this module
turns one graph into a canonical SMILES string without importing a legacy
Pista dataset runtime.  RDKit is intentionally imported only at decode time;
normal corpus loading and training do not acquire an RDKit dependency here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor


_SUPPORTED_BONDS = frozenset({"SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"})


def _indexed_symbols(value: object, *, name: str, first_index: int) -> tuple[str, ...]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{name} must be a non-empty object")
    indices: dict[int, str] = {}
    for symbol, raw_index in value.items():
        if not isinstance(symbol, str) or not symbol:
            raise ValueError(f"{name} symbols must be non-empty strings")
        if isinstance(raw_index, bool) or not isinstance(raw_index, int):
            raise ValueError(f"{name} indices must be integers")
        if raw_index in indices:
            raise ValueError(f"{name} has duplicate index {raw_index}")
        indices[raw_index] = symbol
    expected = set(range(first_index, first_index + len(indices)))
    if set(indices) != expected:
        raise ValueError(f"{name} indices must be contiguous from {first_index}")
    return tuple(indices[index] for index in sorted(indices))


@dataclass(frozen=True, slots=True)
class PistaGraphVocabulary:
    """Explicit categorical vocabulary used to decode Pista endpoint graphs."""

    vocabulary_id: str
    node_symbols: tuple[str, ...]
    edge_names: tuple[str | None, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.vocabulary_id, str) or not self.vocabulary_id.strip():
            raise ValueError("vocabulary_id must be a non-empty string")
        if not self.node_symbols or any(not isinstance(value, str) or not value for value in self.node_symbols):
            raise ValueError("node_symbols must be non-empty symbol strings")
        if not self.edge_names or self.edge_names[0] is not None:
            raise ValueError("edge_names must reserve index zero for no bond")
        if any(value is not None and value not in _SUPPORTED_BONDS for value in self.edge_names):
            raise ValueError("edge_names contains an unsupported bond type")

    @classmethod
    def from_json(cls, path: str | Path, *, vocabulary_id: str) -> "PistaGraphVocabulary":
        """Read one immutable JSON vocabulary entry without rewriting it."""

        source = Path(path).expanduser().resolve(strict=True)
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid graph vocabulary JSON {source}: {error}") from error
        if not isinstance(payload, dict):
            raise ValueError("graph vocabulary JSON must contain an object")
        entry = payload.get(vocabulary_id)
        if not isinstance(entry, dict):
            raise ValueError(f"graph vocabulary {vocabulary_id!r} is absent")
        nodes = _indexed_symbols(entry.get("node_types"), name="node_types", first_index=0)
        edges = _indexed_symbols(entry.get("edge_types"), name="edge_types", first_index=1)
        if any(value not in _SUPPORTED_BONDS for value in edges):
            raise ValueError("edge_types contains an unsupported bond name")
        return cls(
            vocabulary_id=vocabulary_id,
            node_symbols=nodes,
            edge_names=(None, *edges),
        )

    @property
    def node_classes(self) -> int:
        return len(self.node_symbols)

    @property
    def edge_classes(self) -> int:
        return len(self.edge_names)

    def decode(self, X: Tensor, E: Tensor, node_mask: Tensor) -> str | None:
        """Return canonical isomeric SMILES or ``None`` for an invalid graph.

        ``X`` and ``E`` may be hard one-hot states or final probability rows;
        decoding uses their categorical argmax.  Wildcard ``*`` rows represent
        GraphDiT dummy nodes and are omitted together with their incident
        edges.  Padded positions are ignored through ``node_mask``.
        """

        if not isinstance(X, Tensor) or not isinstance(E, Tensor) or not isinstance(node_mask, Tensor):
            raise TypeError("X, E, and node_mask must be tensors")
        if X.ndim != 2 or X.shape[1] != self.node_classes:
            raise ValueError(f"X must have shape [N,{self.node_classes}]")
        nodes = X.shape[0]
        if E.ndim != 3 or E.shape != (nodes, nodes, self.edge_classes):
            raise ValueError(f"E must have shape [N,N,{self.edge_classes}]")
        if node_mask.dtype != torch.bool or node_mask.shape != (nodes,):
            raise ValueError("node_mask must be bool[N]")
        if X.device != E.device or X.device != node_mask.device:
            raise ValueError("X, E, and node_mask must share one device")
        if not X.is_floating_point() or not E.is_floating_point():
            raise TypeError("X and E must be floating categorical tensors")
        if not bool(torch.isfinite(X).all()) or not bool(torch.isfinite(E).all()):
            return None
        try:
            from rdkit import Chem
        except ImportError as error:  # pragma: no cover - dependency-specific
            raise RuntimeError("RDKit is required to decode Pista endpoint graphs") from error

        node_indices = X.argmax(dim=-1).detach().to(device="cpu", dtype=torch.long)
        edge_indices = E.argmax(dim=-1).detach().to(device="cpu", dtype=torch.long)
        valid_nodes = node_mask.detach().to(device="cpu", dtype=torch.bool)
        molecule = Chem.RWMol()
        rdkit_index: dict[int, int] = {}
        try:
            for index in range(nodes):
                if not bool(valid_nodes[index]):
                    continue
                symbol = self.node_symbols[int(node_indices[index])]
                if symbol == "*":
                    continue
                atom = Chem.Atom(symbol)
                rdkit_index[index] = molecule.AddAtom(atom)
            if not rdkit_index:
                return None
            for left in range(nodes):
                left_index = rdkit_index.get(left)
                if left_index is None:
                    continue
                for right in range(left + 1, nodes):
                    right_index = rdkit_index.get(right)
                    if right_index is None:
                        continue
                    edge_name = self.edge_names[int(edge_indices[left, right])]
                    if edge_name is None:
                        continue
                    bond_type = getattr(Chem.BondType, edge_name)
                    molecule.AddBond(left_index, right_index, bond_type)
                    if edge_name == "AROMATIC":
                        molecule.GetAtomWithIdx(left_index).SetIsAromatic(True)
                        molecule.GetAtomWithIdx(right_index).SetIsAromatic(True)
                        molecule.GetBondBetweenAtoms(left_index, right_index).SetIsAromatic(True)
            resolved = molecule.GetMol()
            Chem.SanitizeMol(resolved)
            return Chem.MolToSmiles(resolved, canonical=True, isomericSmiles=True)
        except Exception:
            return None


__all__ = ["PistaGraphVocabulary"]
