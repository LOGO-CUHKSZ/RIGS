"""Strict, stable SMILES canonicalization shared by evaluation methods."""

from __future__ import annotations

from functools import lru_cache
from typing import Any


@lru_cache(maxsize=1)
def _chemistry() -> Any:
    try:
        from rdkit import Chem, RDLogger
    except ImportError as error:  # pragma: no cover - optional dependency boundary
        raise ImportError("evaluation chemistry operations require rdkit") from error
    RDLogger.DisableLog("rdApp.error")
    return Chem


@lru_cache(maxsize=1_000_000)
def _canonical_component(value: str) -> str:
    """Canonicalize one component, tolerating the RDKit fixed-point cycle."""

    Chem = _chemistry()
    molecule = Chem.MolFromSmiles(value)
    if molecule is None:
        raise ValueError(f"cannot parse SMILES component {value!r}")
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    current = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    sequence: list[str] = []
    seen: dict[str, int] = {}
    for _ in range(32):
        if current in seen:
            cycle = sequence[seen[current] :]
            if not cycle:
                raise RuntimeError("empty canonicalization cycle")
            return min(cycle)
        seen[current] = len(sequence)
        sequence.append(current)
        molecule = Chem.MolFromSmiles(current)
        if molecule is None:
            raise ValueError(f"cannot reparse canonical SMILES {value!r}")
        current = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    raise RuntimeError(f"canonicalization did not stabilize for {value!r}")


@lru_cache(maxsize=1_000_000)
def canonical_reaction_smiles(value: str | None) -> str | None:
    """Remove atom maps, canonicalize components, and sort them stably."""

    if not isinstance(value, str) or not value.strip():
        return None
    components = value.strip().split(".")
    if not components or any(not component for component in components):
        return None
    try:
        return ".".join(sorted(_canonical_component(component) for component in components))
    except (RuntimeError, TypeError, ValueError):
        return None


CANONICALIZATION_POLICY = (
    "rdkit_isomeric_remove_atom_maps_component_canonicalize_cycle_min_sort_components"
)
