"""Train-only graph-free batches for projector-only Pista Stage A.

The projector-only Stage-A objective consumes route-semantic embeddings,
canonical supervision, and text-token memories.  It deliberately does *not*
consume dense GraphDiT tensors.  This module reads an immutable metadata
sidecar whose contents were derived once with the normal graph materializer,
then reconstructs exactly the tensors the projector uses without reopening
PyG graph shards on every training micro-batch.

The sidecar is intentionally restricted to the training split.  Validation
and every graph-consuming stage continue through the normal graph runtime.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from retro_guided_generation.data.runtime.canonical import RuntimeGuidanceCondition
from retro_guided_generation.data.runtime.graphdit import (
    GuidanceSurface,
    GuidedStageABatchMaterializer,
    _require_surface,
    _signed_preferences,
    _text_id_for_surface,
    move_guided_stage_a_batch,
)
from retro_guided_generation.models.graphdit import GraphDiTConfig
from retro_guided_generation.models.objectives import GuidedObjectiveBatch
from retro_guided_generation.models.stages import GraphConditionBatch, GuidedStageABatch


PROJECTOR_ONLY_METADATA_SCHEMA = "projector_only_metadata_v1"


class ProjectorOnlyMetadataError(RuntimeError):
    """Raised when a graph-free projector metadata sidecar is invalid."""


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProjectorOnlyMetadataError(f"{name} must be a mapping")
    return value


def _integer_tensor(value: object, name: str, *, length: int | None = None) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.ndim != 1
        or value.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64}
    ):
        raise ProjectorOnlyMetadataError(f"{name} must be a one-dimensional integer tensor")
    result = value.detach().to(device="cpu", dtype=torch.long).contiguous()
    if length is not None and result.numel() != length:
        raise ProjectorOnlyMetadataError(
            f"{name} has length {result.numel()}, expected {length}"
        )
    return result


def _float_tensor(value: object, name: str, *, length: int) -> Tensor:
    if not isinstance(value, Tensor) or value.ndim != 1 or not value.is_floating_point():
        raise ProjectorOnlyMetadataError(f"{name} must be a one-dimensional floating tensor")
    result = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if result.numel() != length or not bool(torch.isfinite(result).all()):
        raise ProjectorOnlyMetadataError(f"{name} must be finite and match candidate count")
    if bool((result < 0.0).any()):
        raise ProjectorOnlyMetadataError(f"{name} must be non-negative")
    return result


def _bool_tensor(value: object, name: str, *, length: int) -> Tensor:
    if not isinstance(value, Tensor) or value.dtype != torch.bool or value.shape != (length,):
        raise ProjectorOnlyMetadataError(f"{name} must be bool[{length}]")
    return value.detach().to(device="cpu", dtype=torch.bool).contiguous()


def _artifact_path(root: Path, manifest: Mapping[str, Any], key: str) -> Path:
    artifacts = _mapping(manifest.get("artifacts"), "metadata manifest.artifacts")
    raw = artifacts.get(key)
    if not isinstance(raw, str) or not raw:
        raise ProjectorOnlyMetadataError(f"metadata manifest lacks artifact {key!r}")
    try:
        path = (root / raw).resolve(strict=True)
        path.relative_to(root)
    except (OSError, ValueError) as error:
        raise ProjectorOnlyMetadataError(
            f"metadata artifact {key!r} is unavailable under {root}"
        ) from error
    return path


def _load_tensors(path: Path) -> Mapping[str, object]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:  # pragma: no cover - external sidecar failure
        raise ProjectorOnlyMetadataError(f"cannot load metadata tensor artifact {path}: {error}") from error
    if not isinstance(payload, Mapping):
        raise ProjectorOnlyMetadataError("metadata tensor artifact must contain a mapping")
    return payload


@dataclass(frozen=True, slots=True)
class ProjectorOnlyMetadataEntry:
    """One exact canonical group slice from a completed metadata sidecar."""

    group_ordinal: int
    semantic_group_uid: str
    route_ids: tuple[str, ...]
    graph_indices: Tensor
    endpoint_weights: Tensor
    max_nodes: int
    listwise_eligible: bool

    @property
    def candidate_count(self) -> int:
        return int(self.graph_indices.numel())


class ProjectorOnlyMetadataStore:
    """Read immutable graph-derived Stage-A metadata without a graph store."""

    def __init__(self, root: str | Path, *, split: str = "train") -> None:
        if split != "train":
            raise ValueError("projector-only graph-free metadata is restricted to the train split")
        self.root = Path(root).expanduser().resolve(strict=True)
        manifest_path = self.root / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ProjectorOnlyMetadataError(
                f"cannot read projector-only metadata manifest {manifest_path}: {error}"
            ) from error
        manifest = _mapping(manifest, "metadata manifest")
        if manifest.get("schema_version") != PROJECTOR_ONLY_METADATA_SCHEMA:
            raise ProjectorOnlyMetadataError("unsupported projector-only metadata schema")
        if manifest.get("status") != "complete":
            raise ProjectorOnlyMetadataError("projector-only metadata sidecar is not complete")
        if manifest.get("split") != split:
            raise ProjectorOnlyMetadataError("metadata split differs from requested split")
        group_count = manifest.get("group_count")
        candidate_count = manifest.get("candidate_count")
        graph_index_entries = manifest.get("graph_index_entries")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in (group_count, candidate_count, graph_index_entries)
        ):
            raise ProjectorOnlyMetadataError(
                "metadata group_count, candidate_count, and graph_index_entries must be positive integers"
            )
        self.split = split
        self.group_count = int(group_count)
        self.candidate_count = int(candidate_count)
        self.graph_index_entries = int(graph_index_entries)
        tensors = _load_tensors(_artifact_path(self.root, manifest, "tensors"))
        group_ptr = _integer_tensor(tensors.get("group_ptr"), "metadata.group_ptr")
        if group_ptr.numel() != self.group_count + 1 or int(group_ptr[0]) != 0:
            raise ProjectorOnlyMetadataError("metadata.group_ptr has an invalid length or origin")
        if bool((group_ptr[1:] <= group_ptr[:-1]).any()) or int(group_ptr[-1]) != self.candidate_count:
            raise ProjectorOnlyMetadataError("metadata.group_ptr must be positive and cover all candidates")
        graph_indices = _integer_tensor(
            tensors.get("graph_indices"), "metadata.graph_indices", length=self.candidate_count
        )
        if bool((graph_indices < 0).any()) or bool((graph_indices >= self.graph_index_entries).any()):
            raise ProjectorOnlyMetadataError("metadata.graph_indices is outside graph-index bounds")
        endpoint_weights = _float_tensor(
            tensors.get("endpoint_weights"), "metadata.endpoint_weights", length=self.candidate_count
        )
        max_nodes = _integer_tensor(
            tensors.get("max_nodes"), "metadata.max_nodes", length=self.group_count
        )
        if bool((max_nodes < 1).any()):
            raise ProjectorOnlyMetadataError("metadata.max_nodes must be positive")
        listwise_eligible = _bool_tensor(
            tensors.get("listwise_eligible"), "metadata.listwise_eligible", length=self.group_count
        )
        product_consistent = _bool_tensor(
            tensors.get("shared_product_consistency"),
            "metadata.shared_product_consistency",
            length=self.group_count,
        )
        if not bool(product_consistent.all()):
            raise ProjectorOnlyMetadataError("metadata contains a group without a shared product condition")
        route_ids_by_group, uid_to_ordinal = self._load_group_rows(
            _artifact_path(self.root, manifest, "groups"), group_ptr=group_ptr
        )
        self._group_ptr = group_ptr
        self._graph_indices = graph_indices
        self._endpoint_weights = endpoint_weights
        self._max_nodes = max_nodes
        self._listwise_eligible = listwise_eligible
        self._route_ids_by_group = route_ids_by_group
        self._uid_to_ordinal = uid_to_ordinal

    def _load_group_rows(
        self,
        path: Path,
        *,
        group_ptr: Tensor,
    ) -> tuple[tuple[tuple[str, ...], ...], dict[str, int]]:
        route_rows: list[tuple[str, ...]] = []
        uid_to_ordinal: dict[str, int] = {}
        try:
            handle = path.open("r", encoding="utf-8")
        except OSError as error:  # pragma: no cover - filesystem failure
            raise ProjectorOnlyMetadataError(f"cannot open metadata group rows {path}: {error}") from error
        with handle:
            for expected_ordinal, line in enumerate(handle):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ProjectorOnlyMetadataError(
                        f"invalid JSON in metadata group row {expected_ordinal + 1}"
                    ) from error
                row = _mapping(row, f"metadata group row {expected_ordinal}")
                if row.get("group_ordinal") != expected_ordinal:
                    raise ProjectorOnlyMetadataError("metadata group rows are not in ordinal order")
                uid = row.get("semantic_group_uid")
                if not isinstance(uid, str) or not uid.strip() or uid != uid.strip():
                    raise ProjectorOnlyMetadataError("metadata semantic_group_uid must be a trimmed string")
                if uid in uid_to_ordinal:
                    raise ProjectorOnlyMetadataError("metadata semantic_group_uid is duplicated")
                raw_routes = row.get("candidate_route_ids")
                if isinstance(raw_routes, str) or not isinstance(raw_routes, list):
                    raise ProjectorOnlyMetadataError("metadata candidate_route_ids must be a list")
                route_ids = tuple(raw_routes)
                if any(
                    not isinstance(route_id, str) or not route_id.strip() or route_id != route_id.strip()
                    for route_id in route_ids
                ):
                    raise ProjectorOnlyMetadataError("metadata candidate_route_ids contains an invalid route")
                expected_count = int(group_ptr[expected_ordinal + 1] - group_ptr[expected_ordinal])
                if len(route_ids) != expected_count:
                    raise ProjectorOnlyMetadataError(
                        "metadata candidate_route_ids differs from its CSR candidate count"
                    )
                uid_to_ordinal[uid] = expected_ordinal
                route_rows.append(route_ids)
        if len(route_rows) != self.group_count:
            raise ProjectorOnlyMetadataError(
                f"metadata has {len(route_rows)} group rows, expected {self.group_count}"
            )
        return tuple(route_rows), uid_to_ordinal

    def entry_for_condition(
        self, condition: RuntimeGuidanceCondition
    ) -> ProjectorOnlyMetadataEntry:
        if not isinstance(condition, RuntimeGuidanceCondition):
            raise TypeError("condition must be RuntimeGuidanceCondition")
        if condition.split != self.split:
            raise ProjectorOnlyMetadataError("runtime condition split differs from metadata split")
        uid = condition.group.semantic_group_uid
        try:
            ordinal = self._uid_to_ordinal[uid]
        except KeyError as error:
            raise KeyError(f"metadata lacks semantic group {uid!r}") from error
        preferences = _signed_preferences(condition.group)
        route_ids = tuple(preference.route_id for preference in preferences)
        expected_route_ids = self._route_ids_by_group[ordinal]
        if route_ids != expected_route_ids:
            raise ProjectorOnlyMetadataError(
                f"metadata candidate ordering differs for semantic group {uid!r}"
            )
        start = int(self._group_ptr[ordinal])
        stop = int(self._group_ptr[ordinal + 1])
        return ProjectorOnlyMetadataEntry(
            group_ordinal=ordinal,
            semantic_group_uid=uid,
            route_ids=route_ids,
            graph_indices=self._graph_indices[start:stop],
            endpoint_weights=self._endpoint_weights[start:stop],
            max_nodes=int(self._max_nodes[ordinal]),
            listwise_eligible=bool(self._listwise_eligible[ordinal]),
        )


class ProjectorOnlyGraphFreeBatchMaterializer(GuidedStageABatchMaterializer):
    """Materialize only tensors consumed by :class:`ProjectorOnlyStageA`.

    A one-node placeholder ``GraphConditionBatch`` retains the established
    outer batch type and formal candidate-count interface.  It is deliberately
    unusable as a graph-training batch; construction is only reachable through
    the projector-only Stage-A configuration guard in ``PistaTrainingSpec``.
    """

    graph_free_projector_only = True

    def __init__(
        self,
        *,
        graphdit: GraphDiTConfig,
        guidance_text_dim: int,
        metadata: ProjectorOnlyMetadataStore,
        text_tokens: Mapping[str, Tensor],
        route_semantic_embedding: Callable[[int], Tensor],
    ) -> None:
        if not isinstance(metadata, ProjectorOnlyMetadataStore):
            raise TypeError("metadata must be ProjectorOnlyMetadataStore")
        if metadata.split != "train":  # defensive if a future loader broadens accepted splits
            raise ValueError("graph-free projector metadata must be train-only")
        if not callable(route_semantic_embedding):
            raise TypeError("route_semantic_embedding must be callable")
        # The inherited helpers provide the exact existing token validation
        # and probability normalization.  ``route_graphs`` is intentionally
        # empty: neither this class nor ProjectorOnlyStageA may fall back to a
        # graph lookup.
        super().__init__(
            graphdit=graphdit,
            guidance_text_dim=guidance_text_dim,
            route_graphs={},
            text_tokens=text_tokens,
            route_semantic_embeddings=None,
            cpu_materialize_workers=1,
        )
        self.metadata = metadata
        self._route_semantic_embedding_by_index = route_semantic_embedding

    def condition_length_cost(
        self,
        condition: RuntimeGuidanceCondition,
        *,
        surface: GuidanceSurface = "true",
    ) -> tuple[int, int]:
        _require_surface(surface)
        entry = self.metadata.entry_for_condition(condition)
        return entry.max_nodes, entry.candidate_count

    def _semantic_embedding(self, graph_index: int) -> Tensor:
        embedding = self._route_semantic_embedding_by_index(graph_index)
        if (
            not isinstance(embedding, Tensor)
            or embedding.ndim != 1
            or embedding.numel() < 1
            or not embedding.is_floating_point()
            or embedding.requires_grad
            or not bool(torch.isfinite(embedding).all())
        ):
            raise ProjectorOnlyMetadataError(
                "route semantic embedding must be a finite detached vector"
            )
        return embedding

    def materialize(
        self,
        conditions: Sequence[RuntimeGuidanceCondition],
        *,
        surface: GuidanceSurface = "true",
        include_shuffled: bool = False,
        device: torch.device | str | None = None,
    ) -> GuidedStageABatch:
        selected_surface = _require_surface(surface)
        if not isinstance(include_shuffled, bool):
            raise TypeError("include_shuffled must be a bool")
        if include_shuffled and selected_surface != "true":
            raise ValueError("shuffled identity controls are only defined for true guidance")
        if not conditions:
            raise ValueError("cannot materialize an empty guidance batch")
        if not all(isinstance(condition, RuntimeGuidanceCondition) for condition in conditions):
            raise TypeError("conditions must contain RuntimeGuidanceCondition values")

        requested_device = torch.device("cpu") if device is None else torch.device(device)
        storage_device = torch.device("cpu")
        text_rows: list[Tensor] = []
        shuffled_text_rows: list[Tensor] | None = [] if include_shuffled else None
        teacher_rows: list[Tensor] = []
        endpoint_rows: list[Tensor] = []
        semantic_rows: list[Tensor] = []
        labels: list[int] = []
        group_sizes: list[int] = []
        branches: list[str] = []
        listwise_eligible: list[bool] = []

        for condition in conditions:
            preferences = _signed_preferences(condition.group)
            entry = self.metadata.entry_for_condition(condition)
            if entry.candidate_count != len(preferences):  # pragma: no cover - entry validates this
                raise ProjectorOnlyMetadataError("metadata candidate count differs from canonical group")
            text_rows.append(self._text_tokens(_text_id_for_surface(condition, selected_surface)))
            if shuffled_text_rows is not None:
                if condition.shuffled is None:
                    raise ValueError(
                        "projector identity training requires a resolved different-axis shuffled guidance"
                    )
                shuffled_text_rows.append(self._text_tokens(condition.shuffled.donor_text_id))
            teacher_rows.append(
                self._normalized(
                    [preference.teacher_probability for preference in preferences],
                    name="selected teacher probabilities",
                )
            )
            endpoint_rows.append(
                self._normalized(
                    entry.endpoint_weights.tolist(),
                    name="selected endpoint weights",
                    fallback_uniform=True,
                )
            )
            semantic_rows.extend(
                self._semantic_embedding(int(graph_index)) for graph_index in entry.graph_indices.tolist()
            )
            labels.extend(1 if preference.label == 1 else 0 for preference in preferences)
            group_sizes.append(entry.candidate_count)
            branches.append(condition.group.branch)
            listwise_eligible.append(entry.listwise_eligible)

        candidates = sum(group_sizes)
        groups = len(group_sizes)
        max_tokens = max(row.shape[0] for row in text_rows)
        graph_dtype = torch.float32
        guidance_tokens = torch.zeros(
            (groups, max_tokens, self.guidance_text_dim),
            dtype=graph_dtype,
            device=storage_device,
        )
        guidance_mask = torch.zeros((groups, max_tokens), dtype=torch.bool, device=storage_device)
        for index, row in enumerate(text_rows):
            length = row.shape[0]
            guidance_tokens[index, :length] = row.to(device=storage_device, dtype=graph_dtype)
            guidance_mask[index, :length] = True

        shuffled_guidance_tokens: Tensor | None = None
        shuffled_guidance_mask: Tensor | None = None
        if shuffled_text_rows is not None:
            max_shuffled_tokens = max(row.shape[0] for row in shuffled_text_rows)
            shuffled_guidance_tokens = torch.zeros(
                (groups, max_shuffled_tokens, self.guidance_text_dim),
                dtype=graph_dtype,
                device=storage_device,
            )
            shuffled_guidance_mask = torch.zeros(
                (groups, max_shuffled_tokens), dtype=torch.bool, device=storage_device
            )
            for index, row in enumerate(shuffled_text_rows):
                length = row.shape[0]
                shuffled_guidance_tokens[index, :length] = row.to(
                    device=storage_device, dtype=graph_dtype
                )
                shuffled_guidance_mask[index, :length] = True

        semantic_width = semantic_rows[0].numel()
        if any(row.numel() != semantic_width for row in semantic_rows):
            raise ProjectorOnlyMetadataError("route semantic embeddings disagree on width")
        route_semantic_candidate = torch.stack(
            [row.to(dtype=torch.float32) for row in semantic_rows]
        ).to(device=storage_device, dtype=torch.float32)
        sizes = torch.tensor(group_sizes, dtype=torch.long, device=storage_device)
        group_ptr = torch.cat((torch.zeros(1, dtype=torch.long, device=storage_device), sizes.cumsum(0)))
        group_index = torch.repeat_interleave(
            torch.arange(groups, dtype=torch.long, device=storage_device),
            sizes,
            output_size=candidates,
        )
        supervision = GuidedObjectiveBatch(
            group_ptr=group_ptr,
            endpoint_probability=torch.cat(endpoint_rows).to(device=storage_device),
            teacher_probability=torch.cat(teacher_rows).to(device=storage_device),
            structural_label=torch.tensor(labels, dtype=torch.long, device=storage_device),
            listwise_eligible=torch.tensor(listwise_eligible, dtype=torch.bool, device=storage_device),
            guidance_branch=tuple(branches),
            group_index=group_index,
        )
        # Keep the historical outer batch shape API without allocating the
        # real [candidate,max_nodes,max_nodes] tensors.  ProjectorOnlyStageA
        # neither reads nor validates this placeholder.
        p_x = torch.zeros(
            (candidates, 1, self.graphdit.node_classes), dtype=graph_dtype, device=storage_device
        )
        p_e = torch.zeros(
            (candidates, 1, 1, self.graphdit.edge_classes), dtype=graph_dtype, device=storage_device
        )
        p_x[..., 0] = 1.0
        p_e[..., 0] = 1.0
        cpu_batch = GuidedStageABatch(
            graphs=GraphConditionBatch(
                p_X=p_x,
                p_E=p_e,
                r_X=p_x.clone(),
                r_E=p_e.clone(),
                node_mask=torch.ones((candidates, 1), dtype=torch.bool, device=storage_device),
                p_mask=torch.zeros((candidates, 1), dtype=torch.bool, device=storage_device),
                cond=torch.zeros(
                    (candidates, self.graphdit.condition_dim),
                    dtype=graph_dtype,
                    device=storage_device,
                ),
            ),
            supervision=supervision,
            group_index=group_index,
            guidance_tokens_group=guidance_tokens,
            guidance_token_mask_group=guidance_mask,
            route_semantic_candidate=route_semantic_candidate,
            shuffled_guidance_tokens_group=shuffled_guidance_tokens,
            shuffled_guidance_token_mask_group=shuffled_guidance_mask,
        )
        return move_guided_stage_a_batch(cpu_batch, requested_device)


__all__ = [
    "PROJECTOR_ONLY_METADATA_SCHEMA",
    "ProjectorOnlyGraphFreeBatchMaterializer",
    "ProjectorOnlyMetadataEntry",
    "ProjectorOnlyMetadataError",
    "ProjectorOnlyMetadataStore",
]
