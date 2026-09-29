"""Minimal canonical-guidance to adapter-only Stage A tensor materialization.

This module deliberately does not own molecular graph conversion, text-model
inference, a cache format, or a trainer.  Callers supply already encoded route
graphs and text token memories, while this module preserves the canonical
guidance relation selected by :mod:`canonical` and creates the one batch
contract consumed by :class:`AdapterOnlyStageA`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, fields, replace
from typing import Literal, TypeAlias

import torch
from torch import Tensor

from retro_guided_generation.data.runtime.canonical import RuntimeGuidanceCondition
from retro_guided_generation.guidance.schema import CandidatePreference, SemanticGroup
from retro_guided_generation.models.graphdit import GraphDiTConfig
from retro_guided_generation.models.objectives import GuidedObjectiveBatch
from retro_guided_generation.models.stages import GraphConditionBatch, GuidedStageABatch


GuidanceSurface: TypeAlias = Literal["true", "opposite", "shuffled"]
RouteGraphKey: TypeAlias = tuple[str, str, str]


@dataclass(frozen=True, slots=True)
class RouteGraphRecord:
    """One already-dense product/route graph pair for a canonical route.

    All graph tensors describe one route candidate, with an unbatched leading
    node dimension.  ``endpoint_weight`` is the non-negative candidate prior
    used to form the objective's within-group endpoint distribution.  Its
    default gives an explicit uniform prior when an upstream dataset has no
    candidate-selection score.
    """

    case_id: str
    candidate_pool_id: str
    route_id: str
    p_X: Tensor
    p_E: Tensor
    r_X: Tensor
    r_E: Tensor
    node_mask: Tensor
    p_mask: Tensor
    cond: Tensor
    endpoint_weight: float = 1.0

    def __post_init__(self) -> None:
        for name in ("case_id", "candidate_pool_id", "route_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or value != value.strip():
                raise ValueError(f"{name} must be a non-empty trimmed string")
        weight = float(self.endpoint_weight)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError("endpoint_weight must be finite and non-negative")

        p_x, p_e, r_x, r_e = self.p_X, self.p_E, self.r_X, self.r_E
        if not all(isinstance(value, Tensor) for value in (p_x, p_e, r_x, r_e)):
            raise TypeError("graph fields must be tensors")
        if not all(value.is_floating_point() for value in (p_x, p_e, r_x, r_e)):
            raise TypeError("graph fields must use floating-point one-hot tensors")
        if p_x.ndim != 2 or p_x.shape != r_x.shape or p_x.shape[0] < 1:
            raise ValueError("p_X and r_X must share a non-empty [N,Kx] shape")
        if p_e.ndim != 3 or p_e.shape != r_e.shape:
            raise ValueError("p_E and r_E must share a [N,N,Ke] shape")
        nodes = p_x.shape[0]
        if p_e.shape[:2] != (nodes, nodes) or p_e.shape[2] < 1:
            raise ValueError("node and edge graph dimensions are inconsistent")
        if self.node_mask.dtype != torch.bool or self.node_mask.shape != (nodes,):
            raise TypeError("node_mask must be bool[N]")
        if self.p_mask.dtype != torch.bool or self.p_mask.shape != (nodes,):
            raise TypeError("p_mask must be bool[N]")
        if not self.node_mask.any():
            raise ValueError("node_mask must contain at least one valid node")
        if self.cond.ndim != 1 or not self.cond.is_floating_point():
            raise TypeError("cond must be a floating-point [C] tensor")
        tensors = (
            p_x,
            p_e,
            r_x,
            r_e,
            self.node_mask,
            self.p_mask,
            self.cond,
        )
        if any(value.device != p_x.device for value in tensors):
            raise ValueError("all RouteGraphRecord tensors must share one device")
        if any(value.requires_grad for value in tensors):
            raise ValueError("RouteGraphRecord tensors must not require gradients")
        if not all(torch.isfinite(value).all() for value in (p_x, p_e, r_x, r_e, self.cond)):
            raise ValueError("RouteGraphRecord contains NaN or Inf")
        if not torch.equal(p_e, p_e.transpose(0, 1)):
            raise ValueError("p_E must be symmetric")
        if not torch.equal(r_e, r_e.transpose(0, 1)):
            raise ValueError("r_E must be symmetric")

    @property
    def key(self) -> RouteGraphKey:
        return (self.case_id, self.candidate_pool_id, self.route_id)


def _require_surface(surface: str) -> GuidanceSurface:
    if surface not in {"true", "opposite", "shuffled"}:
        raise ValueError("surface must be 'true', 'opposite', or 'shuffled'")
    return surface  # type: ignore[return-value]


def _text_id_for_surface(
    condition: RuntimeGuidanceCondition,
    surface: GuidanceSurface,
) -> str:
    if surface == "true":
        return condition.true_text.text_id
    if surface == "opposite":
        return condition.opposite_text.text_id
    if condition.shuffled is None:
        raise ValueError("shuffled surface requires a resolved shuffled donor")
    return condition.shuffled.donor_text_id


def _signed_preferences(group: SemanticGroup) -> tuple[CandidatePreference, ...]:
    """Keep only the explicitly signed binary contrast endpoints.

    Canonical records retain neutral and unavailable routes for auditability.
    They must not enter the binary Stage A loss as implicit negatives.  The
    selected teacher mass is renormalized over exactly the retained endpoints.
    """

    selected = tuple(
        preference
        for preference in group.candidate_preferences
        if preference.eligible and preference.label in {-1, 1}
    )
    labels = {preference.label for preference in selected}
    if labels != {-1, 1}:
        raise ValueError(
            f"semantic group {group.semantic_group_uid!r} has no complete signed contrast"
        )
    if any(preference.teacher_probability < 0.0 for preference in selected):
        raise ValueError("selected teacher probabilities must be non-negative")
    if sum(preference.teacher_probability for preference in selected) <= 0.0:
        raise ValueError("selected teacher probabilities must contain positive mass")
    return selected


class GuidedStageABatchMaterializer:
    """Materialize a small collection of runtime-resolved guidance groups.

    The input records may come from any graph converter or text encoder.  This
    class does not persist those records and therefore introduces no cache
    identity, hash, or dataset-specific convention.  A trainer selects the
    runtime conditions and invokes :meth:`materialize` for each microbatch.
    """

    def __init__(
        self,
        *,
        graphdit: GraphDiTConfig,
        guidance_text_dim: int,
        route_graphs: Mapping[RouteGraphKey, RouteGraphRecord],
        text_tokens: Mapping[str, Tensor],
        route_semantic_embeddings: Mapping[RouteGraphKey, Tensor] | None = None,
        cpu_materialize_workers: int = 1,
    ) -> None:
        if not isinstance(graphdit, GraphDiTConfig):
            raise TypeError("graphdit must be GraphDiTConfig")
        if (
            isinstance(guidance_text_dim, bool)
            or not isinstance(guidance_text_dim, int)
            or guidance_text_dim < 1
        ):
            raise ValueError("guidance_text_dim must be a positive integer")
        if (
            isinstance(cpu_materialize_workers, bool)
            or not isinstance(cpu_materialize_workers, int)
            or cpu_materialize_workers < 1
        ):
            raise ValueError("cpu_materialize_workers must be a positive integer")
        self.graphdit = graphdit
        self.guidance_text_dim = guidance_text_dim
        self.route_graphs = route_graphs
        self.text_tokens = text_tokens
        self.route_semantic_embeddings = route_semantic_embeddings
        self.cpu_materialize_workers = cpu_materialize_workers

    def _route_graph(self, group: SemanticGroup, route_id: str) -> RouteGraphRecord:
        key: RouteGraphKey = (group.case_id, group.candidate_pool_id, route_id)
        try:
            record = self.route_graphs[key]
        except KeyError as error:
            raise KeyError(f"missing graph record for canonical route {key!r}") from error
        if not isinstance(record, RouteGraphRecord):
            raise TypeError(f"graph record {key!r} must be RouteGraphRecord")
        if record.key != key:
            raise ValueError(f"graph record identity differs for canonical route {key!r}")
        if record.p_X.shape[1] != self.graphdit.node_classes:
            raise ValueError("graph record node class width differs from GraphDiTConfig")
        if record.p_E.shape[2] != self.graphdit.edge_classes:
            raise ValueError("graph record edge class width differs from GraphDiTConfig")
        if record.cond.shape != (self.graphdit.condition_dim,):
            raise ValueError("graph record condition width differs from GraphDiTConfig")
        # Legacy Pista endpoint tensors are padded to the raw atom cap plus
        # the model's configured dummy-node budget.  ``max_n_len`` alone is
        # therefore not the dense tensor limit when ``n_dummy`` is nonzero.
        # This preserves the established n_dummy=20 representation; it does
        # not expand dummy capacity or admit a new graph encoding.
        max_nodes = self.graphdit.max_n_len + self.graphdit.n_dummy
        if record.p_X.shape[0] > max_nodes:
            raise ValueError("graph record exceeds GraphDiTConfig node capacity")
        return record

    def _text_tokens(self, text_id: str) -> Tensor:
        try:
            tokens = self.text_tokens[text_id]
        except KeyError as error:
            raise KeyError(f"missing token memory for text {text_id!r}") from error
        if (
            not isinstance(tokens, Tensor)
            or tokens.ndim != 2
            or tokens.shape[0] < 1
            or tokens.shape[1] != self.guidance_text_dim
            or not tokens.is_floating_point()
        ):
            raise ValueError(
                "text token memory must be a non-empty floating "
                f"[L,{self.guidance_text_dim}] tensor"
            )
        if tokens.requires_grad or not torch.isfinite(tokens).all():
            raise ValueError("text token memory must be finite and detached")
        return tokens

    def condition_length_cost(
        self,
        condition: RuntimeGuidanceCondition,
        *,
        surface: GuidanceSurface = "true",
    ) -> tuple[int, int]:
        """Return ``(max_nodes, signed_candidate_count)`` for one group.

        The graph records are resolved through the same cache-backed path as
        materialization.  The provider uses this bounded metadata pass to
        keep groups with incompatible node/candidate costs out of one padded
        batch; the actual tensors are still materialized exactly once.
        """

        _require_surface(surface)
        if not isinstance(condition, RuntimeGuidanceCondition):
            raise TypeError("condition must be RuntimeGuidanceCondition")
        preferences = _signed_preferences(condition.group)
        # Pista's endpoint store can expose node counts directly from its
        # shard/index metadata.  Cost-balanced batching calls this method for
        # a global window on every rank; avoid densifying every candidate graph
        # in that metadata-only pass.  Full graph materialization and product
        # consistency checks still happen below in ``materialize`` for the
        # selected rank lane.
        node_count = getattr(self.route_graphs, "node_count", None)
        if callable(node_count):
            node_counts = tuple(
                int(
                    node_count(
                        (condition.group.case_id, condition.group.candidate_pool_id, preference.route_id)
                    )
                )
                for preference in preferences
            )
            if not node_counts or any(value < 1 for value in node_counts):
                raise ValueError("route graph node counts must be positive")
            return max(node_counts), len(preferences)
        records = tuple(
            self._route_graph(condition.group, preference.route_id)
            for preference in preferences
        )
        self._assert_shared_product(condition.group, records)
        return max(record.p_X.shape[0] for record in records), len(records)

    def _route_semantic_embedding(self, record: RouteGraphRecord) -> Tensor:
        if self.route_semantic_embeddings is None:
            raise RuntimeError("route semantic embeddings are not configured")
        try:
            embedding = self.route_semantic_embeddings[record.key]
        except KeyError as error:
            raise KeyError(f"missing semantic route embedding for {record.key!r}") from error
        if (
            not isinstance(embedding, Tensor)
            or embedding.ndim != 1
            or embedding.numel() < 1
            or not embedding.is_floating_point()
            or embedding.requires_grad
            or not bool(torch.isfinite(embedding).all())
        ):
            raise ValueError("route semantic embedding must be a finite detached vector")
        return embedding

    @staticmethod
    def _assert_shared_product(
        group: SemanticGroup,
        records: Sequence[RouteGraphRecord],
    ) -> None:
        reference = records[0]
        for record in records[1:]:
            for name in ("p_X", "p_E", "node_mask", "p_mask", "cond"):
                if not torch.equal(getattr(record, name), getattr(reference, name)):
                    raise ValueError(
                        f"product condition {name} differs inside semantic group "
                        f"{group.semantic_group_uid!r}"
                    )

    @staticmethod
    def _runtime_preference_eligible(records: Sequence[RouteGraphRecord]) -> bool:
        """Match the DFM preference-mask eligibility for one candidate group.

        Annotation availability and a signed preference contrast are not enough
        to guarantee a graph-ranking signal: the retained reactant endpoints
        can still be identical at every valid node and edge.  The objective
        derives listwise eligibility from exactly that endpoint disagreement.
        Compute the cached flag from the same target IDs and upper-triangular
        valid-edge convention, so the materialized supervision remains an
        honest description of the runtime batch.
        """

        if not records:
            raise ValueError("cannot derive preference eligibility from an empty group")
        reference = records[0]
        node_mask = reference.node_mask
        valid_upper_edge = (
            node_mask.unsqueeze(-1)
            & node_mask.unsqueeze(-2)
            & torch.ones(
                (node_mask.numel(), node_mask.numel()), dtype=torch.bool, device=node_mask.device
            ).triu(diagonal=1)
        )
        reference_node_ids = reference.r_X.argmax(dim=-1)
        reference_edge_ids = reference.r_E.argmax(dim=-1)
        for record in records[1:]:
            node_disagreement = (record.r_X.argmax(dim=-1) != reference_node_ids) & node_mask
            if bool(node_disagreement.any()):
                return True
            edge_disagreement = (record.r_E.argmax(dim=-1) != reference_edge_ids) & valid_upper_edge
            if bool(edge_disagreement.any()):
                return True
        return False

    @staticmethod
    def _normalized(
        values: Sequence[float],
        *,
        name: str,
        fallback_uniform: bool = False,
    ) -> Tensor:
        result = torch.tensor(values, dtype=torch.float32)
        total = result.sum()
        if not bool(torch.isfinite(result).all()) or not bool((result >= 0.0).all()):
            raise ValueError(f"{name} must be finite and non-negative")
        if not bool(total > 0.0):
            if fallback_uniform:
                return torch.full_like(result, 1.0 / result.numel())
            raise ValueError(f"{name} must contain positive total mass")
        return result / total

    def materialize(
        self,
        conditions: Sequence[RuntimeGuidanceCondition],
        *,
        surface: GuidanceSurface = "true",
        include_shuffled: bool = False,
        device: torch.device | str | None = None,
    ) -> GuidedStageABatch:
        """Create one contiguous adapter-only Stage A batch.

        ``conditions`` must already have been resolved by
        :class:`GuidanceRuntimeIndex`; in particular, selecting ``shuffled``
        uses its true runtime donor instead of a frozen recipient mapping.
        """

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
        # Assemble the complete dense batch in host memory first.  The old
        # path copied every candidate tensor independently to CUDA, creating
        # hundreds of tiny pageable transfers per step.  The completed batch
        # is pinned and moved by ``move_guided_stage_a_batch`` below.
        storage_device = torch.device("cpu")
        per_group_records: list[tuple[RouteGraphRecord, ...]] = []
        text_rows: list[Tensor] = []
        shuffled_text_rows: list[Tensor] | None = [] if include_shuffled else None
        teacher_rows: list[Tensor] = []
        endpoint_rows: list[Tensor] = []
        labels: list[int] = []
        group_sizes: list[int] = []
        branches: list[str] = []
        runtime_listwise_eligibility: list[bool] = []
        semantic_rows: list[Tensor] | None = (
            [] if self.route_semantic_embeddings is not None else None
        )

        # The Pista route store can resolve a whole batch in graph-shard
        # order.  The returned mapping keeps the original candidate order,
        # while the store avoids reopening the same GPFS shard for every
        # randomly ordered route in this batch.  Generic Mapping providers
        # continue to use the original lazy lookup path.
        preference_rows = tuple(
            _signed_preferences(condition.group) for condition in conditions
        )
        prefetched_records: Mapping[RouteGraphKey, RouteGraphRecord] | None = None
        prefetch_records = getattr(self.route_graphs, "prefetch_records", None)
        if callable(prefetch_records):
            keys = tuple(
                (
                    condition.group.case_id,
                    condition.group.candidate_pool_id,
                    preference.route_id,
                )
                for condition, preferences in zip(
                    conditions, preference_rows, strict=True
                )
                for preference in preferences
            )
            prefetched_records = prefetch_records(keys)

        def resolve_group_records(
            condition: RuntimeGuidanceCondition,
            preferences: tuple[CandidatePreference, ...],
        ) -> tuple[tuple[RouteGraphRecord, ...], bool]:
            group = condition.group
            records = tuple(
                (
                    self._route_graph(group, preference.route_id)
                    if prefetched_records is None
                    else prefetched_records[
                        (group.case_id, group.candidate_pool_id, preference.route_id)
                    ]
                )
                for preference in preferences
            )
            self._assert_shared_product(group, records)
            return records, self._runtime_preference_eligible(records)

        # Pista has already materialized all selected records through its
        # batch-local prefetch path above.  Product consistency and runtime
        # eligibility are then independent per group, so perform their tensor
        # comparisons concurrently without allowing a generic Mapping's lazy
        # lookup/cache implementation to be called from several threads.
        if (
            prefetched_records is not None
            and self.cpu_materialize_workers > 1
            and len(conditions) > 1
        ):
            workers = min(self.cpu_materialize_workers, len(conditions))
            with ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix="guided-group-check",
            ) as executor:
                futures = tuple(
                    executor.submit(resolve_group_records, condition, preferences)
                    for condition, preferences in zip(conditions, preference_rows, strict=True)
                )
                group_record_results = tuple(future.result() for future in futures)
        else:
            group_record_results = tuple(
                resolve_group_records(condition, preferences)
                for condition, preferences in zip(conditions, preference_rows, strict=True)
            )

        for condition, preferences, (records, runtime_eligible) in zip(
            conditions,
            preference_rows,
            group_record_results,
            strict=True,
        ):
            group = condition.group
            per_group_records.append(records)
            runtime_listwise_eligibility.append(runtime_eligible)
            if semantic_rows is not None:
                semantic_rows.extend(self._route_semantic_embedding(record) for record in records)
            text_rows.append(self._text_tokens(_text_id_for_surface(condition, selected_surface)))
            if shuffled_text_rows is not None:
                if condition.shuffled is None:
                    raise ValueError(
                        "projector identity training requires a resolved different-axis shuffled guidance"
                    )
                shuffled_text_rows.append(
                    self._text_tokens(condition.shuffled.donor_text_id)
                )
            teacher_rows.append(
                self._normalized(
                    [preference.teacher_probability for preference in preferences],
                    name="selected teacher probabilities",
                )
            )
            endpoint_rows.append(
                self._normalized(
                    [record.endpoint_weight for record in records],
                    name="selected endpoint weights",
                    fallback_uniform=True,
                )
            )
            labels.extend(1 if preference.label == 1 else 0 for preference in preferences)
            group_sizes.append(len(preferences))
            branches.append(group.branch)

        flat_records = tuple(record for records in per_group_records for record in records)
        candidates = len(flat_records)
        groups = len(per_group_records)
        max_nodes = max(record.p_X.shape[0] for record in flat_records)
        max_tokens = max(row.shape[0] for row in text_rows)
        graph_dtype = torch.float32

        p_x = torch.zeros(
            (candidates, max_nodes, self.graphdit.node_classes),
            dtype=graph_dtype,
            device=storage_device,
        )
        r_x = torch.zeros_like(p_x)
        p_e = torch.zeros(
            (candidates, max_nodes, max_nodes, self.graphdit.edge_classes),
            dtype=graph_dtype,
            device=storage_device,
        )
        r_e = torch.zeros_like(p_e)
        node_mask = torch.zeros((candidates, max_nodes), dtype=torch.bool, device=storage_device)
        p_mask = torch.zeros_like(node_mask)
        cond = torch.zeros(
            (candidates, self.graphdit.condition_dim), dtype=graph_dtype, device=storage_device
        )

        def copy_graph_records(start: int, stop: int) -> None:
            """Copy a disjoint candidate range into the finished CPU batch.

            The output tensors are preallocated and every worker exclusively
            owns its leading candidate indices.  Therefore the resulting
            tensor order is exactly the serial order while PyTorch can run
            the CPU copies concurrently outside the Python GIL.
            """

            for index in range(start, stop):
                record = flat_records[index]
                nodes = record.p_X.shape[0]
                p_x[index, :nodes] = record.p_X.to(device=storage_device, dtype=graph_dtype)
                r_x[index, :nodes] = record.r_X.to(device=storage_device, dtype=graph_dtype)
                p_e[index, :nodes, :nodes] = record.p_E.to(device=storage_device, dtype=graph_dtype)
                r_e[index, :nodes, :nodes] = record.r_E.to(device=storage_device, dtype=graph_dtype)
                node_mask[index, :nodes] = record.node_mask.to(device=storage_device)
                p_mask[index, :nodes] = record.p_mask.to(device=storage_device)
                cond[index] = record.cond.to(device=storage_device, dtype=graph_dtype)

        workers = min(self.cpu_materialize_workers, candidates)
        if workers == 1:
            copy_graph_records(0, candidates)
        else:
            chunk_size = math.ceil(candidates / workers)
            ranges = tuple(
                (start, min(start + chunk_size, candidates))
                for start in range(0, candidates, chunk_size)
            )
            with ThreadPoolExecutor(
                max_workers=len(ranges),
                thread_name_prefix="guided-batch-copy",
            ) as executor:
                futures = tuple(
                    executor.submit(copy_graph_records, start, stop)
                    for start, stop in ranges
                )
                for future in futures:
                    future.result()

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
            if len(shuffled_text_rows) != groups:
                raise AssertionError("shuffled guidance rows disagree with materialized groups")
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

        route_semantic_candidate: Tensor | None = None
        if semantic_rows is not None:
            semantic_width = semantic_rows[0].numel()
            if any(row.numel() != semantic_width for row in semantic_rows):
                raise ValueError("route semantic embeddings disagree on width")
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
            listwise_eligible=torch.tensor(
                runtime_listwise_eligibility,
                dtype=torch.bool,
                device=storage_device,
            ),
            guidance_branch=tuple(branches),
            group_index=group_index,
        )
        cpu_batch = GuidedStageABatch(
            graphs=GraphConditionBatch(
                p_X=p_x,
                p_E=p_e,
                r_X=r_x,
                r_E=r_e,
                node_mask=node_mask,
                p_mask=p_mask,
                cond=cond,
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


def _move_batch_tensor(value: Tensor, device: torch.device) -> Tensor:
    if value.device == device:
        return value
    if device.type == "cuda" and value.device.type == "cpu":
        # Pin only the finished batch, not every source graph.  This keeps
        # the graph cache small while allowing one non-blocking transfer per
        # dense field from the host staging area.
        if value.numel() > 0 and not value.is_pinned():
            value = value.pin_memory()
        return value.to(device=device, non_blocking=True)
    return value.to(device=device)


def _move_dataclass_tensors(value: object, device: torch.device) -> object:
    updates: dict[str, object] = {}
    for field in fields(value):
        field_value = getattr(value, field.name)
        if isinstance(field_value, Tensor):
            updates[field.name] = _move_batch_tensor(field_value, device)
    return replace(value, **updates)


def move_guided_stage_a_batch(batch: GuidedStageABatch, device: torch.device | str) -> GuidedStageABatch:
    """Move a fully materialized batch, preserving its nested dataclasses."""

    if not isinstance(batch, GuidedStageABatch):
        raise TypeError("batch must be GuidedStageABatch")
    target = torch.device(device)
    graphs = _move_dataclass_tensors(batch.graphs, target)
    supervision = _move_dataclass_tensors(batch.supervision, target)
    return replace(
        batch,
        graphs=graphs,
        supervision=supervision,
        group_index=_move_batch_tensor(batch.group_index, target),
        guidance_tokens_group=_move_batch_tensor(batch.guidance_tokens_group, target),
        guidance_token_mask_group=_move_batch_tensor(batch.guidance_token_mask_group, target),
        route_semantic_candidate=(
            None
            if batch.route_semantic_candidate is None
            else _move_batch_tensor(batch.route_semantic_candidate, target)
        ),
        shuffled_guidance_tokens_group=(
            None
            if batch.shuffled_guidance_tokens_group is None
            else _move_batch_tensor(batch.shuffled_guidance_tokens_group, target)
        ),
        shuffled_guidance_token_mask_group=(
            None
            if batch.shuffled_guidance_token_mask_group is None
            else _move_batch_tensor(batch.shuffled_guidance_token_mask_group, target)
        ),
        # Virtual-lane offsets are tiny CPU control metadata. They are never
        # consumed by a CUDA kernel and remain on CPU to avoid scalar syncs.
        virtual_microbatch_group_ptr=batch.virtual_microbatch_group_ptr,
    )


__all__ = [
    "GuidanceSurface",
    "GuidedStageABatchMaterializer",
    "RouteGraphKey",
    "RouteGraphRecord",
    "move_guided_stage_a_batch",
]
