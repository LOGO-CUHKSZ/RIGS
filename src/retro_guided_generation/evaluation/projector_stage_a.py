"""Exact frozen-validation evidence for projector-only Stage A.

The formal trainer intentionally owns only training/recovery.  This module
evaluates each registered candidate checkpoint over every semantic group in a
split exactly once, with canonical evaluation wording and the already-defined
different-axis shuffled donor.  It never calls the training sampler, changes
draw wording, or uses test data for checkpoint selection.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor

from retro_guided_generation.data.runtime.pista import PistaStageABatchProvider
from retro_guided_generation.models.stages import (
    ProjectorOnlyStageA,
    build_projector_only_stage_a,
)
from retro_guided_generation.training.formal import (
    DistributedTrainingContext,
    initialize_distributed_training,
)
from retro_guided_generation.training.pista import (
    PistaTrainingSpec,
    _provider,
    load_pista_training_spec,
)


PROJECTOR_STAGE_A_VALIDATION_SCHEMA = "retro_guided_generation.projector_stage_a_validation.v1"


def _validated_evaluation_selection_mask(
    *,
    semantic_selected_group_mask: Tensor,
    identity_selected_group_mask: Tensor,
    identity_weight: float,
    expected_groups: int,
    scope: str,
) -> Tensor:
    """Validate semantic eligibility without reviving a disabled identity objective."""

    selected = semantic_selected_group_mask.detach().to(device="cpu", dtype=torch.bool)
    identity_selected = identity_selected_group_mask.detach().to(
        device="cpu", dtype=torch.bool
    )
    expected_shape = (expected_groups,)
    if selected.shape != expected_shape or identity_selected.shape != expected_shape:
        raise RuntimeError(f"{scope} selection masks have an unexpected shape")
    if float(identity_weight) == 0.0:
        if bool(identity_selected.any()):
            raise RuntimeError(f"{scope} identity-disabled selection mask is not empty")
    elif not torch.equal(selected, identity_selected):
        raise RuntimeError(f"{scope} selection masks disagree")
    return selected


def _atomic_torch_save(path: Path, payload: Mapping[str, object]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite validation artifact: {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
        os.chmod(path, 0o664)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite validation artifact: {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
        os.chmod(path, 0o664)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _require_checkpoint_state(path: Path) -> Mapping[str, Tensor]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older torch compatibility
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"Stage-A candidate checkpoint is not a mapping: {path}")
    state = payload.get("state_dict")
    if not isinstance(state, Mapping):
        raise RuntimeError(f"Stage-A candidate checkpoint has no state_dict: {path}")
    typed = {name: value for name, value in state.items() if isinstance(name, str) and isinstance(value, Tensor)}
    if len(typed) != len(state):
        raise RuntimeError(f"Stage-A candidate state contains a non-tensor entry: {path}")
    return typed


def _parse_checkpoint_argument(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("--checkpoint must use NAME=/absolute/path.ckpt")
    name, raw_path = value.split("=", 1)
    name = name.strip()
    if not name or any(character.isspace() for character in name):
        raise ValueError("checkpoint name must be a non-empty whitespace-free label")
    path = Path(raw_path).expanduser().resolve(strict=True)
    return name, path


def _candidate_models(
    spec: PistaTrainingSpec,
    checkpoints: Mapping[str, Path],
    *,
    device: torch.device,
) -> dict[str, ProjectorOnlyStageA]:
    if spec.stage != "stage-a" or spec.projector_only_stage_a is None:
        raise ValueError("projector validation requires a projector-only Stage-A configuration")
    models: dict[str, ProjectorOnlyStageA] = {}
    for name, path in checkpoints.items():
        model, _report = build_projector_only_stage_a(spec.projector_only_stage_a)
        incompatible = model.load_state_dict(_require_checkpoint_state(path), strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:  # strict=True already raises
            raise RuntimeError(f"Stage-A candidate did not load strictly: {path}")
        model.to(device)
        model.eval()
        models[name] = model
    return models


def _exact_validation_batches(
    provider: PistaStageABatchProvider,
    *,
    context: DistributedTrainingContext,
) -> Iterator[tuple[tuple[int, ...], object]]:
    """Yield one disjoint exact-cover lane without calling ``sample_true``.

    Each rank receives every ``world_size``-th semantic-group ordinal.  The
    packing limit is the registered Pista batch contract, but it is merely a
    transport optimization: every selected condition was already fixed by its
    split ordinal, canonical evaluation wording, and evaluation shuffle seed.
    """

    runtime = provider.runtime
    materializer = provider.materializer
    current: list[tuple[int, object, int, int]] = []
    candidate_count = 0
    max_nodes = 0
    for ordinal in range(context.rank, runtime.group_count(provider.split), context.world_size):
        condition = runtime.condition(
            split=provider.split,
            group_ordinal=ordinal,
            epoch=0,
            draw_index=ordinal,
            include_shuffled=provider.include_shuffled,
            evaluation=True,
        )
        nodes, group_candidates = materializer.condition_length_cost(condition)
        if group_candidates > provider.max_candidates_per_batch:
            raise RuntimeError(
                f"validation group ordinal {ordinal} exceeds max_candidates_per_batch"
            )
        proposed_candidates = candidate_count + group_candidates
        proposed_nodes = max(max_nodes, nodes)
        proposed_cells = proposed_candidates * proposed_nodes * proposed_nodes
        exceeds = (
            proposed_candidates > provider.max_candidates_per_batch
            or (
                provider.max_padded_graph_cells_per_batch is not None
                and current
                and proposed_cells > provider.max_padded_graph_cells_per_batch
            )
            or len(current) >= provider.max_groups_per_batch
        )
        if exceeds:
            ordinals = tuple(item[0] for item in current)
            conditions = tuple(item[1] for item in current)
            yield ordinals, materializer.materialize(
                conditions,
                include_shuffled=provider.include_shuffled,
                device=context.device,
            )
            current = []
            candidate_count = 0
            max_nodes = 0
        current.append((ordinal, condition, nodes, group_candidates))
        candidate_count += group_candidates
        max_nodes = max(max_nodes, nodes)
    if current:
        ordinals = tuple(item[0] for item in current)
        conditions = tuple(item[1] for item in current)
        yield ordinals, materializer.materialize(
            conditions,
            include_shuffled=provider.include_shuffled,
            device=context.device,
        )


def _cat(values: list[Tensor], *, dtype: torch.dtype) -> Tensor:
    if not values:
        return torch.empty(0, dtype=dtype)
    return torch.cat(values, dim=0).to(device="cpu", dtype=dtype)


def _prepare_output(output_dir: Path, context: DistributedTrainingContext) -> None:
    if context.rank == 0:
        if output_dir.is_symlink() or (output_dir.exists() and not output_dir.is_dir()):
            raise FileExistsError(f"refusing to overwrite validation output directory: {output_dir}")
        if output_dir.exists():
            if any(output_dir.iterdir()):
                raise FileExistsError(
                    f"refusing to overwrite non-empty validation output directory: {output_dir}"
                )
        else:
            output_dir.mkdir(parents=True, mode=0o775)
        os.chmod(output_dir, 0o775)
    if context.distributed:
        dist.barrier()
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise RuntimeError(f"rank-zero did not prepare validation output directory: {output_dir}")


def _load_rank_payload(path: Path) -> Mapping[str, object]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older torch compatibility
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, Mapping):
        raise RuntimeError(f"rank validation artifact is not a mapping: {path}")
    return value


def _merged_summary(
    *,
    output_dir: Path,
    context: DistributedTrainingContext,
    checkpoints: Mapping[str, Path],
    total_groups: int,
) -> dict[str, object]:
    rank_payloads = [
        _load_rank_payload(output_dir / f"groups_rank{rank:02d}.pt")
        for rank in range(context.world_size)
    ]
    ordinal = torch.cat(
        [
            value["group_ordinal"]
            for value in rank_payloads
            if isinstance(value.get("group_ordinal"), Tensor)
        ],
        dim=0,
    ).to(dtype=torch.long)
    order = torch.argsort(ordinal)
    expected = torch.arange(total_groups, dtype=torch.long)
    if ordinal.numel() != total_groups or not torch.equal(ordinal.index_select(0, order), expected):
        raise RuntimeError("validation lanes did not cover every semantic group exactly once")
    masks = [
        value["selected_group_mask"]
        for value in rank_payloads
        if isinstance(value.get("selected_group_mask"), Tensor)
    ]
    selected = torch.cat(masks, dim=0).to(dtype=torch.bool).index_select(0, order)
    if not bool(selected.any()):
        raise RuntimeError("frozen validation found no selected instruction groups")
    candidates: dict[str, object] = {}
    for name, path in checkpoints.items():
        per_rank: list[Mapping[str, object]] = []
        for value in rank_payloads:
            raw_candidates = value.get("candidates")
            if not isinstance(raw_candidates, Mapping) or not isinstance(raw_candidates.get(name), Mapping):
                raise RuntimeError(f"rank validation artifact lacks candidate {name!r}")
            per_rank.append(raw_candidates[name])
        merged: dict[str, Tensor] = {}
        for metric in (
            "listwise_kl",
            "pairwise_loss",
            "pair_accuracy",
            "score_margin",
            "top1_accuracy",
            "identity_correct",
            "identity_margin",
        ):
            values = [value.get(metric) for value in per_rank]
            if not all(isinstance(value, Tensor) for value in values):
                raise RuntimeError(f"rank validation artifacts lack {metric!r} for {name!r}")
            merged[metric] = torch.cat(values, dim=0).to(dtype=torch.float32).index_select(0, order)
        candidates[name] = {
            "checkpoint": str(path),
            "metrics": {
                "listwise_kl": float(merged["listwise_kl"][selected].mean().item()),
                "pairwise_loss": float(merged["pairwise_loss"][selected].mean().item()),
                "true_pair_accuracy": float(merged["pair_accuracy"][selected].mean().item()),
                "score_margin": float(merged["score_margin"][selected].mean().item()),
                "top1_accuracy": float(merged["top1_accuracy"][selected].mean().item()),
                "identity_accuracy": float(merged["identity_correct"][selected].mean().item()),
                "identity_score_margin": float(merged["identity_margin"][selected].mean().item()),
            },
        }
    return {
        "schema_version": PROJECTOR_STAGE_A_VALIDATION_SCHEMA,
        "status": "complete",
        "split": "validation",
        "evaluation_protocol": {
            "coverage": "every_semantic_group_exactly_once",
            "true_wording": "canonical",
            "shuffled_wording": "runtime_cross_case_different_axis",
            "training_sampler_called": False,
            "test_used": False,
        },
        "world_size": context.world_size,
        "group_count": total_groups,
        "selected_group_count": int(selected.sum().item()),
        "candidates": candidates,
        "rank_artifacts": [f"groups_rank{rank:02d}.pt" for rank in range(context.world_size)],
    }


def finalize_existing_projector_stage_a_validation(
    *,
    output_dir: Path,
    checkpoints: Mapping[str, Path],
) -> dict[str, object]:
    """Atomically summarize already-complete rank artifacts without replaying validation.

    This narrowly supports recovery when every DDP rank has already written its
    immutable ``groups_rankXX.pt`` artifact but the rank-zero summary step did
    not complete.  It refuses partial lane sets and derives the expected group
    count only from a complete, rank-labelled artifact set; ``_merged_summary``
    then verifies dense exactly-once ordinal coverage before the summary write.
    """

    if output_dir.is_symlink() or not output_dir.is_dir():
        raise RuntimeError(f"existing-validation finalization requires a real output directory: {output_dir}")
    summary_path = output_dir / "summary.json"
    if summary_path.exists() or summary_path.is_symlink():
        raise FileExistsError(f"refusing to overwrite validation summary: {summary_path}")

    first_path = output_dir / "groups_rank00.pt"
    first = _load_rank_payload(first_path)
    world_size = first.get("world_size")
    if type(world_size) is not int or world_size < 1:
        raise RuntimeError(f"rank validation artifact has an invalid world_size: {first_path}")
    total_groups = 0
    for rank in range(world_size):
        path = output_dir / f"groups_rank{rank:02d}.pt"
        payload = _load_rank_payload(path)
        if payload.get("rank") != rank or payload.get("world_size") != world_size:
            raise RuntimeError(f"rank validation artifact identity mismatch: {path}")
        ordinal = payload.get("group_ordinal")
        if not isinstance(ordinal, Tensor):
            raise RuntimeError(f"rank validation artifact lacks group ordinals: {path}")
        total_groups += int(ordinal.numel())
    context = DistributedTrainingContext(
        rank=0,
        world_size=world_size,
        local_rank=0,
        device=torch.device("cpu"),
    )
    summary = _merged_summary(
        output_dir=output_dir,
        context=context,
        checkpoints=checkpoints,
        total_groups=total_groups,
    )
    _atomic_json(summary_path, summary)
    return summary


def evaluate_projector_stage_a_candidates(
    *,
    spec: PistaTrainingSpec,
    checkpoints: Mapping[str, Path],
    output_dir: Path,
) -> dict[str, object] | None:
    """Evaluate all candidates once against the exact frozen validation split."""

    if not checkpoints:
        raise ValueError("at least one Stage-A candidate checkpoint is required")
    if spec.stage != "stage-a" or spec.projector_only_stage_a is None:
        raise ValueError("evaluation requires a projector-only Stage-A specification")
    context = initialize_distributed_training(spec.device)
    try:
        _prepare_output(output_dir, context)
        provider = _provider(spec, "validation")
        models = _candidate_models(spec, checkpoints, device=context.device)
        local_ordinal: list[Tensor] = []
        local_selected: list[Tensor] = []
        local_candidates: dict[str, dict[str, list[Tensor]]] = {
            name: {
                "listwise_kl": [],
                "pairwise_loss": [],
                "pair_accuracy": [],
                "score_margin": [],
                "top1_accuracy": [],
                "identity_correct": [],
                "identity_margin": [],
            }
            for name in checkpoints
        }
        batch_count = 0
        for ordinals, batch in _exact_validation_batches(provider, context=context):
            expected_groups = len(ordinals)
            with torch.inference_mode():
                reference_mask: Tensor | None = None
                for name, model in models.items():
                    output = model(batch)
                    semantic = output.semantic
                    selected = _validated_evaluation_selection_mask(
                        semantic_selected_group_mask=semantic.selected_group_mask,
                        identity_selected_group_mask=output.identity_selected_group_mask,
                        identity_weight=model.config.identity_weight,
                        expected_groups=expected_groups,
                        scope="Stage-A validation",
                    )
                    if reference_mask is None:
                        reference_mask = selected
                    elif not torch.equal(reference_mask, selected):
                        raise RuntimeError("candidate checkpoint changed validation selection eligibility")
                    destination = local_candidates[name]
                    destination["listwise_kl"].append(semantic.group_listwise_kl.detach().cpu())
                    destination["pairwise_loss"].append(semantic.group_pairwise_loss.detach().cpu())
                    destination["pair_accuracy"].append(semantic.group_pair_accuracy.detach().cpu())
                    destination["score_margin"].append(semantic.group_score_margin.detach().cpu())
                    destination["top1_accuracy"].append(semantic.group_top1_accuracy.detach().cpu())
                    destination["identity_correct"].append(
                        output.identity_correct_by_group.detach().cpu().float()
                    )
                    destination["identity_margin"].append(output.identity_margin_by_group.detach().cpu())
                if reference_mask is None:  # pragma: no cover - checkpoints checked above
                    raise AssertionError("no candidate model was evaluated")
            local_ordinal.append(torch.tensor(ordinals, dtype=torch.long))
            local_selected.append(reference_mask)
            batch_count += 1
            if batch_count % 100 == 0:
                print(
                    json.dumps(
                        {"rank": context.rank, "validation_batches": batch_count, "last_group_ordinal": ordinals[-1]},
                        sort_keys=True,
                    ),
                    flush=True,
                )
        rank_payload: dict[str, object] = {
            "schema_version": PROJECTOR_STAGE_A_VALIDATION_SCHEMA,
            "rank": context.rank,
            "world_size": context.world_size,
            "group_ordinal": _cat(local_ordinal, dtype=torch.long),
            "selected_group_mask": _cat(local_selected, dtype=torch.bool),
            "candidates": {
                name: {metric: _cat(values, dtype=torch.float32) for metric, values in metrics.items()}
                for name, metrics in local_candidates.items()
            },
            "batch_count": batch_count,
        }
        _atomic_torch_save(output_dir / f"groups_rank{context.rank:02d}.pt", rank_payload)
        if context.distributed:
            dist.barrier()
        summary: dict[str, object] | None = None
        if context.rank == 0:
            summary = _merged_summary(
                output_dir=output_dir,
                context=context,
                checkpoints=checkpoints,
                total_groups=provider.runtime.group_count("validation"),
            )
            _atomic_json(output_dir / "summary.json", summary)
        if context.distributed:
            dist.barrier()
        return summary
    finally:
        if context.distributed and dist.is_initialized():
            dist.destroy_process_group()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rgg-evaluate-projector-stage-a",
        description="evaluate projector-only Stage-A candidate checkpoints on frozen validation",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="repeat once per named candidate checkpoint",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--finalize-existing",
        action="store_true",
        help="summarize complete groups_rankXX.pt artifacts without replaying validation",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    checkpoints: dict[str, Path] = {}
    for raw in args.checkpoint:
        name, path = _parse_checkpoint_argument(raw)
        if name in checkpoints:
            raise ValueError(f"duplicate checkpoint label: {name}")
        checkpoints[name] = path
    output_dir = args.output_dir.expanduser()
    if args.finalize_existing:
        # Keep recovery tied to a parseable registered Stage-A configuration,
        # while intentionally avoiding graph materialization and candidate replay.
        load_pista_training_spec(args.config)
        summary = finalize_existing_projector_stage_a_validation(
            output_dir=output_dir,
            checkpoints=checkpoints,
        )
    else:
        summary = evaluate_projector_stage_a_candidates(
            spec=load_pista_training_spec(args.config),
            checkpoints=checkpoints,
            output_dir=output_dir,
        )
    if summary is not None:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PROJECTOR_STAGE_A_VALIDATION_SCHEMA",
    "evaluate_projector_stage_a_candidates",
    "finalize_existing_projector_stage_a_validation",
]
