"""Dataset-neutral categorical predefined-DFM sampling.

The training code exposes a single exact categorical transition for Stage-B
REC1.  Evaluation needs the same transition repeatedly, but it must not
depend on a legacy dataset module, a chemistry decoder, or an experiment
specific batch class.  This module owns only the graph-tensor rollout.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import torch
from torch import Tensor

from .rollin import (
    sample_predefined_student_transition,
    sample_predefined_student_transition_per_sample_generators,
    sample_predefined_student_transition_stateless,
)


_PER_SAMPLE_TRANSITION_KERNELS = frozenset(
    {"rowwise_reference_v1", "batched_probability_v1", "stateless_counter_v1"}
)


@runtime_checkable
class PredefinedGraphModel(Protocol):
    """Minimal graph-model call signature needed by the generic sampler."""

    def __call__(
        self,
        *,
        t: Tensor,
        node_mask: Tensor,
        cond: Tensor,
        t_X: Tensor,
        t_E: Tensor,
        p_X: Tensor,
        p_E: Tensor,
        p_mask: Tensor,
        **kwargs: object,
    ) -> Mapping[str, Tensor]: ...


@dataclass(frozen=True, slots=True)
class PredefinedSamplingConfig:
    """Explicit finite schedule for eta=omega=0 predefined DFM sampling."""

    steps: int = 50
    per_sample_transition_kernel: str = "rowwise_reference_v1"

    def __post_init__(self) -> None:
        if isinstance(self.steps, bool) or not isinstance(self.steps, int) or self.steps < 1:
            raise ValueError("predefined sampling steps must be a positive integer")
        if (
            not isinstance(self.per_sample_transition_kernel, str)
            or self.per_sample_transition_kernel not in _PER_SAMPLE_TRANSITION_KERNELS
        ):
            raise ValueError(
                "per_sample_transition_kernel must be one of "
                + ", ".join(sorted(_PER_SAMPLE_TRANSITION_KERNELS))
            )


@dataclass(frozen=True, slots=True)
class PredefinedSamplingOutput:
    """Terminal probability prediction plus the preceding hard sampled state.

    ``X``/``E`` reproduce the historical sampler's final softmax output at
    ``t=1``.  ``terminal_state_X``/``terminal_state_E`` preserve the actual
    hard state immediately before that final endpoint prediction so downstream
    evaluation can decode the molecule actually reached by the trajectory.
    """

    X: Tensor
    E: Tensor
    terminal_state_X: Tensor
    terminal_state_E: Tensor
    traces: tuple["PredefinedSamplingTrace", ...] = ()


@dataclass(frozen=True, slots=True)
class PredefinedSamplingTrace:
    """One detached pre-transition state from a finite sampler rollout."""

    t: Tensor
    r_X: Tensor
    r_E: Tensor


_RESERVED_MODEL_KWARGS = frozenset(
    {
        "t",
        "node_mask",
        "cond",
        "t_X",
        "t_E",
        "p_X",
        "p_E",
        "p_mask",
    }
)


def _validate_graph_inputs(
    *,
    p_X: Tensor,
    p_E: Tensor,
    node_mask: Tensor,
    p_mask: Tensor,
    cond: Tensor,
) -> tuple[int, torch.device, torch.dtype]:
    if not all(isinstance(value, Tensor) for value in (p_X, p_E, node_mask, p_mask, cond)):
        raise TypeError("predefined sampler inputs must be tensors")
    if p_X.ndim != 3 or p_X.shape[-1] < 2 or not p_X.is_floating_point():
        raise ValueError("p_X must be a floating [B,N,Kx] tensor with Kx >= 2")
    batch_size, nodes, _node_classes = p_X.shape
    if (
        p_E.ndim != 4
        or p_E.shape[:3] != (batch_size, nodes, nodes)
        or p_E.shape[-1] < 2
        or not p_E.is_floating_point()
    ):
        raise ValueError("p_E must be a floating [B,N,N,Ke] tensor with Ke >= 2")
    if node_mask.dtype != torch.bool or node_mask.shape != (batch_size, nodes):
        raise ValueError("node_mask must be bool[B,N]")
    if p_mask.dtype != torch.bool or p_mask.shape != (batch_size, nodes):
        raise ValueError("p_mask must be bool[B,N]")
    if cond.ndim != 2 or cond.shape[0] != batch_size or not cond.is_floating_point():
        raise ValueError("cond must be floating[B,C]")
    tensors = (p_E, node_mask, p_mask, cond)
    if any(value.device != p_X.device for value in tensors):
        raise ValueError("predefined sampler inputs must share one device")
    if not bool(torch.isfinite(p_X).all()) or not bool(torch.isfinite(p_E).all()):
        raise ValueError("predefined sampler product graphs must be finite")
    if not bool(torch.isfinite(cond).all()):
        raise ValueError("predefined sampler condition must be finite")
    if not bool(torch.equal(p_E, p_E.transpose(1, 2))):
        raise ValueError("predefined sampler requires symmetric product edges")
    return batch_size, p_X.device, cond.dtype


def _validated_logits(
    logits: Mapping[str, Tensor], *, p_X: Tensor, p_E: Tensor) -> Mapping[str, Tensor]:
    if not isinstance(logits, Mapping):
        raise TypeError("predefined graph model must return a mapping")
    try:
        node_logits = logits["X"]
        edge_logits = logits["E"]
    except KeyError as error:
        raise KeyError("predefined graph model logits must contain X and E") from error
    if not isinstance(node_logits, Tensor) or not isinstance(edge_logits, Tensor):
        raise TypeError("predefined graph model logits must be tensors")
    if node_logits.shape != p_X.shape or edge_logits.shape != p_E.shape:
        raise ValueError("predefined graph model logits must match product graph shapes")
    if node_logits.device != p_X.device or edge_logits.device != p_X.device:
        raise ValueError("predefined graph model logits must share the graph device")
    if not node_logits.is_floating_point() or not edge_logits.is_floating_point():
        raise TypeError("predefined graph model logits must be floating point")
    if not bool(torch.isfinite(node_logits).all()) or not bool(torch.isfinite(edge_logits).all()):
        raise ValueError("predefined graph model logits must be finite")
    return {"X": node_logits, "E": edge_logits}


def _normalize_generators(
    generator: torch.Generator | Sequence[torch.Generator],
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[tuple[torch.Generator, ...], bool]:
    """Validate sampling RNGs and identify the requested sampling mode.

    A single generator retains the historical vectorized stream semantics.  A
    sequence assigns one independent, explicit stream to every batch row; the
    model forward remains batched while each categorical transition is sampled
    with its row's stream.  This permits a formal evaluator to audit a unique
    seed for every trajectory without mutating global RNG state.
    """

    if isinstance(generator, torch.Generator):
        values = (generator,) * batch_size
        per_sample = False
    elif isinstance(generator, Sequence) and not isinstance(generator, (str, bytes)):
        if len(generator) != batch_size:
            raise ValueError("per-sample generators must have one entry per batch row")
        values = tuple(generator)
        per_sample = True
    else:
        raise TypeError("generator must be a torch.Generator or a sequence of torch.Generator")
    for item in values:
        if not isinstance(item, torch.Generator):
            raise TypeError("every per-sample generator must be a torch.Generator")
        if torch.device(item.device).type != device.type:
            raise ValueError("generator must be device-local to predefined sampler tensors")
    return values, per_sample


def _normalize_stateless_seeds(
    generator: Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> Tensor:
    if not isinstance(generator, Tensor):
        raise TypeError("stateless_counter_v1 requires a tensor of frozen trajectory seeds")
    if generator.ndim != 1 or generator.shape[0] != batch_size:
        raise ValueError("stateless_counter_v1 seeds must have shape [B]")
    if generator.device != device:
        raise ValueError("stateless_counter_v1 seeds must share the graph device")
    if generator.dtype == torch.bool or generator.is_floating_point() or generator.is_complex():
        raise TypeError("stateless_counter_v1 seeds must have an integer dtype")
    result = generator.to(dtype=torch.int64)
    if bool(torch.any(result < 0)):
        raise ValueError("stateless_counter_v1 seeds must be non-negative")
    return result


@torch.inference_mode()
def sample_predefined_graph(
    model: PredefinedGraphModel,
    *,
    p_X: Tensor,
    p_E: Tensor,
    node_mask: Tensor,
    p_mask: Tensor,
    cond: Tensor,
    generator: torch.Generator | Sequence[torch.Generator] | Tensor,
    config: PredefinedSamplingConfig = PredefinedSamplingConfig(),
    model_kwargs: Mapping[str, Any] | None = None,
    trace_steps: Sequence[int] | None = None,
) -> PredefinedSamplingOutput:
    """Sample a dense graph with the fixed predefined categorical bridge.

    The rollout starts at the product endpoint, advances from ``t=0`` to
    ``t=1`` in exactly ``config.steps`` transition calls, and finally performs
    the historical terminal model prediction at ``t=1``.  The caller supplies
    either one explicit device-local generator (the historical shared-stream
    vectorized mode), one generator per batch row (independently seeded
    trajectories), or, for ``stateless_counter_v1``, one frozen integer seed
    per batch row.  None of these modes mutates global RNG state.
    """

    if not isinstance(config, PredefinedSamplingConfig):
        raise TypeError("config must be PredefinedSamplingConfig")
    batch_size, device, time_dtype = _validate_graph_inputs(
        p_X=p_X,
        p_E=p_E,
        node_mask=node_mask,
        p_mask=p_mask,
        cond=cond,
    )
    stateless_seeds: Tensor | None = None
    if config.per_sample_transition_kernel == "stateless_counter_v1":
        if not isinstance(generator, Tensor):
            raise TypeError("stateless_counter_v1 requires generator to be a tensor of frozen seeds")
        stateless_seeds = _normalize_stateless_seeds(
            generator,
            batch_size=batch_size,
            device=device,
        )
        generators: tuple[torch.Generator, ...] = ()
        per_sample_generators = False
    else:
        if isinstance(generator, Tensor):
            raise TypeError("tensor sampling seeds are supported only by stateless_counter_v1")
        generators, per_sample_generators = _normalize_generators(
            generator,
            batch_size=batch_size,
            device=device,
        )
    if model_kwargs is None:
        kwargs: dict[str, Any] = {}
    elif not isinstance(model_kwargs, Mapping):
        raise TypeError("model_kwargs must be a mapping or None")
    else:
        kwargs = dict(model_kwargs)
    overlap = sorted(_RESERVED_MODEL_KWARGS.intersection(kwargs))
    if overlap:
        raise ValueError(f"model_kwargs must not replace required graph inputs: {overlap}")

    if trace_steps is None:
        requested_trace_steps: frozenset[int] = frozenset()
    else:
        if isinstance(trace_steps, (str, bytes)):
            raise TypeError("trace_steps must be a sequence of step indices or None")
        normalized_steps: list[int] = []
        for index, value in enumerate(trace_steps):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"trace_steps[{index}] must be an integer")
            if value < 0 or value >= config.steps:
                raise ValueError("trace_steps must lie in [0, config.steps)")
            normalized_steps.append(value)
        if len(set(normalized_steps)) != len(normalized_steps):
            raise ValueError("trace_steps cannot contain duplicates")
        requested_trace_steps = frozenset(normalized_steps)

    current_X = p_X.detach().clone()
    current_E = p_E.detach().clone()
    traces: list[PredefinedSamplingTrace] = []
    for step in range(config.steps):
        t = torch.full(
            (batch_size,),
            float(step) / float(config.steps),
            dtype=time_dtype,
            device=device,
        )
        if step in requested_trace_steps:
            traces.append(
                PredefinedSamplingTrace(
                    t=t.detach().clone(),
                    r_X=current_X.detach().clone(),
                    r_E=current_E.detach().clone(),
                )
            )
        s = torch.full(
            (batch_size,),
            float(step + 1) / float(config.steps),
            dtype=time_dtype,
            device=device,
        )
        logits = _validated_logits(
            model(
                t=t,
                node_mask=node_mask,
                cond=cond,
                t_X=current_X,
                t_E=current_E,
                p_X=p_X,
                p_E=p_E,
                p_mask=p_mask,
                **kwargs,
            ),
            p_X=p_X,
            p_E=p_E,
        )
        if stateless_seeds is not None:
            sampled = sample_predefined_student_transition_stateless(
                logits=logits,
                current_X=current_X,
                current_E=current_E,
                p_X=p_X,
                p_E=p_E,
                node_mask=node_mask,
                p_mask=p_mask,
                t=t,
                s=s,
                seeds=stateless_seeds,
                step_index=step,
            )
            current_X, current_E = sampled.t_X, sampled.t_E
        elif per_sample_generators and config.per_sample_transition_kernel == "rowwise_reference_v1":
            transitions = [
                sample_predefined_student_transition(
                    logits={"X": logits["X"][index : index + 1], "E": logits["E"][index : index + 1]},
                    current_X=current_X[index : index + 1],
                    current_E=current_E[index : index + 1],
                    p_X=p_X[index : index + 1],
                    p_E=p_E[index : index + 1],
                    node_mask=node_mask[index : index + 1],
                    p_mask=p_mask[index : index + 1],
                    t=t[index : index + 1],
                    s=s[index : index + 1],
                    generator=generators[index],
                )
                for index in range(batch_size)
            ]
            current_X = torch.cat([item.t_X for item in transitions], dim=0)
            current_E = torch.cat([item.t_E for item in transitions], dim=0)
        elif per_sample_generators:
            sampled = sample_predefined_student_transition_per_sample_generators(
                logits=logits,
                current_X=current_X,
                current_E=current_E,
                p_X=p_X,
                p_E=p_E,
                node_mask=node_mask,
                p_mask=p_mask,
                t=t,
                s=s,
                generators=generators,
            )
            current_X, current_E = sampled.t_X, sampled.t_E
        else:
            sampled = sample_predefined_student_transition(
                logits=logits,
                current_X=current_X,
                current_E=current_E,
                p_X=p_X,
                p_E=p_E,
                node_mask=node_mask,
                p_mask=p_mask,
                t=t,
                s=s,
                generator=generators[0],
            )
            current_X, current_E = sampled.t_X, sampled.t_E

    terminal_logits = _validated_logits(
        model(
            t=torch.ones((batch_size,), dtype=time_dtype, device=device),
            node_mask=node_mask,
            cond=cond,
            t_X=current_X,
            t_E=current_E,
            p_X=p_X,
            p_E=p_E,
            p_mask=p_mask,
            **kwargs,
        ),
        p_X=p_X,
        p_E=p_E,
    )
    return PredefinedSamplingOutput(
        X=torch.softmax(terminal_logits["X"], dim=-1).detach(),
        E=torch.softmax(terminal_logits["E"], dim=-1).detach(),
        terminal_state_X=current_X.detach(),
        terminal_state_E=current_E.detach(),
        traces=tuple(traces),
    )


__all__ = [
    "PredefinedGraphModel",
    "PredefinedSamplingConfig",
    "PredefinedSamplingOutput",
    "PredefinedSamplingTrace",
    "sample_predefined_graph",
]
