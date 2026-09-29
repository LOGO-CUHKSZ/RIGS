"""Validation-selected Stage-B deployment for the supplied Top30 recipe.

Preparation is CPU-only. Validation invokes the existing sampler. Selection
re-scores the saved validation outputs before publishing tensor-only states.
"""
from copy import deepcopy
import json
import math
from pathlib import Path

from reproduce import load_run, read_rows, write_json, write_rows


ALPHAS = (0.15, 0.20, 0.25, 0.50, 1.0)
SELECTION_RULE = 'Top30 P-first: maximize true P@1, then validity, then minimize N@1; smaller alpha/earlier epoch on ties'


def _json(path):
    return json.loads(path.read_text())


def _candidates(run):
    config = _json(run / 'configs/stage_b.json')
    epochs = config['run']['formal']['candidate_epochs']
    if not epochs or any(type(epoch) is not int or epoch < 1 for epoch in epochs):
        raise ValueError('Stage-B candidates must be positive integer epochs')
    if len(set(epochs)) != len(epochs):
        raise ValueError('Duplicate Stage-B candidate epoch')
    return {epoch: run / f'stage_b/checkpoints/epoch_{epoch:06d}.ckpt' for epoch in sorted(epochs)}


def _validation_rows(path, draws):
    from retro_guided_generation.evaluation.pista import validate_pista_frozen_rows
    rows = list(read_rows(path))
    if not rows or any(row.get('source_split') != 'validation' for row in rows):
        raise ValueError('Stage-B selection requires validation data; test rows are forbidden')
    if any(row.get('branch') != 'metric' for row in rows):
        raise ValueError('The Top30 recipe requires metric validation groups')
    if len({row['group_uid'] for row in rows}) != len(rows):
        raise ValueError('Duplicate validation group identity')
    validate_pista_frozen_rows(rows, expected_draw_count=draws)
    return rows


def prepare_b_validation(args):
    from retro_guided_generation.evaluation.pista import (
        root_choice_indices_for_group, root_draw_seeds_for_group,
        trajectory_seeds_for_group,
    )
    run, prepared = load_run(args.run_dir)
    destination = run / 'validation_stage_b'
    if destination.exists():
        raise FileExistsError('Stage-B validation is already prepared')
    corpus = Path(prepared['data_root']) / 'data/evaluation/rdkit5_native'
    source = corpus / 'evaluation/validation_frozen.jsonl'
    first = next(read_rows(source))
    rows = _validation_rows(source, len(first['trajectory_seeds']))
    selection = _json(corpus / 'evaluation/validation_selection.json')
    if (selection.get('split') != 'validation'
            or selection['selected_group_count'] != len(rows)
            or selection['selected_case_count'] != len({row['case_id'] for row in rows})):
        raise ValueError('Validation inventory differs from its delivered selection record')
    # D1 stores the selected groups and RC slots with a one-draw scaffold.
    # Materialize the 100-draw arithmetic schedule; labels/groups stay fixed.
    draws = 100
    for row in rows:
        index = row['evaluation_group_index']
        row['root_draw_seeds'] = root_draw_seeds_for_group(evaluation_group_index=index, draw_count=draws)
        row['trajectory_seeds'] = trajectory_seeds_for_group(evaluation_group_index=index, draw_count=draws)
        row['root_choice_indices'] = root_choice_indices_for_group(root_draw_seeds=row['root_draw_seeds'])
    config = deepcopy(_json(run / 'configs/stage_b.json'))
    config['paths']['corpus_root'] = str(corpus)
    # Use the bounded generation profile already used by the interface check.
    profile = deepcopy(_json(run / 'configs/execution_profile.json'))
    profile['profile_id'] = 'top30_stage_b_validation_g1_t10_v1'
    profile['trajectory_batch'] = 10
    profile['adapter_options']['pista']['max_groups_per_generation_batch'] = 1
    epochs = list(_candidates(run))
    plan = {'schema_version': 'iclr2027.stage_b_validation_plan.v1', 'split': 'validation',
            'test_used': False, 'epochs': epochs, 'alphas': list(ALPHAS),
            'scale_epoch': max(epochs), 'groups': len(rows),
            'cases': len({row['case_id'] for row in rows}), 'draw_count': draws,
            'sampling_steps': 50, 'selection_rule': SELECTION_RULE,
            'scale_then_epoch': True, 'metric_units': 'fractions',
            'coordinates': len(ALPHAS) + len(epochs) - 1,
            'generated_arm': 'true', 'content_hashes_computed': False}
    destination.mkdir()
    write_rows(destination / 'frozen.jsonl.gz', rows)
    write_json(destination / 'inference.json', config)
    write_json(destination / 'execution_profile.json', profile)
    write_json(destination / 'plan.json', plan)
    print(json.dumps(plan, indent=2))


def _load_validation(run):
    root = run / 'validation_stage_b'
    plan = _json(root / 'plan.json')
    if (plan.get('split') != 'validation' or plan.get('test_used') is not False
            or plan.get('selection_rule') != SELECTION_RULE
            or plan.get('alphas') != list(ALPHAS)
            or plan.get('epochs') != list(_candidates(run))):
        raise ValueError('Stage-B validation plan differs from the registered recipe')
    if plan['scale_epoch'] != max(plan['epochs']):
        raise ValueError('Select the alpha at the terminal registered epoch')
    if plan['draw_count'] != 100 or plan['sampling_steps'] != 50:
        raise ValueError('Stage-B validation requires 100 draws and 50 sampling steps')
    training = _json(run / 'configs/stage_b.json')
    validation_config = _json(root / 'inference.json')
    expected_validation = deepcopy(training)
    prepared = _json(run / 'prepared.json')
    expected_validation['paths']['corpus_root'] = str(Path(prepared['data_root']) / 'data/evaluation/rdkit5_native')
    if expected_validation != validation_config:
        raise ValueError('Stage-B configuration changed after validation preparation')
    rows = _validation_rows(root / 'frozen.jsonl.gz', plan['draw_count'])
    if len(rows) != plan['groups'] or len({row['case_id'] for row in rows}) != plan['cases']:
        raise ValueError('Validation coverage differs from the plan')
    return root, plan, rows


def _coordinate(root, epoch, alpha):
    return root / f'epoch_{epoch:06d}_alpha_{alpha:g}' / 'true.jsonl'


def _metrics(rows, path, alpha, plan, profile):
    from score_results import _validate_generated_identity, index_rows
    from retro_guided_generation.evaluation.inference import score_generated_group
    generated = index_rows(list(read_rows(path)), key='group_uid', label='Stage-B validation')
    if set(generated) != {row['group_uid'] for row in rows}:
        raise ValueError('Stage-B validation group coverage differs')
    totals = {name: 0.0 for name in ('freq_positive_at_1', 'validity', 'freq_negative_at_1')}
    for row in rows:
        canonical = _validate_generated_identity(
            row, generated[row['group_uid']], condition='true', alpha=alpha,
            draw_count=plan['draw_count'], sampling_steps=plan['sampling_steps'],
            profile_id=profile['profile_id'])
        metrics = score_generated_group(
            canonical, row['positive_target_smiles'], row['negative_target_smiles'],
            top_k=(1,), expected_draw_count=plan['draw_count'],
            require_canonical_known_sets=True, require_canonical_generated=True)['metrics']
        for name in totals:
            value = float(metrics[name])
            if not math.isfinite(value):
                raise ValueError('Non-finite Stage-B validation metric')
            totals[name] += value
    return {name: value / len(rows) for name, value in totals.items()}


def _rank(record, coordinate):
    metrics = record['metrics']
    return (-metrics['freq_positive_at_1'], -metrics['validity'],
            metrics['freq_negative_at_1'], record[coordinate])


def _collect(run, *, device=None):
    """Generate if requested, then select using actual keyed validation scores."""
    root, plan, rows = _load_validation(run)
    prepared = _json(run / 'prepared.json')
    profile = _json(root / 'execution_profile.json')
    candidates = _candidates(run)
    for path in candidates.values():
        if not path.is_file():
            raise FileNotFoundError('Missing registered Stage-B candidate: ' + path.name)
    records = []

    def evaluate(epoch, alpha):
        path = _coordinate(root, epoch, alpha)
        if device is not None and not path.exists():
            from retro_guided_generation.evaluation.pista import generate_pista_arms
            path.parent.mkdir(parents=True, exist_ok=True)
            generate_pista_arms(
                training_config=root / 'inference.json', checkpoint=candidates[epoch],
                frozen_manifest=root / 'frozen.jsonl.gz', split='validation',
                expected_draw_count=plan['draw_count'], arm_outputs={'true': path},
                sampling_steps=plan['sampling_steps'], guidance_alpha=alpha, device=device,
                execution_profile=root / 'execution_profile.json',
                vocabulary_path=prepared['vocabulary'], vocabulary_id='uspto_full')
        result = {'epoch': epoch, 'alpha': alpha, 'checkpoint': str(candidates[epoch]),
                  'metrics': _metrics(rows, path, alpha, plan, profile)}
        records.append(result)
        return result

    scale_rows = [evaluate(plan['scale_epoch'], alpha) for alpha in plan['alphas']]
    scale_winner = min(scale_rows, key=lambda row: _rank(row, 'alpha'))
    alpha = scale_winner['alpha']
    epoch_rows = [scale_winner]
    epoch_rows.extend(evaluate(epoch, alpha) for epoch in plan['epochs'] if epoch != plan['scale_epoch'])
    selected = min(epoch_rows, key=lambda row: _rank(row, 'epoch'))
    return {'status': 'complete', 'split': 'validation', 'test_used': False,
            'selection_rule': SELECTION_RULE, 'groups': len(rows),
            'draw_count': plan['draw_count'], 'scale_epoch': plan['scale_epoch'],
            'candidates': records, 'selected': selected, 'metric_units': 'fractions'}


def validate_b(args):
    run, _ = load_run(args.run_dir)
    if (run / 'validation_stage_b/summary.json').exists():
        raise FileExistsError('Stage-B validation summary already exists')
    result = _collect(run, device=args.device)
    write_json(run / 'validation_stage_b/summary.json', result)
    print(json.dumps(result, indent=2))


def select_b(args):
    import torch
    from retro_guided_generation.evaluation.projector_stage_a import _require_checkpoint_state
    run, prepared = load_run(args.run_dir)
    if any((run / name).exists() for name in ('deployment', 'outputs', 'reports')):
        raise FileExistsError('Deploy only before test generation/scoring starts in this run')
    summary = _json(run / 'validation_stage_b/summary.json')
    if summary.get('status') != 'complete' or summary.get('split') != 'validation' or summary.get('test_used') is not False:
        raise ValueError('Require completed validation-only Stage-B selection')
    # Re-score saved arm files; a hand-edited scalar summary cannot pick a weight.
    checked = _collect(run)
    if summary != checked:
        raise ValueError('Stage-B summary differs from saved validation outputs')
    selected = checked['selected']
    source = Path(selected['checkpoint'])
    payload = torch.load(source, map_location='cpu', weights_only=False)
    if payload.get('artifact_kind') != 'candidate_checkpoint' or payload.get('candidate_epoch') != selected['epoch']:
        raise ValueError('Stage-B checkpoint metadata differs from the selected candidate')
    state = payload.get('state_dict')
    if not isinstance(state, dict) or not state or any(not isinstance(k, str) or not isinstance(v, torch.Tensor) for k, v in state.items()):
        raise ValueError('Stage-B candidate must contain a nonempty tensor state_dict')
    training = _json(run / 'configs/stage_b.json')
    stage_a_state = dict(_require_checkpoint_state(Path(training['paths']['stage_a_checkpoint'])))
    if not stage_a_state:
        raise ValueError('Stage-A checkpoint must contain a nonempty tensor state_dict')
    destination = run / 'deployment'
    destination.mkdir()
    stage_a = destination / 'stage_a.pt'
    stage_b = destination / 'stage_b.pt'
    torch.save({'state_dict': stage_a_state}, stage_a)
    torch.save({'state_dict': state, 'candidate_epoch': selected['epoch']}, stage_b)
    inference = deepcopy(training)
    inference['paths']['corpus_root'] = _json(run / 'configs/inference.json')['paths']['corpus_root']
    inference['paths']['stage_a_checkpoint'] = str(stage_a)
    formal = _json(run / 'configs/formal_eval.json')
    formal['models']['top30'].update(checkpoint=str(stage_b), alpha=selected['alpha'])
    prepared.update(checkpoint=str(stage_b), alpha=selected['alpha'],
                    selected_stage_b_epoch=selected['epoch'],
                    selection_source='current-run frozen validation',
                    deployment_selection=str(destination / 'selection.json'))
    write_json(destination / 'selection.json', checked)
    write_json(run / 'configs/inference.json', inference)
    write_json(run / 'configs/formal_eval.json', formal)
    # Publish the run pointer last; infer/score reject inconsistent pointers.
    write_json(run / 'prepared.json', prepared)
    print(json.dumps({'selected_epoch': selected['epoch'], 'alpha': selected['alpha'],
                      'selection_split': 'validation', 'deployment': 'deployment/stage_b.pt'}, indent=2))
