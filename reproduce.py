"""Prepare and run the recorded USPTO Top30/65M/RDKit5 recipe.

Preparation and dependency checks do not start training or generation.
"""
from pathlib import Path
import argparse
import gzip
import json
import math
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(variable, '1')


def read_rows(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_rows(path, rows):
    with gzip.open(path, 'wt', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def expand(value, roots):
    if isinstance(value, dict):
        return {key: expand(item, roots) for key, item in value.items()}
    if isinstance(value, list):
        return [expand(item, roots) for item in value]
    if isinstance(value, str):
        for key, root in roots.items():
            value = value.replace('${' + key + '}', str(root))
        if '${' in value:
            raise ValueError('Unresolved path placeholder: ' + value)
    return value


def prepare(args):
    roots = {key: getattr(args, field).expanduser().resolve(strict=(key != 'ASSETS_ROOT'))
             for key, field in [('DATA_ROOT', 'data_root'), ('WEIGHTS_ROOT', 'weights_root'), ('ASSETS_ROOT', 'assets_root')]}
    run = args.run_dir.expanduser().resolve()
    if run.exists():
        raise FileExistsError('Choose a new run directory: ' + str(run))
    for root in (roots['DATA_ROOT'], roots['WEIGHTS_ROOT']):
        if not (root / 'catalog.json').is_file():
            raise FileNotFoundError(root / 'catalog.json')
    roots['RUN_ROOT'] = run
    configs = {path.stem: expand(json.loads(path.read_text()), roots)
               for path in (ROOT / 'configs' / 'top30_65m_rdkit5').glob('*.json')}
    formal = configs['formal_eval']
    frozen = roots['DATA_ROOT'] / 'data/evaluation/rdkit5_common_top100/strict100_pista_frozen_test_m100.jsonl.gz'
    inventory = roots['DATA_ROOT'] / 'data/evaluation/rdkit5_common_top100/selected_top100_endpoint_inventory.jsonl.gz'
    rows = list(read_rows(frozen))
    if len(rows) != 2560:
        raise ValueError('The recorded Common Top100 inventory must contain 2,560 signed groups')
    if args.cohort == 'check':
        selected = []
        for axis in formal['protocol']['axes']:
            families = {}
            for row in rows:
                if row['axis_id'] == axis:
                    families.setdefault(row['case_id'], {})[int(row['polarity'])] = row
            pairs = [value for value in families.values() if -1 in value and 1 in value][:2]
            if len(pairs) != 2:
                raise ValueError('The check cohort needs two cases per axis for the recorded bootstrap')
            for pair in pairs:
                selected.extend([pair[-1], pair[1]])
        rows = sorted(selected, key=lambda row: row['evaluation_group_index'])
    run.mkdir(parents=True)
    (run / 'data').mkdir()
    # Copy frozen records without reseeding or relabeling. The check cohort
    # preserves two complete signed families per axis and all 100 saved draws.
    frozen_out = run / 'data/frozen.jsonl.gz'
    inventory_out = run / 'data/endpoint_inventory.jsonl.gz'
    write_rows(frozen_out, rows)
    cases = {row['case_id'] for row in rows}
    write_rows(inventory_out, (row for row in read_rows(inventory) if row['case_id'] in cases))
    expected = formal['protocol']['expected']
    expected.update(selected_groups=len(rows), selected_cases=len(cases),
                    selected_families=len(rows) // 2, total_arms=3,
                    trajectories_per_arm=len(rows) * 100, total_trajectories=len(rows) * 300)
    if args.cohort == 'check':
        formal['experiment_id'] += '_interface_check'
        formal['protocol']['bootstrap_replicates'] = 100
        formal['protocol']['families_per_axis'] = 2
        profile = configs['execution_profile']
        profile['profile_id'] = 'top30_65m_interface_check_g1_t10_v1'
        profile['trajectory_batch'] = 10
        profile['adapter_options']['pista']['max_groups_per_generation_batch'] = 1
        profile['launcher_options']['shards_per_arm'] = 1
        formal['execution']['profile_id'] = profile['profile_id']
    for name, value in configs.items():
        write_json(run / 'configs' / (name + '.json'), value)
    prepared = {'schema_version': 'iclr2027.uspto_top30_recipe.v1', 'cohort': args.cohort,
                'data_root': str(roots['DATA_ROOT']), 'weights_root': str(roots['WEIGHTS_ROOT']),
                'assets_root': str(roots['ASSETS_ROOT']), 'frozen_manifest': str(frozen_out),
                'endpoint_inventory': str(inventory_out),
                'checkpoint': str(roots['WEIGHTS_ROOT'] / 'rdkit5/top30_65m/stage_b.pt'),
                'vocabulary': str(roots['DATA_ROOT'] / 'data/shared/graph_vocabulary.json'),
                'groups': len(rows), 'cases': len(cases), 'draws_per_group_arm': 100,
                'total_trajectories': len(rows) * 300, 'sampling_steps': 50,
                'alpha': 0.5, 'selected_stage_b_epoch': 14,
                'selection_source': 'recorded validation-frozen coordinate',
                'training_epochs': {'stage_a': 30, 'stage_b': 14},
                'feature_cache_recomputed': False, 'model_executed': False}
    write_json(run / 'prepared.json', prepared)
    print(json.dumps(prepared, indent=2))


def load_run(path):
    run = path.expanduser().resolve(strict=True)
    return run, json.loads((run / 'prepared.json').read_text())


def check(args):
    run, prepared = load_run(args.run_dir)
    errors = []
    for name in ('stage_a', 'stage_b', 'inference'):
        config = json.loads((run / 'configs' / (name + '.json')).read_text())
        for key, value in config['paths'].items():
            if isinstance(value, str) and not Path(value).exists():
                errors.append(name + ':' + key + ': ' + value)
    for key in ('checkpoint', 'vocabulary', 'frozen_manifest', 'endpoint_inventory'):
        if not Path(prepared[key]).is_file():
            errors.append(key + ': ' + prepared[key])
    if errors:
        raise FileNotFoundError('Qwen token caches are external inputs and are not distributed. '
                                'Set --assets-root when preparing the run. Missing inputs:\n' + '\n'.join(errors))
    from retro_guided_generation.training.pista import load_pista_training_spec
    from retro_guided_generation.evaluation.pista import validate_pista_frozen_rows
    specs = {name: load_pista_training_spec(run / 'configs' / (name + '.json'))
             for name in ('stage_a', 'stage_b', 'inference')}
    rows = list(read_rows(prepared['frozen_manifest']))
    validate_pista_frozen_rows(rows, expected_draw_count=100,
                             root_draw_seed_base=20260721, trajectory_seed_base=42000)
    report = {'status': 'passed', 'configurations': 3, 'frozen_groups': len(rows),
              'model_forward_executed': False, 'training_executed': False,
              'generation_executed': False, 'weights_loaded': False}
    if args.load_weights:
        import torch
        torch.set_num_threads(1)
        from retro_guided_generation.evaluation.pista import _load_pista_model, _evaluation_context, _rebuild_condition
        model = _load_pista_model(specs['inference'], prepared['checkpoint'])
        context = _evaluation_context(specs['inference'], split='test', branches=('metric',))
        token_shapes = []
        for row in rows[:10]:
            condition = _rebuild_condition(context, row, fallback_signed_candidate_upper_bound=100)
            for text_id in (condition.true_text.text_id, condition.shuffled.donor_text_id):
                tokens = context.text_tokens[text_id]
                token_shapes.append(list(tokens.shape))
        report.update(weights_loaded=True, state_load='zero missing or unexpected keys',
                      model_parameters=sum(parameter.numel() for parameter in model.parameters()),
                      token_samples=len(token_shapes), token_shapes=token_shapes)
        from retro_guided_generation.training.pista import _build_pista_stage, _provider
        from retro_guided_generation.evaluation.projector_stage_a import _candidate_models
        _candidate_models(specs['stage_a'], {'saved': specs['stage_b'].stage_a_checkpoint},
                          device=torch.device('cpu'))
        stage_b, initialization = _build_pista_stage(specs['stage_b'])
        report['stage_b_initialization'] = initialization
        del stage_b
        report['training_views'] = {}
        for name, split in (('stage_a', 'train'), ('stage_a', 'validation'), ('stage_b', 'train')):
            print('Opening ' + name + ':' + split, file=sys.stderr, flush=True)
            provider = _provider(specs[name], split)
            try:
                report['training_views'][name + ':' + split] = provider.runtime.group_count(split)
            finally:
                provider.close()
            print('Opened ' + name + ':' + split, file=sys.stderr, flush=True)
    write_json(run / 'dependency_check.json', report)
    print(json.dumps(report, indent=2))


def infer(args):
    run, prepared = load_run(args.run_dir)
    formal = json.loads((run / 'configs/formal_eval.json').read_text())['models']['top30']
    if formal['checkpoint'] != prepared['checkpoint'] or formal['alpha'] != prepared['alpha']:
        raise ValueError('Deployment and scoring configurations disagree')
    from retro_guided_generation.evaluation.pista import generate_pista_arms
    output = run / 'outputs/formal/top30/arms' if args.shard_count == 1 else run / f'outputs/shards/{args.shard_index:03d}/arms'
    output.mkdir(parents=True, exist_ok=True)
    return generate_pista_arms(training_config=run / 'configs/inference.json', checkpoint=prepared['checkpoint'],
        frozen_manifest=prepared['frozen_manifest'], arm_outputs={name: output / (name + '.jsonl') for name in ('reference', 'true', 'shuffled')},
        expected_draw_count=100, sampling_steps=50, guidance_alpha=prepared['alpha'], device=args.device,
        execution_profile=run / 'configs/execution_profile.json',
        shard_index=args.shard_index, shard_count=args.shard_count,
        vocabulary_path=prepared['vocabulary'], vocabulary_id='uspto_full', progress_every_groups=32)


def merge(args):
    run, prepared = load_run(args.run_dir)
    from retro_guided_generation.evaluation.pista import merge_pista_arm_shards
    output = run / 'outputs/formal/top30/arms'
    output.mkdir(parents=True, exist_ok=True)
    for arm in ('reference', 'true', 'shuffled'):
        merge_pista_arm_shards(arm=arm, frozen_manifest=prepared['frozen_manifest'],
            input_jsonl=[run / f'outputs/shards/{index:03d}/arms/{arm}.jsonl' for index in range(args.shard_count)],
            output_jsonl=output / (arm + '.jsonl'), expected_group_count=prepared['groups'])


def train(args):
    run, _ = load_run(args.run_dir)
    from retro_guided_generation.training.pista import load_pista_training_spec, run_pista_formal_training
    spec = load_pista_training_spec(run / 'configs' / ('stage_' + args.stage + '.json'))
    print(json.dumps(run_pista_formal_training(spec), indent=2, default=str))


def stage_a_candidates(run):
    config = json.loads((run / 'configs/stage_a.json').read_text())
    return {f'A0_epoch_{epoch:06d}': run / f'stage_a/checkpoints/epoch_{epoch:06d}.ckpt'
            for epoch in config['run']['formal']['candidate_epochs']}


def validate_a(args):
    run, _ = load_run(args.run_dir)
    candidates = stage_a_candidates(run)
    for path in candidates.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    from retro_guided_generation.training.pista import load_pista_training_spec
    from retro_guided_generation.evaluation.projector_stage_a import evaluate_projector_stage_a_candidates
    summary = evaluate_projector_stage_a_candidates(
        spec=load_pista_training_spec(run / 'configs/stage_a.json'),
        checkpoints=candidates, output_dir=run / 'validation_stage_a')
    if summary is not None:
        print(json.dumps(summary, indent=2))


def select_a(args):
    run, _ = load_run(args.run_dir)
    summary = json.loads((run / 'validation_stage_a/summary.json').read_text())
    if summary.get('status') != 'complete' or summary.get('split') != 'validation':
        raise ValueError('Require completed frozen validation')
    if summary['evaluation_protocol'].get('test_used') is not False:
        raise ValueError('Selection must not use test data')
    candidates = stage_a_candidates(run)
    if set(summary['candidates']) != set(candidates):
        raise ValueError('Require all registered Stage-A candidates')
    ranking = []
    for name, path in candidates.items():
        row = summary['candidates'][name]
        if Path(row['checkpoint']).resolve() != path.resolve():
            raise ValueError('Validation checkpoint differs from the registered candidate')
        metrics = row['metrics']
        key = (float(metrics['listwise_kl']), -float(metrics['true_pair_accuracy']),
               -float(metrics['identity_accuracy']))
        if not all(math.isfinite(value) for value in key):
            raise ValueError('Non-finite validation metrics')
        ranking.append((key, name, path))
    _, name, source = min(ranking)
    destination = run / 'stage_a/selected_projector.pt'
    if destination.exists() or (run / 'stage_b').exists():
        raise FileExistsError('Selection already saved or Stage B already started')
    import torch
    from retro_guided_generation.evaluation.projector_stage_a import _require_checkpoint_state
    torch.save({'state_dict': dict(_require_checkpoint_state(source))}, destination)
    config_path = run / 'configs/stage_b.json'
    config = json.loads(config_path.read_text())
    config['paths']['stage_a_checkpoint'] = str(destination)
    write_json(config_path, config)
    selection = {'selected': name, 'source': str(source), 'destination': str(destination),
                 'split': 'validation', 'test_used': False,
                 'rule': 'minimum listwise KL, maximum true-pair accuracy, maximum identity accuracy; earlier epoch on an exact tie'}
    write_json(run / 'stage_a/selection.json', selection)
    print(json.dumps(selection, indent=2))


def main():
    from stage_b_flow import prepare_b_validation, validate_b, select_b
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    command = commands.add_parser('prepare', help='materialize paths and the saved evaluation inventory; no model execution')
    for name in ('data-root', 'weights-root', 'run-dir'):
        command.add_argument('--' + name, type=Path, required=True)
    command.add_argument('--assets-root', type=Path, default=ROOT.parent / 'external_features',
                         help='external Qwen caches; intentionally not distributed')
    command.add_argument('--cohort', choices=('check', 'full'), default='full')
    command.set_defaults(function=prepare)
    for name, function in [('check', check), ('infer', infer), ('merge', merge), ('train', train),
                           ('validate-a', validate_a), ('select-a', select_a),
                           ('prepare-b-validation', prepare_b_validation),
                           ('validate-b', validate_b), ('select-b', select_b)]:
        command = commands.add_parser(name)
        command.add_argument('--run-dir', type=Path, required=True)
        command.set_defaults(function=function)
        if name == 'check':
            command.add_argument('--load-weights', action='store_true', help='load weights and read token samples on CPU; no forward pass')
        if name == 'infer':
            command.add_argument('--device', default='cuda:0')
            command.add_argument('--shard-index', type=int, default=0)
            command.add_argument('--shard-count', type=int, default=1)
        if name == 'validate-b':
            command.add_argument('--device', default='cuda:0')
        if name == 'merge':
            command.add_argument('--shard-count', type=int, required=True)
        if name == 'train':
            command.add_argument('--stage', choices=('a', 'b'), required=True)
    command = commands.add_parser('score', help='score completed arm outputs with the recorded estimator')
    command.add_argument('--run-dir', type=Path, required=True)
    command.set_defaults(function=lambda args: subprocess.run([sys.executable, str(ROOT / 'score_results.py'), '--run-dir', str(args.run_dir)], check=True))
    args = parser.parse_args()
    args.function(args)


if __name__ == '__main__':
    main()
