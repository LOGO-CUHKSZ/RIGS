# RIGS: Top30 / 65M / RDKit5

Model, training, inference and evaluation code for *Coverage Before Control: Route-Instruction Grounding and Steering for Controllable Retrosynthesis*.

This release provides one setting: USPTO Top30 training support, RDKit5 guidance, and a 65M Base model. It includes constructed data and selected model states. **Qwen instruction-token caches and their generation code are not distributed. Training and guided inference require compatible external caches.** Candidate expansion, instruction generation, annotation and dataset-construction pipelines are also excluded.

## Contents and setup

Download the [code ZIP](releases/rigs_arxiv_code.zip) here, and get `rigs_arxiv_data.tar.gz` and `rigs_arxiv_weights.tar.gz` from the [public Hugging Face release](https://huggingface.co/datasets/goldship2009/RIGS-arxiv-Top30-65M).

Extract the three archives beside one another:

```text
workspace/
  rigs_arxiv_code/
  rigs_arxiv_data/
  rigs_arxiv_weights/
```

Use Python 3.11 and install a suitable CUDA build of PyTorch 2.7.1, followed by `python -m pip install -r requirements.txt`. Run commands from `rigs_arxiv_code/`.

Inspect the supplied data or run synthetic CPU tests without the external caches:

```bash
python data_tools/load_data.py --root ../rigs_arxiv_data --list
python data_tools/load_data.py --root ../rigs_arxiv_data --dataset guidance/rdkit5/top30 --split train --limit 1
python -m unittest discover -s tests -v
```

## Training and evaluation

The following commands require separately supplied Qwen caches, using the [external feature interface](docs/external_features.md):

```bash
export RIGS_FEATURES_ROOT=/path/to/external_features
bash scripts/prepare.sh runs/train full
python reproduce.py check --run-dir runs/train
bash scripts/train_stage_a.sh runs/train
bash scripts/train_stage_b.sh runs/train cuda:0
bash scripts/evaluate.sh runs/train cuda:0
```

Both training stages retain the recorded four-GPU configuration. To evaluate the supplied weights, prepare a fresh run and call `evaluate.sh` directly. The `check` cohort contains 20 instruction groups and 6,000 trajectories across Reference, Matched and Shuffled; `full` contains 2,560 groups and 768,000 trajectories. Neither is executed by package preparation. `RIGS_DRY_RUN=1` prints the shell commands without launching them.

See [training and evaluation](docs/workflow.md), [data formats](docs/data.md), [saved results](results/README.md), and [release scope](docs/release_scope.md). A missing cache is a declared external dependency, not a request to download the former review package.

Verification records cover [synthetic CPU and source checks](docs/verification/code_check.json) and [data interfaces and model-state loading](docs/verification/package_check.json). Full training and generation have not been rerun for this release.
