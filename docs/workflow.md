# Training and evaluation

The selected model uses seed 42, Stage-B epoch 14 and guidance strength 0.5. Model size refers to the Base generator. The Stage-B state also includes frozen reference weights and guidance parameters.

Configurations are in `configs/top30_65m_rdkit5/`. Preparation copies the frozen test inventory and resolves paths; it can finish without the external Qwen caches. The dependency check and model commands require those caches and fail when they are absent.

| Stage | Epochs | Learning rate | GPUs | Groups per GPU | Accumulation |
|---|---:|---:|---:|---:|---:|
| A | 30 | 0.0001 | 4 | 4 | 2 |
| B | 14 | 0.00003 | 4 | 4 | 2 |

Both stages use seed 42 and bf16 mixed precision. Stage A aligns the projector to 512-dimensional native reaction features. Stage B uses a 16-layer generator with node width 512, edge width 256 and eight attention heads; its training guidance strength is 1.

`train_stage_a.sh` trains, validates the candidate checkpoints and selects the projector by listwise KL, true-pair accuracy, then identity accuracy, with the earlier epoch breaking a tie. `train_stage_b.sh` trains, generates validation outputs and selects a deployment. The validation selection first compares strengths 0.15, 0.20, 0.25, 0.50 and 1.0 at epoch 14, then compares epochs 1, 2, 4, 6, 8, 10, 12 and 14 at the selected strength. It uses validation P@1, validity and N@1, with smaller strength or earlier epoch breaking remaining ties. This requires 3,072,000 validation trajectories. A fresh training run is selected anew; the supplied epoch is not assumed to win.

The test inventory retains 100 draws per instruction group and arm, 50 sampling steps, and the recorded seeds. Full scoring uses 10,000 case-cluster bootstrap replicates; the small check cohort uses 100. The release uses one group and ten trajectories per generation batch to limit memory. Resource requirements and full numerical agreement of this 65M release have not been established by a new model run.

Inference compares Reference, Matched (`true` in source files), and Shuffled. After generation, unique valid precursor sets are ranked by sampling frequency. Scoring uses the unchanged known-set evaluator and the supplied Common Top100 inventory. Saved results in `results/` are historical values, not outputs of package verification.

Run directories are relative to the caller. `RIGS_DATA_ROOT`, `RIGS_WEIGHTS_ROOT` and `RIGS_FEATURES_ROOT` override the sibling directories. Prepare a new run after relocating its inputs. CPU data readers and synthetic tests do not require the external language features.
