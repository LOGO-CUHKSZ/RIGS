# Supplied data

`rigs_arxiv_data/catalog.json` lists the released components. The package contains the recorded Top30 graph inputs, RDKit5 instruction pools and preferences, fixed native validation inputs, and the Common Top100 test inventory. It also includes the 512-dimensional native reaction embeddings and train-only metadata consumed by Stage A. Qwen embeddings are excluded.

The original training, validation and test membership, candidate order, preference labels, rewards and target probabilities are retained. Candidate records include only the routes needed by the Top30 inputs. Unused source-candidate records and non-RDKit metric fields are omitted. Existing identifiers may contain legacy `top60` strings; they remain unchanged so the original labels, graph indices and model inputs continue to join correctly. They do not imply that a Top60 training dataset is included.

The Common Top100 test inventory is a separate evaluation support, used for scoring the Top30-trained model. Its larger candidate lists are intentional. Native validation records are retained for checkpoint selection; test labels must not be used for that selection.

| Component | Main records |
|---|---|
| `guidance/rdkit5/top30` | Product/precursor cases, semantic preference groups, evaluation mappings, and text pools |
| `base/top30` | Graph index tensors, route-to-graph sidecars and collated graph shards |
| `evaluation/rdkit5_native` | Fixed validation selection and canonical native guidance inputs |
| `evaluation/rdkit5_common_top100` | Frozen test groups and known precursor inventory |
| `features/route_native512` | Recorded Stage-A reaction features for train/validation |
| `features/projector_train` | Train-only projector batching metadata |

Each semantic group is identified by its product, candidate pool, descriptor axis and direction. Its candidate preferences carry a route ID, label, eligibility, normalized metric, reward and teacher probability. Stage-A batching retains the eligible positive and negative candidates; neutral entries remain in the data but do not enter that contrast. Case records map route IDs to precursor strings. Text pools retain canonical instructions and supplied paraphrases. Graph readers use the saved shard and local-row coordinates.

Use the root README's `data_tools/load_data.py` commands to inspect records. JSON/CSV inspection requires only Python; graph inspection also requires PyTorch and PyTorch Geometric. Dataset construction, expansion, annotation, label generation and cache-production tools are not distributed.
