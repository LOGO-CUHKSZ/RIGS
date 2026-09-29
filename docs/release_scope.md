# Release scope

This arXiv release contains the model, loss functions, Stage-A/B training, validation selection, sampling and evaluation code for Top30 / 65M / RDKit5. It provides selected Base, Stage-A and Stage-B states and the matching constructed data. The selected Stage-B checkpoint is epoch 14 at strength 0.5, seed 42.

Excluded items include candidate-expansion and instruction-annotation pipelines, dataset-construction code, Qwen token caches and cache generation, other training supports and model capacities, Metric4, Structure, Pistachio, and additional seeds. Model states omit optimizer recovery state.

The release is not a self-contained end-to-end reproduction package: training and guided inference require compatible external Qwen features. The provided data and weights can be inspected independently. Full training, validation sampling and test generation are not performed while preparing this package. Synthetic CPU checks establish software interfaces rather than model quality.

The runtime includes shared modules with historical `pista` or `structure` names where required by the model/schema implementation. These do not supply Pistachio data or an annotation workflow. The bound-dataset module retains only read-only evaluation binding; its offline materialization routines are removed.

The source inventory is in [source_manifest.json](verification/source_manifest.json). A separate release verification report records the checks actually completed for these archives. The public archives contain no Git history, original experiment directories, personal paths or links to the broader review distribution.
