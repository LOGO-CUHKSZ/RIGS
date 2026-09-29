# External instruction features

Qwen instruction-token caches, Qwen weights, and feature-generation scripts are not included or downloaded automatically. The supplied training and guided-inference implementation reads cached token sequences; it has no online text-encoding fallback.

Point `RIGS_FEATURES_ROOT` at an externally prepared directory before `prepare.sh`:

```text
external_features/
  language/
    manifest.json
    table.jsonl
    texts.jsonl
    shards/tokens-00000.safetensors
  language_extension/
    manifest.json
    table.jsonl
    texts.jsonl
    shards/tokens-00000.safetensors
```

Additional shards may be listed in the manifests. The recorded encoder is Qwen3-Embedding-8B. The model interface expects 4,096-dimensional token vectors in bfloat16, keyed to the exact guidance strings in the supplied text pools. A pooled sentence vector is not a replacement for the token sequence. Matching shape alone does not establish compatibility with the trained projector.

`PistaTokenCache` and `PistaTokenExtensionCache` in `src/retro_guided_generation/data/runtime/pista.py` define the serialized input interface. Base rows identify the shard and full/query spans; the loader returns the query-token slice. `texts.jsonl` maps exact strings to row indices. Extension rows cover strings absent from the base cache. The base and extension must agree on their feature width and prompt template and must not duplicate text entries. The extension path may be set to JSON `null` in the prepared configs if the base cache covers every required string.

This describes how existing features are consumed. Cache construction, tokenization and encoder-extraction code are not part of this release. Compatibility and numerical agreement of independently prepared features remain the user's responsibility.

The supplied data archive separately includes the recorded 512-dimensional reaction features and train-only projector metadata used by the 65M Stage-A configuration. These contain no Qwen instruction embeddings. Stage B and guided inference require instruction features even when the selected weights are used.
