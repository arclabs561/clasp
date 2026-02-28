# rankops

Operations on ranked lists: fuse multiple retrievers, then rerank. Pairs with **rankfns** (scoring kernels).

`rankops` covers the post-retrieval pipeline:

- **Fusion** — RRF, ISR, CombMNZ, Borda, DBSF, weighted variants (score-agnostic or score-based)
- **Reranking** — MaxSim (ColBERT late interaction), MMR/DPP diversity, Matryoshka two-stage
- **Evaluation** — NDCG, MRR, recall\@k, fusion parameter optimization

## Quickstart

```toml
[dependencies]
rankops = "0.1.0"
```

```rust
use rankops::{rrf, ScoredItem};

// List A: [doc1, doc2] (e.g. BM25)
let list_a: Vec<(u32, f32)> = vec![(1, 1.0), (2, 0.5)];
// List B: [doc2, doc3] (e.g. dense)
let list_b: Vec<(u32, f32)> = vec![(2, 0.9), (3, 0.4)];

// RRF with default k=60
let fused = rrf(&list_a, &list_b);
// doc2 ranks highest (appears in both)
assert_eq!(fused[0].0, 2);
```

## Features

- `serde`: enable serialization for configs and types.

## License

MIT OR Apache-2.0
