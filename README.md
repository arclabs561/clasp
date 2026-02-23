# clasp

Rank fusion and reranking for hybrid search.

`clasp` covers the post-retrieval ranking pipeline end-to-end:

- **Fusion** — RRF, ISR, CombMNZ, Borda, DBSF, weighted variants (score-agnostic or score-based)
- **Reranking** — MaxSim (ColBERT late interaction), MMR/DPP diversity, Matryoshka two-stage
- **Evaluation** — NDCG, MRR, recall\@k, fusion parameter optimization

## Quickstart

```toml
[dependencies]
clasp = "0.1.0"
```

```rust
use clasp::{rrf, ScoredItem};

// List A: [doc1, doc2]
let list_a = vec![
    ScoredItem { id: 1, score: 1.0 },
    ScoredItem { id: 2, score: 0.5 },
];

// List B: [doc2, doc3]
let list_b = vec![
    ScoredItem { id: 2, score: 0.9 },
    ScoredItem { id: 3, score: 0.4 },
];

// Fuse with Reciprocal Rank Fusion (k=60)
let fused = rrf(&[&list_a, &list_b], 60.0);

// doc2 should be top because it appears in both
assert_eq!(fused[0].id, 2);
```

## Features

- `serde`: enable serialization for configs and types.

## License

MIT OR Apache-2.0
