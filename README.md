# clasp

Rank fusion primitives for hybrid search.

`clasp` implements small, explainable fusion operators over multiple retrieval result lists:
RRF, ISR, CombMNZ, Borda, DBSF, and weighted variants.

This crate is used as the fusion stage in `cerno` (re-exported as `cerno::fusion`).

## Features

- `serde`: enable serialization for configs and types.

## License

MIT OR Apache-2.0
