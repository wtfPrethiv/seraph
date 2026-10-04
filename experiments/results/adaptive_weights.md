Views: lexical, semantic, structural; grid step 0.1 (66 cells); 1000 dev queries; norm=minmax. NDCG@10 computed from cached per-view runs.

| weighting | dev NDCG@10 | dev_stdin NDCG@10 |
|---|---|---|
| static tuned (lexical=0.16, semantic=0.64, structural=0.2) | 0.8531 | 0.7396 |
| best grid cell, in-sample (lexical=0.2, semantic=0.5, structural=0.3) | 0.8533 | 0.7434 |
| best grid cell, 5-fold CV | 0.8518 | 0.7418 |
| rule-based adaptive | 0.8505 | 0.7420 |
| learned adaptive, ridge, all cells (5-fold CV) | 0.8413 | 0.7267 |
| learned adaptive, ridge, top 3 cells (5-fold CV) | 0.8516 | 0.7409 |
| learned adaptive, ridge, top 6 cells (5-fold CV) | 0.8510 | 0.7365 |
| learned adaptive, LightGBM, all cells (5-fold CV) | 0.8451 | 0.7289 |
| learned adaptive, LightGBM, top 3 cells (5-fold CV) | 0.8531 | 0.7395 |
| learned adaptive, LightGBM, top 6 cells (5-fold CV) | 0.8525 | 0.7372 |
| per-query oracle (upper bound) | 0.8967 | 0.7966 |
