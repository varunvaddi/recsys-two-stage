# Offline evaluation report

## 1. Question

Does a two-stage recommender (several candidate generators, then a learned ranker, then a bias-aware re-ranker) beat a strong popularity baseline on Amazon Video Games purchases, and what does it cost in fairness to new items and in latency?

## 2. Setup

- **Data:** Amazon Reviews 2023, Video Games; ratings of 4+ stars as positives; 3-core filter; 229,475 users, 42,325 items.
- **Split:** global time split by timestamp: train (oldest 80%), validation (next 10%), test (newest 10%).
- **Candidates:** top 50 from each of popularity (365 days), ALS, item-CF, two-tower (HNSW search) and content embeddings, merged to about 220 per user.
- **Ranker:** LightGBM LambdaRank with 16 features (source ranks, item popularity and recency, user history, popularity gap, content similarity), early stopping on NDCG@10, trained on 70% of a 30,000-user sample, tested on the other 30%.
- **Re-ranker:** z-scored popularity penalty (lambda) and a number of slots reserved for items absent from the history.
- **Metrics:** Recall@10, NDCG@10, MAP@10, catalog coverage@10, Recall@10 on new items; 95% bootstrap intervals by resampling users.

## 3. Protocol for the final test

The ranker is trained on train-period features with validation-period labels and then frozen. For the test, the candidate sources and all features are refit on train+validation history, scored by the frozen ranker, and judged on the test period. The operating point (2 new-item slots, no popularity penalty) was chosen on validation data before the test period was touched, and the test was run once.

## 4. Results

### 4.1 Final test period

<!-- test:START -->
|  | recall@10 | ci95_low | ci95_high | ndcg@10 | map@10 | coverage@10 | recall@10 new items |
|---|---|---|---|---|---|---|---|
| popularity (365d) | 0.0198 | 0.0185 | 0.0212 | 0.0119 | 0.0072 | 0.0003 | 0.0000 |
| als | 0.0123 | 0.0113 | 0.0135 | 0.0075 | 0.0048 | 0.0482 | 0.0000 |
| itemcf | 0.0055 | 0.0048 | 0.0063 | 0.0035 | 0.0023 | 0.7907 | 0.0000 |
| twotower | 0.0074 | 0.0066 | 0.0082 | 0.0045 | 0.0027 | 0.0144 | 0.0000 |
| content | 0.0068 | 0.0060 | 0.0076 | 0.0039 | 0.0025 | 0.6539 | 0.0057 |
| RANKER | 0.0243 | 0.0228 | 0.0259 | 0.0145 | 0.0092 | 0.1097 | 0.0001 |
| RANKER + 2 new-item slots | 0.0248 | 0.0233 | 0.0264 | 0.0146 | 0.0092 | 0.1618 | 0.0081 |
<!-- test:END -->

### 4.2 Development period (validation)

<!-- dev:START -->
|  | recall@10 | ndcg@10 | map@10 | coverage@10 | recall@10 new items |
|---|---|---|---|---|---|
| top10 of pop | 0.0197 | 0.0125 | 0.0079 | 0.0004 | 0.0000 |
| top10 of als | 0.0135 | 0.0085 | 0.0056 | 0.0424 | 0.0000 |
| top10 of itemcf | 0.0078 | 0.0048 | 0.0033 | 0.5270 | 0.0000 |
| top10 of twotower | 0.0114 | 0.0072 | 0.0047 | 0.0105 | 0.0000 |
| top10 of content | 0.0081 | 0.0044 | 0.0027 | 0.4432 | 0.0062 |
| RANKER (LightGBM) | 0.0340 | 0.0203 | 0.0132 | 0.0506 | 0.0000 |
<!-- dev:END -->

### 4.3 Why the ranker wins, and where it does not

<!-- shap:START -->
| feature | mean |SHAP| | direction (+ = higher value raises score) |
|---|---|---|
| item_pop_recent | 1.04 | 0.6287 |
| content_sim | 0.1982 | 0.9760 |
| best_rank | 0.1129 | -0.9226 |
| item_age_days | 0.0860 | -0.3544 |
| rank_pop | 0.0851 | -0.6646 |
| rank_als | 0.0851 | -0.6871 |
| n_sources | 0.0827 | 0.8905 |
| item_pop | 0.0669 | -0.6522 |
| pop_gap | 0.0484 | -0.8100 |
| rank_twotower | 0.0368 | -0.1095 |
<!-- shap:END -->

<!-- ablations:START -->
|  | recall@10 | ndcg@10 | coverage@10 | recall change |
|---|---|---|---|---|
| full model | 0.0307 | 0.0181 | 0.0389 | 0.0000 |
| - source: pop | 0.0268 | 0.0161 | 0.0647 | -0.1264 |
| - source: als | 0.0311 | 0.0174 | 0.0463 | 0.0148 |
| - source: itemcf | 0.0316 | 0.0181 | 0.0303 | 0.0306 |
| - source: twotower | 0.0322 | 0.0183 | 0.0416 | 0.0485 |
| - source: content | 0.0320 | 0.0187 | 0.0314 | 0.0437 |
| - features: source ranks | 0.0296 | 0.0172 | 0.0305 | -0.0345 |
| - features: item popularity & age | 0.0274 | 0.0156 | 0.0637 | -0.1053 |
| - features: user activity | 0.0308 | 0.0179 | 0.0459 | 0.0044 |
| - features: content similarity | 0.0293 | 0.0175 | 0.0475 | -0.0452 |
<!-- ablations:END -->

### 4.4 Popularity bias and exposure

<!-- bias:START -->
|  | pop | als | itemcf | twotower | content | RANKER | actual (validation) |
|---|---|---|---|---|---|---|---|
| Q1 least popular | 0.0000 | 0.0000 | 0.0018 | 0.0000 | 0.1509 | 0.0003 | 0.3373 |
| Q2 | 0.0000 | 0.0000 | 0.3807 | 0.0000 | 0.1569 | 0.0039 | 0.0727 |
| Q3 | 0.0000 | 0.0000 | 0.1834 | 0.0000 | 0.1665 | 0.0035 | 0.0523 |
| Q4 | 0.0000 | 0.0000 | 0.1302 | 0.0000 | 0.1972 | 0.0086 | 0.1061 |
| Q5 most popular | 1.00 | 1.00 | 0.3038 | 1.0000 | 0.3286 | 0.9837 | 0.4315 |
<!-- bias:END -->

<!-- rerank:START -->
| new_slots | lambda | recall@10 | ndcg@10 | coverage@10 | top20%_share | bottom20%_share | top1%_share | new_item_recall |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.0000 | 0.0340 | 0.0203 | 0.0506 | 0.9837 | 0.0003 | 0.6444 | 0.0000 |
| 0 | 0.5000 | 0.0241 | 0.0138 | 0.2237 | 0.4883 | 0.3675 | 0.1292 | 0.0116 |
| 0 | 1.00 | 0.0137 | 0.0076 | 0.2770 | 0.2080 | 0.5422 | 0.0177 | 0.0157 |
| 2 | 0.0000 | 0.0334 | 0.0201 | 0.1194 | 0.8164 | 0.1602 | 0.5499 | 0.0068 |
| 2 | 0.5000 | 0.0238 | 0.0137 | 0.2239 | 0.4834 | 0.3727 | 0.1280 | 0.0116 |
| 2 | 1.00 | 0.0137 | 0.0076 | 0.2770 | 0.2080 | 0.5422 | 0.0177 | 0.0157 |
<!-- rerank:END -->

### 4.5 Other surfaces

<!-- together:START -->
|  | recall@10 | ndcg@10 | coverage@10 | recall (niche anchor) | recall (mid anchor) | recall (popular anchor) | anchors with a full real list |
|---|---|---|---|---|---|---|---|
| popularity (same list for every game) | 0.0230 | 0.0141 | 0.0004 | 0.0192 | 0.0218 | 0.0279 | – |
| co-occurrence alpha=0.0 | 0.0186 | 0.0115 | 0.1287 | 0.0192 | 0.0170 | 0.0198 | 0.6918 |
| co-occurrence alpha=0.5 | 0.0141 | 0.0080 | 0.2462 | 0.0193 | 0.0108 | 0.0122 | 0.6918 |
| co-occurrence alpha=0.75 | 0.0112 | 0.0060 | 0.2798 | 0.0191 | 0.0099 | 0.0045 | 0.6918 |
| co-occurrence alpha=1.0 | 0.0095 | 0.0052 | 0.2878 | 0.0189 | 0.0080 | 0.0016 | 0.6918 |
<!-- together:END -->

<!-- seq:START -->
|  | recall@10 | ndcg@10 | coverage@10 | recall (short history) | recall (mid history) | recall (long history) |
|---|---|---|---|---|---|---|
| popularity (pop_365d) | 0.0230 | 0.0141 | 0.0004 | 0.0217 | 0.0264 | 0.0208 |
| last game -> co-occurrence (7a) | 0.0186 | 0.0115 | 0.1287 | 0.0209 | 0.0181 | 0.0169 |
| GRU over last games (7b) | 0.0138 | 0.0084 | 0.0083 | 0.0151 | 0.0139 | 0.0126 |
<!-- seq:END -->

### 4.6 Approximate nearest-neighbour search

<!-- faiss:START -->
|  | overlap@100 | recall@10 vs truth | p50 ms | p95 ms | p99 ms | batch QPS | build s |
|---|---|---|---|---|---|---|---|
| exact (numpy) | 1.00 | 0.0139 | 0.4280 | 0.5198 | 0.5542 | 2,884 | 0.0000 |
| faiss flat (exact) | 1.00 | 0.0139 | 0.2097 | 0.2241 | 0.2348 | 57,240 | 0.0018 |
| hnsw efSearch=16 | 0.5822 | 0.0126 | 0.0199 | 0.0321 | 0.0386 | 300,143 | 0.9487 |
| hnsw efSearch=64 | 0.9061 | 0.0139 | 0.0603 | 0.0693 | 0.0758 | 103,880 | 0.9487 |
| hnsw efSearch=256 | 0.9940 | 0.0139 | 0.1445 | 0.1577 | 0.1634 | 43,536 | 0.9487 |
| ivf nprobe=1 | 0.1884 | 0.0114 | 0.0092 | 0.0108 | 0.0110 | 227,131 | 0.2786 |
| ivf nprobe=8 | 0.5963 | 0.0136 | 0.0189 | 0.0225 | 0.0234 | 224,093 | 0.2786 |
| ivf nprobe=32 | 0.9083 | 0.0140 | 0.0362 | 0.0380 | 0.0391 | 159,363 | 0.2786 |
<!-- faiss:END -->

<!-- faiss_scale:START -->
|  | overlap@10 | p50 ms | p95 ms | p99 ms |
|---|---|---|---|---|
| exact (numpy) | 1.00 | 12.39 | 14.66 | 14.97 |
| faiss flat (exact) | 1.00 | 5.67 | 5.72 | 6.49 |
| hnsw efSearch=16 | 0.8532 | 0.0204 | 0.0295 | 0.0352 |
| hnsw efSearch=64 | 0.9944 | 0.0530 | 0.0798 | 0.1055 |
| hnsw efSearch=128 | 1.00 | 0.0892 | 0.1218 | 0.1581 |
| hnsw efSearch=256 | 1.00 | 0.1632 | 0.2006 | 0.2466 |
<!-- faiss_scale:END -->

### 4.7 Serving and load

<!-- serving:START -->
|  | p50 ms | p95 ms | p99 ms | max RPS (1 core) |
|---|---|---|---|---|
| 1_candidate_sources_ms | 2.18 | 2.45 | 2.55 | – |
| 2_features_ms | 0.3155 | 0.3946 | 0.4549 | – |
| 3_ranker_ms | 0.5678 | 0.6356 | 0.6638 | – |
| 4_rerank_ms | 0.2112 | 0.2457 | 0.2728 | – |
| total_ms | 3.32 | 3.66 | 3.84 | 301 |
| related() | 0.0750 | 0.0820 | 0.0858 | – |
| up_next() | 0.0959 | 0.1017 | 0.1047 | – |
| trending() | 0.0764 | 0.0798 | 0.0820 | – |
<!-- serving:END -->

<!-- load:START -->
| setup | concurrency | requests | rps | p50_ms | p95_ms | p99_ms | error_rate |
|---|---|---|---|---|---|---|---|
| 1 worker | 1 | 2183 | 218 | 6.25 | 7.49 | 8.42 | 0.0000 |
| 1 worker | 8 | 3276 | 327 | 27.62 | 39.85 | 58.14 | 0.0000 |
| 1 worker | 32 | 2676 | 265 | 108 | 262 | 501 | 0.0000 |
| 4 workers | 1 | 1734 | 173 | 6.64 | 12.36 | 20.50 | 0.0000 |
| 4 workers | 8 | 5233 | 523 | 11.66 | 39.75 | 51.58 | 0.0000 |
| 4 workers | 32 | 2863 | 284 | 62.02 | 360 | 588 | 0.0000 |
<!-- load:END -->

### 4.8 Drift

<!-- drift:START -->
| feature | psi | status | ref_mean | cur_mean |
|---|---|---|---|---|
| user_last_days | 0.0947 | ok | 1,277 | 1,536 |
| user_n_hist | 0.0442 | ok | 4.64 | 4.83 |
| user_n_recent | 0.0215 | ok | 0.5097 | 0.3679 |
| item_pop_recent | 0.0185 | ok | 3.10 | 2.10 |
| item_age_days | 0.0089 | ok | 2,572 | 2,737 |
| item_pop | 0.0032 | ok | 28.21 | 27.51 |
| user_avg_log_pop | 0.0016 | ok | 4.69 | 4.66 |
<!-- drift:END -->

## 5. Threats to validity

- The development-to-test drop (about +73% to +23% lift) shows that the ranker's advantage depends on how recent its training data is. A fresher ranker has not been measured.
- Offline Recall on logged purchases rewards predicting what people already buy; it undervalues discovery. Coverage and new-item recall are reported to partly offset this, but they are not user-outcome measures.
- Users are sampled (30,000), and each is weighted equally, so heavy and light buyers count the same.
- Source ablations use one seed; differences under about one percentage point of relative recall are not reliable.
- The load test shares a laptop with its load generator, so throughput beyond about 8 concurrent clients reflects the machine as much as the service.