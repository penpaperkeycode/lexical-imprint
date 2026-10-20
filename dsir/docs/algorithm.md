# DSIR Algorithm — paper-aligned description

## Setup

Given:
- A **source corpus** $D_s$ of $N_s$ documents (e.g., DCLM, FineWeb-Edu).
- A **target distribution** represented by a small benchmark corpus $D_t$
  (here: CSR7 — ARC-E, ARC-C, HellaSwag, SciQ, OpenBookQA, PIQA, SWAG).
- A budget $K \ll N_s$ — number of source rows to select for downstream training.

Goal: pick $K$ source rows whose n-gram distribution best matches the target.

## Hashed n-gram features

For each text $x$:
1. NFKC normalize, lowercase, whitespace tokenize → token list.
2. Extract all unigrams + bigrams (n_max = 2).
3. Hash each n-gram with murmurhash3 (`mmh3.hash`, unsigned) mod $B$ (B=10 000).

This produces a multiset of bucket ids per text, of cardinality $\approx 2|x|$.

## Bucket count multinomials

- Build target bucket counts $C_t \in \mathbb{Z}^B$: for every n-gram in every
  doc of $D_t$, increment its bucket.
- Same for source: $C_s$.

Apply Laplace add-$\alpha$ smoothing ($\alpha=1$):

$$
P_t(b) = \frac{C_t(b) + \alpha}{\sum_{b'}\bigl(C_t(b') + \alpha\bigr)}
\qquad
P_s(b) = \frac{C_s(b) + \alpha}{\sum_{b'}\bigl(C_s(b') + \alpha\bigr)}
$$

## Per-document log importance weight

For source row $x$:

$$
\log iw(x) = \sum_{g \in \text{ngrams}(x)}\bigl[\log P_t(\phi(g)) - \log P_s(\phi(g))\bigr]
$$

(Sum, not average — paper's choice. Documents with more in-target n-grams accumulate
larger weight.)

We pre-compute the per-bucket vector $r = \log P_t - \log P_s$ once (Phase C)
and for each source row simply sum $r$ over its hashed n-grams (Phase D).

## Top-K deterministic selection

We keep the $K$ rows with the largest $\log iw(x)$, ties broken by row index
(smaller idx kept).

> The DSIR paper proposes a Gumbel-top-K sampling variant that is statistically
> equivalent to importance-weighted resampling without replacement. We use the
> deterministic top-K variant to match our paper's CE-filter framework — both
> methods rank source rows by an analytical score and keep the top fraction.

## Reference

Xie, S. M., Santurkar, S., Ma, T., & Liang, P.\ (2023). *Data Selection for
Language Models via Importance Resampling.* In Advances in Neural Information
Processing Systems 36.
