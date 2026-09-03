# Peering Inside BLIP-2's Q-Former

**Are all 32 of BLIP-2's learned visual queries actually necessary — or is most of the bottleneck dead weight?**

BLIP-2 ([Li et al., 2023](https://arxiv.org/abs/2301.12597)) connects a frozen vision
encoder to a frozen large language model through a small trainable bridge called the
Q-Former. Before any visual information reaches the LLM, it has to pass through 32
learned query vectors — a hard bottleneck the paper justifies conceptually ("forcing
the queries to extract the visual information most relevant to the text") but never
actually measures. Is 32 the right number? Are the queries doing 32 different jobs, or
mostly the same job 32 times? Nobody in the original paper checks.

This project checks. Using the public `Salesforce/blip2-itm-vit-g` checkpoint and a
class-balanced subset of COCO val2017, we ran five experiments that build on each
other, moving from "does the representation carry information" to "does removing
90%+ of it actually break anything" to "does that finding hold up when you look at
which caption actually gets chosen, not just whether it was correct" to, finally,
"does it survive fresh data, an independently-rederived ranking, an exhaustive search
for something better, and an entirely different dataset" — all on the same checkpoint
throughout, no architecture or model swaps between experiments.

**Short version of the result:** it's not dead weight evenly. A handful of the 32
queries are doing essentially all the work; several others are almost completely
inert — one of them (query 14) alone reproduces 95% of the full ensemble's retrieval
performance, while five others (queries 4, 9, 10, 16, 23) contribute *exactly zero*
retrieval signal despite scoring reasonably well on a linear classification probe. The
two evaluation methods disagreed with each other, and figuring out *why* they
disagreed turned out to be one of the most interesting parts of the project — the
other being that this result survived every attempt to break it: a fresh, disjoint
batch of COCO images, an independently re-derived ranking (Spearman r=0.996 against
the original), an exhaustive brute-force search for a better combination, and a
completely different dataset (Flickr30k), see Experiment E.

---

## Table of contents

- [Background: what actually is a Q-Former](#background-what-actually-is-a-q-former)
- [Setup](#setup)
- [Experiment A — Representation Probing](#experiment-a--representation-probing)
- [Experiment B — Query Specialization](#experiment-b--query-specialization)
- [Understanding Recall@K](#understanding-recallk-and-why-this-project-leans-on-it)
- [Experiment C — Query Ablation](#experiment-c--query-ablation)
- [Experiment D — From Ablation to Caption Selection](#experiment-d--from-ablation-to-caption-selection)
- [Experiment E — Generalization Beyond One Sample](#experiment-e--generalization-beyond-one-sample)
- [What we'd actually claim](#what-wed-actually-claim)
- [Limitations](#limitations)
- [Repo contents / how to run it](#repo-contents--how-to-run-it)

---

## Background: what actually is a Q-Former

Skip this section if you already know the architecture.

BLIP-2 never fine-tunes its vision encoder or its language model. Instead, it trains
one small module — the **Q-Former** (188M parameters, ~110M of which are inherited
from BERT-base) — to sit between them. The Q-Former holds **32 learned query
vectors**, each 768-dimensional. These start as generic trainable parameters (no
image-specific content) and, per image, go through:

1. **Self-attention** among the 32 queries (every block) — lets them coordinate/differentiate.
2. **Cross-attention** against the frozen ViT's patch features (every *other* block) —
   the only point where actual image content enters. Each query pulls in a
   weighted mix of the 257 ViT patch tokens (256 patches + 1 CLS token), based on
   what it currently "knows" to look for.

After several such blocks, the 32 queries' final states are projected down to a
shared 256-dim image-text embedding space (for retrieval/matching) or projected into
an LLM's embedding space and prepended as soft visual prompts (for generation). The
paper trains this jointly with three losses on frozen-ViT + trainable-Q-Former,
before ever attaching an LLM:

- **ITC** (contrastive) — pull matching image/caption pairs together, push
  non-matching pairs apart, queries and text encoded independently.
- **ITM** (matching) — binary classifier, is this image-caption pair real or fake,
  full bidirectional attention between queries and text.
- **ITG** (generation) — generate the caption conditioned only on the queries,
  causal attention, forces the queries to carry everything needed for language
  generation.

The checkpoint this project uses, `blip2-itm-vit-g`, is the version Salesforce
further fine-tuned specifically on COCO for retrieval (Sec. 4.4 of the paper) — no
LLM attached, just the vision side + Q-Former + projection heads, which is exactly
the surface this project needs for Experiments A-C.

---

## Setup

- **Model:** `Salesforce/blip2-itm-vit-g` (HF `transformers`), fp16, via
  `Blip2VisionModelWithProjection` + `Blip2TextModelWithProjection`.
- **Dataset:** COCO val2017. Since COCO gives per-object annotations rather than a
  single image label, each image is assigned the *supercategory* (person, vehicle,
  animal, furniture, food, outdoor, indoor, appliance, electronic, kitchen,
  accessory, sports) that covers the most pixel area. Capped at 150 images per
  supercategory (COCO is heavily skewed toward "person") to avoid the probing
  experiments just measuring "can you detect a person." Final subset: **1,599
  images across 12 classes** (two rare categories, "kitchen" and "accessory"/"sports,"
  don't reach the 150 cap: 98 and 81/72 images respectively).
- **Hardware:** single T4 GPU (Colab), fp16 inference throughout.

---

## Experiment A — Representation Probing

**Question:** does the Q-Former's 32-query output carry *more linearly-accessible*
semantic information than the raw frozen ViT features it's built from — or does
compressing 257 patch vectors down to 32 queries lose information?

**Method:** for every image, mean-pool the raw ViT patch features (256 patches,
excluding CLS) into one 1408-dim vector, and separately mean-pool the Q-Former's 32
query outputs into one 768-dim vector. Train two probes on each representation to
predict the supercategory label: a logistic regression (tests linear separability)
and a 5-NN classifier (tests whether same-class points cluster locally, even without
a clean linear boundary).

**Result:**

| representation | logistic regression acc. | k-NN acc. |
|---|---|---|
| Raw ViT (mean-pooled patches) | 0.766 | 0.691 |
| Q-Former (mean-pooled queries) | 0.762 | **0.762** |

The linear-separability numbers are essentially tied (0.766 vs. 0.762) — the
Q-Former isn't handing you *more* raw class information than the ViT patches had.
But k-NN accuracy jumps from 0.691 to 0.762 for the Q-Former space. That's the
interesting part: the Q-Former isn't adding information, it's **reorganizing** it —
pulling same-class images closer together in its 32-query space than they were in
raw ViT-patch space, without actually increasing how linearly separable the classes
are. This is the first hint that whatever the Q-Former is doing, it's not simple
compression — it's restructuring the geometry around whatever it was trained to be
useful for (matching images to captions), which doesn't have to line up with
"good for a downstream linear classifier."

---

## Experiment B — Query Specialization

**Question:** are the 32 queries doing 32 different jobs, or mostly duplicating each
other?

Four angles on the same question, all using the *full* per-query state (32, 768) —
not mean-pooled this time — plus the Q-Former's cross-attention maps (averaged over
6 cross-attention layers × 12 heads):

**1. Redundancy.** Average pairwise cosine similarity between all 32 queries, within
each image, averaged across the dataset. Mean off-diagonal similarity: **0.466**,
max: **1.0** — some query pairs are producing near-identical vectors for every image,
which is a strong hint of dead/duplicate capacity before we even look at anything
else.

**2. Attention entropy.** How spatially focused vs. diffuse each query's attention
over the image is. Max possible entropy (uniform attention over all 257 patches) is
5.549 nats; even the *most focused* query (index 17) only reaches 4.034 nats, and the
most diffuse (index 7) reaches 4.577. In other words, no query is doing tight,
localized "object detector" style attention — they're all fairly spread out, just to
varying degrees. This matters for interpreting the heatmaps: don't expect
crisp bounding-box-like attention.

**3. Per-query probe importance.** Train a *separate* linear probe on each query's
raw 768-d vector alone (not pooled with the others), rank all 32 by standalone
classification accuracy:

- Best 5: queries **14, 19, 7, 27, 8** (accuracy 0.767–0.787)
- Worst 5: queries **18, 13, 29, 26, 5** (accuracy 0.670–0.710)
- Overall mean across all 32: 0.736 ± 0.027

**4. Dead-query detection.** Cross-image variance per query (`qformer_full.var(axis=0).mean(-1)`
— how much a query's output actually *changes* from image to image). Five queries
stand out as essentially frozen: **4, 9, 10, 16, 23**, each with cross-image variance
≈ 0.0007, roughly two orders of magnitude below the typical "living" query
(0.09–0.15). Query 18 sits in between (variance 0.026) — semi-dead.

**The catch, and the reason Experiment C exists:** those five near-dead queries
still score 0.726–0.728 on the classification probe — barely below the 0.736 average
across all 32 queries. If you only looked at the probe-importance ranking, you'd
conclude these queries are contributing *almost as much as everything else*. They
aren't. `StandardScaler` rescales every feature to unit variance before the
classifier sees it, so a query whose raw output barely changes between images still
gets blown up to full scale — and if *any* tiny, consistent correlation with class
label survives that rescaling (even from what's functionally noise), logistic
regression can exploit it. Classification probes, in other words, are structurally
bad at telling "genuinely useful" apart from "technically nonzero." This is exactly
why Experiment C switches to a metric that doesn't get to rescale anything.

---

## Understanding Recall@K (and why this project leans on it)

Recall@K is the standard metric for **image-text retrieval**, and it's worth being
precise about it because Experiment C's whole argument depends on it being a sharper
instrument than a classification probe.

**Setup.** You have N images and N captions (one real, correct caption per image).
Encode every image and every caption into the same embedding space. For image *i*,
rank *all N candidate captions* by similarity to image *i*'s embedding. **Recall@K**
is the fraction of images for which the *one correct* caption lands somewhere in the
top K of that ranking.




A few things that matter for reading the numbers correctly:

- **It gets easier as K grows.** Recall@1 demands the correct caption be the single
  best match out of N candidates; Recall@10 only demands it be somewhere in the top
  10. Recall@K is monotonically non-decreasing in K by construction — always report
  which K you mean.
- **Chance level depends on the pool size N**, not on K alone. With N = 1,599
  candidate captions, a model outputting random embeddings would get Recall@1 ≈
  1/1599 ≈ **0.0006**. Our full-model Recall@1 of 0.662 is therefore roughly **1,100×
  above chance**, not just "66%" in isolation — that context matters when a number
  like 0.632 (the single-query result) looks like a big drop from 0.662 but is still
  essentially the same distance from random guessing.
- **It's bidirectional.** Image→Text Recall@K (given an image, find its caption)
  and Text→Image Recall@K (given a caption, find its image) are two separate
  numbers computed the same way with the roles swapped, and they don't have to
  match (though in our run they came out identical at K=1: 0.662 both ways, and
  close at K=5/K=10: 0.909 vs. 0.894, 0.956 vs. 0.948).
- **BLIP-2's own similarity convention matters here.** Because the image side
  produces *32* query embeddings instead of one, image-text similarity isn't a
  single cosine similarity — the paper defines it as the **maximum** cosine
  similarity across the image's 32 queries against the text embedding (not the mean,
  not a learned combination — literally the max). That single design choice is what
  makes Experiment C possible: if you restrict "which queries participate in that
  max" to a chosen subset (even a subset of size 1), you get Recall@K computed on
  exactly the ablated set of queries, with zero change to the underlying formula.

That's the whole trick behind Experiment C: **per-query Recall@1** just means
running the exact same Recall@K computation, but replacing "max over all 32
queries" with "the similarity from one specific query, alone."

---

## Experiment C — Query Ablation

**Question:** how many of the 32 queries are actually load-bearing for retrieval, and
does *informed* pruning (guided by each query's own standalone usefulness) hold up
better than *naive random* pruning at the same query count?

**Setup:** image-text retrieval on the same 1,599-image COCO subset, one caption per
image (first COCO annotation). Full-model similarity = max cosine similarity across
the 32 per-query embeddings vs. the caption embedding, exactly as above.

**Baseline (all 32 queries):**

| direction | R@1 | R@5 | R@10 |
|---|---|---|---|
| Image → Text | 0.662 | 0.909 | 0.956 |
| Text → Image | 0.662 | 0.894 | 0.948 |

**Per-query Recall@1** (using each query alone, no max-pooling over the others) gives
a ranking: best individual query is **14** (R@1 = 0.632 — essentially matching the
*entire 32-query ensemble* on its own), followed by 27, 8, 7, 19. The worst five —
**4, 9, 10, 16, 23** — score **exactly 0.000**. Not "low." Zero. These are the same
five queries Experiment B flagged as near-frozen by cross-image variance, and now
retrieval confirms it in the sharpest way possible: a query whose output barely
changes across 1,599 different images is, unsurprisingly, useless for telling those
images apart.

This also resolves Experiment B's puzzle: retrieval R@1 correlates strongly with
both cross-image variance (Spearman r = 0.694, p < 0.0001) and classification probe
accuracy (r = 0.788, p < 0.0001) — so the classification probe *was* detecting a real
effect, it was just compressed into a narrow 0.67–0.79 range that made "dead" and
"useful" nearly indistinguishable. Retrieval spreads the exact same underlying effect
out from 0.000 to 0.632 — an unambiguous signal the probe was too coarse to surface.

**Ablation — how many queries can you drop?**

| queries kept | informed R@1 | random R@1 (mean ± std, 10 trials) |
|---|---|---|
| 32 | 0.662 | 0.662 ± 0.000 |
| 16 | 0.663 | 0.655 ± 0.005 |
| 8  | 0.664 | 0.622 ± 0.030 |
| 4  | 0.660 | 0.538 ± 0.142 |
| 2  | 0.650 | 0.488 ± 0.126 |
| 1  | 0.632 | 0.412 ± 0.195 |

Two findings stacked on top of each other:

1. **Informed pruning barely costs anything.** Guided by each query's own
   standalone Recall@1, you can throw away 31 of the 32 queries and keep 95% of full
   performance (0.632 vs. 0.662). The "bottleneck," in other words, has enormous
   redundancy in it — most of the 32 slots aren't adding retrieval-relevant
   information the best one or two don't already have.
2. **Random pruning degrades badly, and unpredictably.** At k=4, random selection
   averages 0.538 (vs. 0.660 informed) with a standard deviation of 0.142 — a
   coin-flip's worth of a chance of landing on a dead query and collapsing
   performance. At k=1, that std balloons to 0.195. Informed selection isn't just
   better on average; it removes the *variance* entirely, because it's the specific
   identity of the query, not the count, that determines whether you get useful
   signal.

---

## Experiment D — From Ablation to Caption Selection

Experiment C only checks whether the *correct* caption gets retrieved in the
top-K. Experiment D asks a stricter question: even when the single-query model
doesn't retrieve ground truth, **how similar is what it picks to what the full
32-query ensemble would have picked?** That's a harder bar to clear — no credit
for matching ground truth some fraction of the time, it has to actually agree
with the full model's own choice.

This stays entirely on `blip2-itm-vit-g` — no second checkpoint, no text
generation, no new model download. `blip2-itm-vit-g` has no language model
attached, so "generation" was never really on the table for this checkpoint;
what we can ask instead is which caption it *selects* from the candidate pool,
which is exactly the operation Experiment C's Recall@K already performs — we're
just reading out the actual selected caption instead of a yes/no about whether
it was correct.

For every image, three "selected" captions: using the full 32-query
max-similarity (BLIP-2's own scoring convention), using only query 14 (the
single best query from Experiment C's ranking), and using one random query per
image as a control. The informed and random selections are compared against the
full-model selection using the caption embeddings already computed in
Experiment C (`text_embeds_all`) — no re-embedding needed. That gives two
similarity distributions across images (`sim_top`, `sim_rand`), compared with
KL divergence.

Because this is pure matrix algebra on embeddings already sitting in memory,
there's no reason to subsample — it runs over the full 1,599-image dataset, so
unlike a generation-based version of this experiment there's no small-sample
noise to caveat.

**Result:**

| condition | mean similarity to full-model selection | exact-match rate |
|---|---|---|
| top query (q14) | 0.968 ± 0.076 | 0.814 |
| random query | 0.857 ± 0.151 | 0.426 |

KL(top || random) = **0.3321 nats**, KL(random || top) = **0.5126 nats**.
The asymmetry itself is informative: KL(random || top) is larger because
the "top query" distribution is almost entirely piled into the [0.9, 1.0] bin —
under that as the reference, the random-query distribution's real spread into
the 0.4–0.8 range looks very surprising. Going the other direction, the
random-query distribution has enough spread to already assign some probability
mass everywhere the top-query distribution has mass, so it's "less surprised"
by comparison. Same underlying gap, read from two directions.

*(The notebook saves this plot to `caption_selection_similarity_distributions.png`
in the Colab runtime — download it from Colab's file browser and drop it next to
this README to make the image link above resolve.)*

Query 14's selections agree with the full 32-query ensemble's own pick on 81.4%
of images, and even where they disagree, they land close (mean cosine similarity
0.968) — the histogram is a hard spike against the right edge. A random query
agrees only 42.6% of the time, with much more mass pulled down into the
0.4–0.8 range.

The comparison worth sitting with: query 14's *exact-match rate against the full
model* (0.814) is meaningfully higher than its *Recall@1 against ground truth*
from Experiment C (0.632). Those are different targets — R@1 asks "did you find
the correct caption," this asks "did you make the same call the full ensemble
made" — and the gap between them says something specific: a good chunk of query
14's "wrong" answers (by the ground-truth standard) are the *same* wrong answer
the full 32-query model also reaches, not different mistakes. Query 14 isn't
just occasionally right and otherwise noisy — it's tracking the full ensemble's
actual decision boundary, errors included.

The qualitative examples make the gradation visible directly. Some images are
easy enough that all three regimes (full/top/random) land on the identical
caption text. Others show the full model itself missing ground truth (e.g. an
image whose true caption was "A nearly empty plate containing broccoli and
brown sauce" — the full model instead picked "A plate of broccoli and meat on a
table"), and query 14's pick stays topically close ("a white bowl filled with
meat, vegetables and broth") while the random-query pick drifts further off
("a plate of food next to eating utensils"). The single query isn't just
matching ground truth some of the time — it's tracking what the full ensemble
attends to, mistakes included, in a way a random query demonstrably doesn't.

---

## Experiment E — Generalization Beyond One Sample

Everything up to Experiment D was found and measured on the *same* 1,599 images.
Two gaps follow directly from that: (1) the query ranking was derived and evaluated
on the same data, which is the retrieval-metrics equivalent of tuning on your test
set, and (2) "top-k" so far means the k individually-best queries, greedily assumed
to combine well — never checked against the actual best possible *combination*.
Experiment E addresses both, plus checks whether the finding survives moving to an
entirely different dataset — the same zero-shot-transfer protocol the BLIP-2 paper
itself uses (Table 5: COCO-finetuned, evaluated zero-shot on Flickr30k).

**A note on scale**: COCO val2017 only has 4,952 usable labeled images total, and the
original 1,599-image subset already used *every single available image* in three of
the twelve supercategories (72 "sports", 81 "accessory", 98 "kitchen" — all consumed).
A second class-balanced subset of the same shape is therefore impossible. That's fine:
retrieval never used the class labels to begin with (those were only ever needed for
Experiments A/B's classification probes), so the holdout batch below is simply the
next 1,599 unused images, unbalanced by design.

### E1 — a second, disjoint 1,599-image COCO batch

3,353 images remained unused after the original subset. A fresh, unbalanced batch of
1,599 was drawn from them (label distribution is informational only — retrieval
doesn't touch it): person 682, vehicle 295, animal 260, furniture 235, food 55,
indoor 31, outdoor 28, appliance 13 (kitchen/accessory/sports: 0 remaining, as
expected). Its embeddings were extracted with the identical pipeline used throughout.

### E2 — does the discovery-set ranking still win on data it never saw?

`retrieval_ranking` (query 14 first, derived entirely from the original 1,599) was
evaluated on this fresh holdout batch — queries it played no part in selecting from:

| queries kept | informed (discovery-ranked) R@1 on holdout | random R@1 on holdout |
|---|---|---|
| 32 | 0.633 | 0.633 ± 0.000 |
| 16 | 0.635 | 0.629 ± 0.006 |
| 8  | 0.635 | 0.600 ± 0.027 |
| 4  | 0.632 | 0.512 ± 0.137 |
| 2  | 0.635 | 0.451 ± 0.122 |
| 1  | 0.609 | 0.395 ± 0.189 |

(Full 32-query R@1 on holdout is 0.633 vs. 0.662 on discovery — ordinary sample
variation, not a red flag.) Query 14 alone: **0.609** vs. random's **0.395** — nearly
the identical gap seen on the original data (0.632 vs. 0.412). **The finding
generalizes to images it never touched.**

### E3 — split-half reliability: re-derive the ranking from scratch, independently

Rather than reuse the original ranking, all 32 queries were re-ranked **using only
the holdout set** — a completely independent computation, zero information shared
with the original ranking:



**Spearman r = 0.996** (p < 0.0001) between the two independently-derived 32-length
rankings. Top-5 overlap: 5/5 (queries 14 and 27 simply swap first/second place). This
is the sharpest test in the whole project — two separate 1,599-image samples,
ranked from scratch with no shared information, landing on almost the exact same
answer is strong evidence that "query 14 matters" is a real, stable property of this
checkpoint's Q-Former, not noise from one particular batch of photos.

### E4 — is greedy top-k actually the best possible combination?

For k = 1, 2, and 4, every possible combination of that many queries was checked
exhaustively (32, then 496, then 35,960 combinations) against the greedy top-k
(the k individually-best queries by the ranking above), all on the discovery set:

| k | brute-force-optimal combo | discovery R@1 | greedy top-k combo | discovery R@1 | match? |
|---|---|---|---|---|---|
| 1 | (14,) | 0.632 | (14,) | 0.632 | MATCH |
| 2 | (14, 27) | 0.650 | (14, 27) | 0.650 | MATCH |
| 4 | (2, 19, 22, 27) | 0.664 | (14, 27, 8, 7) | 0.660 | DIFFERENT |

At k=1 and k=2, greedy composition already finds the true optimum — no missed
synergy. At k=4, brute force found a marginally better combination (+0.004). C(32,8)
= 10.5 million combinations puts k=8 out of practical reach for exhaustive search, so
that count stays greedy-vs-random only (from E2/Experiment C).

**But which combo actually generalizes better?** Both the greedy and brute-force-optimal
combos (found using *only* the discovery set) were evaluated on the untouched holdout:

| k | greedy → holdout R@1 | brute-force-optimal → holdout R@1 |
|---|---|---|
| 1 | 0.609 | 0.609 |
| 2 | 0.635 | 0.635 |
| 4 | **0.632** | 0.630 |

At k=4, the "smarter" brute-force combo — despite scoring higher on the data it was
optimized on — scored *lower* on holdout than the simpler greedy combo. A small,
caught-in-the-act example of overfitting: exhaustively searching for the single best
combination on one specific sample can find something that exploits quirks of that
sample rather than a genuinely more useful pair of queries.

### E5 — cross-dataset transfer: Flickr30k

The COCO-derived ranking was tested on 1,000 Flickr30k image-caption pairs — a
completely different photo distribution and caption-writing style, with zero COCO
involvement in how the ranking was chosen. (Loaded via
`nlphuji/flickr_1k_test_image_text_retrieval`, using the Hub's auto-converted
Parquet branch — the dataset's own legacy loading script is no longer supported by
current `datasets` versions.)

| queries kept | informed (COCO-ranked) R@1 on Flickr30k | random R@1 on Flickr30k |
|---|---|---|
| 32 | 0.877 | 0.877 ± 0.000 |
| 16 | 0.878 | 0.859 ± 0.007 |
| 8  | 0.875 | 0.820 ± 0.034 |
| 4  | 0.857 | 0.710 ± 0.151 |
| 2  | 0.854 | 0.636 ± 0.134 |
| 1  | 0.823 | 0.555 ± 0.245 |

Query 14 alone: **0.823** vs. random's **0.555** on a dataset it never saw during
ranking selection. (Flickr30k's higher absolute numbers across the board reflect that
it's a generally easier retrieval benchmark — more visually distinct images — not
that the finding is stronger here; the *relative* informed-vs-random gap is what
matters, and it holds.) **The finding is not a COCO artifact.**

### E — what this adds up to

Four independent challenges — new images, an independently re-derived ranking, an
exhaustive search for something better, and an unrelated dataset — and the original
Experiment C/D claim survived all four, with one honest nuance surfaced along the way
(brute-force search can mildly overfit at k=4). That's a meaningfully stronger
evidentiary bar than "we measured a number once."

---

## What we'd actually claim

- BLIP-2's 32-query bottleneck is **not** using all 32 slots equally: on this
  checkpoint and dataset, a small handful of queries (14, 19, 7, 27, 8 by
  probe/retrieval ranking) carry almost all the retrieval-relevant signal, while at
  least 5 (4, 9, 10, 16, 23) are functionally dead — their output barely responds to
  the input image at all.
- **Classification-probe accuracy is a misleading way to measure this.** It compresses
  a 0.000-to-0.632-wide real effect (Recall@1) into a narrow, hard-to-read
  0.67-to-0.79 band, because feature standardization rescales even near-constant
  signal up to full scale. If you want to know whether a component of a model is
  doing real work, measure it with a metric close to what the component is *for*
  (here, retrieval similarity), not a generic downstream classifier.
- **A single query tracks the full ensemble's actual decisions, not just ground truth.**
  Query 14 matches the full model's own top-1 caption selection 81.4% of the time
  (mean similarity 0.968) — noticeably higher than its 63.2% Recall@1 against
  ground truth from Experiment C. The gap means a meaningful share of query 14's
  "wrong" answers by the ground-truth standard are the *same* wrong answer the
  full 32-query model also lands on. A random query matches the full model only
  42.6% of the time (KL divergence between the two agreement distributions:
  0.33–0.51 nats depending on direction) — confirming this isn't just "single
  queries are noisy but occasionally lucky," it's a specific, reproducible
  query that's doing the ensemble's job almost alone.
- **None of the above is an artifact of the one 1,599-image sample it was found on.**
  Re-deriving the query ranking from scratch on a completely independent 1,599-image
  batch gives Spearman r=0.996 against the original — about as close to "the same
  answer" as two separate samples can get. The gap between informed and random
  query selection survives unchanged on that fresh batch (0.609 vs. 0.395 at k=1),
  and survives moving to an entirely different dataset, Flickr30k (0.823 vs. 0.555),
  with zero Flickr data involved in choosing which queries mattered.
- **Greedy composition (take the k individually-best queries) is nearly optimal, but
  not perfectly so, and "more optimal" isn't always "generalizes better."** An
  exhaustive search over all 35,960 possible 4-query combinations found one that
  beat greedy selection by 0.004 on the data it was searched on — but that same
  "better" combination scored *lower* than greedy on a fresh holdout batch. Small
  and specific, but a real, caught-in-the-act example of overfitting to one sample.

## Limitations

- Single COCO-derived label per image (dominant supercategory by pixel area) is a
  coarse proxy for "semantic content" — a finer-grained or multi-label setup might
  surface different query specializations. (This only affects Experiments A/B;
  Experiments C–E are retrieval-based and never use the label at all.)
- Every experiment here uses one checkpoint (`blip2-itm-vit-g`). We have not checked
  whether the same 5 "dead" query indices are dead on a *different checkpoint* (e.g.
  `blip2-opt-2.7b`, `blip2-flan-t5-xl`) — query indices carry no inherent
  cross-checkpoint meaning, so this finding is specific to this checkpoint's Q-Former
  until shown otherwise. Generalization *across datasets* (COCO holdout, Flickr30k)
  was tested directly in Experiment E and held up; generalization *across
  checkpoints* remains untested and is the most natural next step.
- Experiment E's brute-force search was limited to k = 1, 2, 4 — C(32,8) is 10.5
  million combinations, well past what's practical to exhaustively check here, so
  whether greedy composition holds up at larger k is untested (informed-vs-random
  at those sizes is still covered, just not informed-vs-true-optimal).



