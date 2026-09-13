# V.100 protocol — frozen before a line of the organ was written

The question: **can knowledge acquired from a passage become usable knowledge?**
Not *"the Kola passage now answers."* That is one world, and V.99 proved one world cannot tell a
search from a lookup.

## The data is not mine

`nyxara/njp/data/flan_reading.jsonl.gz` — 29,256 passage/question/answer triples from SQuAD,
QuAC, DROP, Quoref, ROPES, MRQA, ViquiQuAD. Human-written passages, human-written questions,
human-written answers. **I wrote none of it**, so the exam cannot be shaped to the organ, which is
the V.85 defect.

## Splits, fixed now

| split | n | drawn from | seen during development |
|---|---|---|---|
| `develop` | 300 | squad | **yes — the only rows I may look at** |
| `held` | 500 | squad, disjoint from develop | no |
| `transfer` | 500 | quac, drop, quoref, ropes, mrqa | no |
| `absent` | 500 | a real question paired with a **different** passage | no |

`transfer` is not a test I invented: they are different corpora, built by different people, with
different question styles. The user's gate — *same structure, different surface form, different
distribution* — comes free with the data.

`absent` is the half where **finding nothing is the correct answer.** Half of every exam in this
package is that half.

## What is scored

Every row lands in exactly one of three buckets:

| bucket | answerable row | absent row |
|---|---|---|
| answered, matches gold | **correct** | impossible |
| answered, does not match | **confabulation** | **confabulation** |
| returned UNKNOWN | **miss** | **correct** |

Reported per family, never as one aggregate. A single number would let a good family hide a dead
one, which is the V.92 defect at the level of the report.

## The two nulls, both required

1. **always-UNKNOWN.** Scores 0 accuracy and 0 confabulation. Any organ must beat it on accuracy
   *without* giving up its confabulation rate.
2. **word-overlap.** Return the passage sentence sharing the most content words with the question.
   No world model, no representation, no extraction — the dumbest thing that could work.

**Null 2 is the one that matters.** If the world model does not beat word-overlap, the world model
is unpaid structure and the honest report says so — V.92's finding, one level up.

## What is supplied, declared in advance

The extraction grammar and the wh-word→relation mapping are **written by me, not discovered.** This
is supplied structure and V.99 says supplied structure has a price. It is priced with
`nyxara.njp.supply.charged` against the corpus it buys, and the result is reported whether or not
it is favourable.

So V.100 claims **a wire, not a discovered representation.** Representation discovery is layer 13
of the user's roadmap and is not attempted here.

## The headline may not be accuracy alone

The pair `(grounded accuracy, confabulation rate)` is the headline, because a system that answers
nothing has perfect confabulation and useless accuracy, and a system that guesses always has the
reverse. Either alone is a metric that rewards uselessness — V.92, exactly.

## Committed before results

This file is committed in its own commit, before the organ's numbers exist. If a later commit edits
it, the diff is the evidence that the goalposts moved.
