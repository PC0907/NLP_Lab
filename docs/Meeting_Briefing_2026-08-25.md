# Meeting Briefing — everything, from the beginning

**For:** Adnan, before the supervisor meeting on 2026‑08‑25
**Gap since last meeting:** ~36 days (exam period)
**How to use this:** read Parts 1–4 to understand the project, Part 5–7 for what
we found, Part 8 for the questions he will ask, Part 9 for how to open.

Everything is explained with one running example that appears in every section,
so you only ever have to keep one story in your head.

---

# PART 1 — The problem, in plain language

## 1.1 What the project is about

Companies use language models to read text and fill in a form. You give the model
a document and a template, and it returns structured data — JSON.

**The problem:** the model never tells you which parts it got wrong. It fills in
every box with equal confidence. If it invents a number, it looks exactly like a
number it read correctly.

**An analogy.** Imagine an intern who fills in 100 forms for you overnight. About
a third of the entries are wrong, but they don't mark which ones. You have two
bad options: check all 100 yourself (expensive), or trust all 100 (risky). What
you actually want is for the intern to say *"I'm not sure about these 12."* Then
you check 12 instead of 100.

**Our project builds that "I'm not sure" signal.** We call it a **trust signal**.

## 1.2 The running example

Here is one record from our dataset. Keep this in mind for the whole document.

**Question:** *"The band that recorded 'Kashmir' — in which year was it formed,
and who was its lead guitarist?"*

**Context:** a few Wikipedia paragraphs about Led Zeppelin.

**Template (schema):**
```json
{ "band_name": string, "formation_year": integer, "lead_guitarist": string }
```

**The correct answer (gold):**
```json
{ "band_name": "Led Zeppelin", "formation_year": 1968, "lead_guitarist": "Jimmy Page" }
```

**What the model actually produced:**
```
<think>
The song "Kashmir" is on Physical Graffiti, which is a Led Zeppelin album.
Led Zeppelin formed in London in 1968. Their guitarist was Jimmy Page, who
started the band after the Yardbirds broke up.
</think>

{ "band_name": "Led Zeppelin", "formation_year": 1969, "lead_guitarist": "Jimmy Page" }
```

Look carefully: **the model's own reasoning says 1968, but the JSON says 1969.**
Two fields right, one wrong. Nobody told us which — that's the whole problem.

## 1.3 Why a "reasoning model"

We use **DeepSeek‑R1**, which writes out its thinking inside `<think>...</think>`
before answering. Most models just answer. This one shows its work.

That gives us something extra to look at: not only *what* the model answered, but
*how it got there*. **The central question of my half of the project is whether
that thinking‑out‑loud contains information about which answers are wrong.**

---

# PART 2 — The idea: reading the model's mind

## 2.1 What a "hidden state" is

When a model processes text, it doesn't jump straight to an answer. It passes the
text through a stack of **layers** — ours has 28. At each layer, every word is
represented as a long list of numbers (**3,584** of them for this model). That
list is the **hidden state** — the model's internal representation at that point.

**Analogy.** Think of a factory assembly line with 28 stations. At station 1 the
product is raw material; by station 28 it's finished. If you photograph the
product at station 19, that photo tells you a lot about what it's becoming. The
hidden state is that photograph — 3,584 numbers describing the model's internal
state at that moment.

## 2.2 What a "probe" is

A **probe** is a small, simple classifier we train on those hidden states.

For each field the model produced, we take the hidden state at the moment it
wrote that field, and train a model to predict: **was this field right or wrong?**

We use **logistic regression** — the simplest possible classifier, essentially a
weighted sum. This is deliberate. If something as simple as a weighted sum can
read "this is wrong" out of the hidden state, that means the information is
sitting there in an easily accessible form. A complicated classifier might
*create* the appearance of signal by finding elaborate patterns; a simple one
can't.

**Analogy.** A doctor takes a blood sample and reads one number off it. If that
one number reliably predicts illness, the illness clearly shows up in the blood.
That's a stronger claim than "a machine-learning system with a thousand
parameters can predict it."

## 2.3 What we do with the trust signal

Once every field has a risk score, we do **selective regeneration**: ask the model
again only for the fields most likely to be wrong.

**Analogy.** Back to the intern. They flag 20 of 100 entries. You only re-do
those 20. If most of the actual mistakes are inside those 20, you fixed most of
your problem for a fifth of the work.

---

# PART 3 — The pipeline, stage by stage

Seven steps, each producing a file the next one reads.

**Stage 1 — Extraction (needs a GPU).**
Feed the model the question + context + schema. Save: the JSON it produced, its
reasoning trace, its confidence in each word, and the hidden states.
*Our example:* saves the JSON above, the `<think>` text, and a 3,584-number
vector for each of `band_name`, `formation_year`, `lead_guitarist`.

**Stage 2 — Labeling.**
Compare against gold, mark each field right or wrong.
*Our example:* `band_name` ✓, `formation_year` ✗, `lead_guitarist` ✓.

**Stage 3 — Train the probe.**
Train logistic regression on the hidden states to predict the labels.

**Stage 4 — Compare to baselines.**
Check whether the probe beats the free alternative (Part 4.7).

**Stage 7 — The reasoning experiment.** *(my main contribution)*
Try different ways of adding reasoning‑trace information to the probe.

**Stage 8 — Controls.** *(the rigour)*
Try to break our own result and see if it survives.

**Stage 9 — Selective regeneration.**
Turn the scores into a cost‑versus‑quality curve.

*(Stages 5 and 6 were earlier versions, now superseded.)*

---

# PART 4 — The vocabulary

**This section is the one to memorise.** Every term he might ask about.

## 4.1 Field

One box in the form. Our example has three: `band_name`, `formation_year`,
`lead_guitarist`. Everything we measure is *per field*, not per document — that's
the point, we want to know *which* box is wrong.

## 4.2 The error label (`is_error`)

1 if the field is wrong, 0 if right. `formation_year` gets a 1.

## 4.3 Structure‑aware matching — and a bug we fixed

Early on, our error rate came out around **95 %**, which was obviously wrong — the
model isn't that bad. The cause: the model would answer `"percussion"` while the
gold stored `{"family": {"name": "percussion"}}`. Same answer, different shape,
and naive string comparison called it wrong.

We built a matcher that compares *values* rather than *shapes*, plus number
tolerance and date parsing. The error rate dropped to a realistic **~42 %**.

**Why this matters:** it's a genuine contribution. Anyone measuring extraction
accuracy this way would have reported a badly inflated error rate.

We report **three** matching strictnesses so nobody thinks we tuned it to
flatter ourselves: strict 46.2 %, auto 42.9 %, structure‑aware 42.7 %.

## 4.4 Layer

Which of the 28 assembly-line stations we photograph. We tested 14 of them. The
middle-to-late layers work best — around **layer 19–21**.

**Why the middle?** Early layers are still working out what the words mean. Late
layers are focused on producing the next token. The middle holds the model's
"understanding," which is where doubt lives.

## 4.5 AUROC — our main measurement

A number from 0.5 to 1.0 that answers: *if I pick one wrong field and one right
field at random, how often does my score rank the wrong one as riskier?*

- **0.5** = coin flip, useless
- **0.80** = right 80 % of the time
- **1.0** = perfect

Ours is **0.83**. In plain terms: *given one good field and one bad field, our
signal correctly identifies the bad one 83 % of the time.*

## 4.6 LODO — the honest test

**LODO = Leave‑One‑Document‑Out.**

Normal cross-validation shuffles all fields and splits randomly. That's too easy:
fields from the *same document* can land in both training and testing, so the
probe can memorise document‑specific quirks and look better than it is.

LODO instead holds out **one entire document**, trains on the other 973, and
tests on the held‑out one — 974 times.

**Analogy.** Studying for an exam with the actual exam questions mixed into your
practice set versus studying from a completely different textbook. LODO is the
second, and it's the number we report.

**Have this ready:** *"We use leave-one-document-out because random splits leak
information across fields of the same document and inflate the score."*

## 4.7 Baselines — what we must beat

The model already gives a free confidence signal: how certain it was about each
word (**log‑probability**). If our probe can't beat that, it isn't worth the
trouble.

- mean log‑prob: **0.750**
- min log‑prob: **0.753**
- **our probe: 0.827**

We beat the free option by **+0.074 AUROC**.

## 4.8 per‑doc AUROC vs pooled AUROC

Two different questions:

- **per‑doc:** *within this one document, can I rank its wrong fields above its
  right ones?* Computed per document, then averaged.
- **pooled:** *across all 5,071 fields from all documents, can I rank the wrong
  ones on top?*

Both matter. Per-doc is "which box in this form should I check." Pooled is "I
have a budget for 1,000 checks across the whole batch — which 1,000?"

We report both, and they agree.

## 4.9 The reasoning trace and the attribution vector

The `<think>` text is a stream of words, each with its own hidden state.

**Attribution** means: for a given field, find where its *value* appears in the
reasoning, and take the hidden states of exactly those words.

*Our example:* for `lead_guitarist` = "Jimmy Page", we find "Jimmy Page" in the
trace and average the hidden states of those two words. That average is the
**attribution vector** (`attr_vec`) — our attempt at capturing "how the model
thought about *this specific field*."

For `formation_year` = "1969", we search the trace for "1969" and **find
nothing** — the trace says 1968. So its attribution vector is all zeros, and a
flag records *"this value was never reasoned about."* That's a red flag, and in
this case the field really is wrong.

## 4.10 The variants — what "fused_attr" and the others mean

**This is what you asked about directly.** "Fused" simply means *glued together*.
The probe takes a list of numbers as input; each variant is a different list.

**Analogy:** a doctor making a diagnosis. Each variant is a different set of test
results you hand them.

| Variant | What the probe is given | Length |
|---|---|---|
| **answer** | Only the hidden state at the answer word | 3,584 |
| **fused_attr** | answer **+** attribution vector, glued side by side | 7,168 |
| **fused_scalars** | answer **+** 7 hand-made numbers (was it mentioned? how often? where?) | 3,591 |
| **fused_both** | answer + attribution vector + the 7 numbers | 7,175 |
| **scalars_only** | Only the 7 hand-made numbers | 7 |
| **fused_decomposed** | answer + attribution vector **split into two parts** (see 4.11) | 10,752 |

So `fused_attr` = *"here is the answer‑word snapshot, and here is a snapshot of
the model thinking about this field — use both."*

## 4.11 fused_decomposed — the fix that gave our best result

Every field's attribution vector can be written as two pieces added together:

```
attribution vector  =  document part  +  field part
```

- the **document part** is the average across all fields in that document — it's
  the same for every field ("this document was hard")
- the **field part** is what's left over, unique to that field ("how the model
  thought about *this* box")

`fused_attr` hands the probe the **sum** of the two, mixed into one block. The
problem: the document part is large and the field part is small, and the training
procedure penalises large inputs — so the small field part effectively gets
thrown away. **We were never actually testing the field part.**

`fused_decomposed` hands over the **two pieces separately**, so the probe can
weigh each on its own merits.

**Analogy.** You give a doctor a single number that adds together the patient's
weight and their fever. Useless — you can't tell which is which. Report weight
and temperature *separately* and both become usable.

Same information, better presentation → **60 % more effect** (+0.036 instead of
+0.023). This is now our proposed method.

## 4.12 The controls — how we try to break our own result

The obvious objection: *"You added 3,584 extra numbers and the score went up. Of
course it did — more inputs always help."*

To rule that out, we built variants that add **exactly the same number of extra
inputs** but destroy the thing we claim matters:

| Control | What it does | What it tests |
|---|---|---|
| **ctrl_shuffled** | Give each field *another field's* attribution vector | Does matching the right reasoning to the right field matter? |
| **ctrl_random** | Replace with random noise of the same size | Does the extra width alone help? |
| **ctrl_docmean** | Give every field the document average | Is the effect field‑specific or document‑wide? |
| **ctrl_tracemean** | Use the *whole* reasoning trace, not just value mentions | Does it matter which words we pool? |
| **ctrl_centered** | Give **only** the field part | Does the field part help by itself? |

**If a control performs as well as our method, our explanation is wrong.** That
is exactly what happened — and finding it out is the point of running them.

## 4.13 Statistics

- **paired test (Wilcoxon):** compares the two methods *document by document*
  rather than comparing two averages. Much more sensitive — like comparing two
  drugs by testing both on each patient rather than on two separate groups.
- **p‑value:** the chance of seeing this result if the method actually did
  nothing. p = 0.00002 means "about 1 in 50,000."
- **Holm correction:** running seven tests means seven chances to get lucky. Holm
  makes each threshold stricter to compensate. **All our headline results survive
  it.**
- **bootstrap confidence interval:** re-run the calculation 2,000 times on random
  re-samples of the documents to see how much the answer wobbles. Ours is
  [+0.019, +0.055] — comfortably above zero.

## 4.14 Selective regeneration terms

- **budget:** the fraction of fields we re-ask. 20 % = re-do 1 field in 5.
- **errors caught (recall):** of all the wrong fields, what fraction did we flag?
- **precision:** of the fields we flagged, what fraction were actually wrong?
- **break‑even repair rate:** how good the re-generation has to be for the effort
  to pay off. Ours is **0.01** — it pays off if re-asking fixes even 1 % of what
  it touches. Extremely permissive.

---

# PART 5 — What we had done before the exam break

State as of Update 07, ~36 days ago:

1. Built the full pipeline for DeepSeek‑R1 on the SOB dataset.
2. Fixed the labeling bug: 95 % → 42 %.
3. Probe worked: **0.85 CV / 0.81 LODO** on **300 documents**, beating log‑prob
   baselines.
4. Document‑level reasoning fusion looked like a **null result** — no help.
5. Field‑localized attribution looked like it **worked**: +0.023, **p = 0.044**.

We told a three‑act story: *the probe works → pooling reasoning at document level
fails → localizing it per field succeeds.*

**Two known weaknesses:** p = 0.044 is borderline and wouldn't survive correction,
and there was no proof the gain wasn't just "more inputs."

---

# PART 6 — What we did in the last few days

## 6.1 Scaled the corpus: 300 → 994 documents

More documents = more statistical power. The dataset had 5,000 available.

We also made extraction **resumable**, so a job killed by the time limit picks up
where it stopped instead of restarting. The 700 new documents cost 4h19m of GPU.

## 6.2 Built the controls (Stage 8)

The five controls in 4.12, plus proper statistics: Holm correction and bootstrap
confidence intervals.

## 6.3 Built the cost‑quality analysis (Stage 9)

This was **completely missing** and it's what the project title promises. Turns
AUROC into "spend this much, catch that many errors."

## 6.4 Ran the geometry diagnostic

Measured how different the per‑field attribution vectors actually are within a
document, to find out whether field‑level information even exists to be found.

## 6.5 Found and fixed the decomposition problem

Realised `fused_attr` was structurally incapable of testing our hypothesis
(4.11), and built `fused_decomposed`.

---

# PART 7 — The results

## 7.1 The probe works and beats free baselines

| Signal | AUROC |
|---|---:|
| mean log‑prob (free) | 0.750 |
| min log‑prob (free) | 0.753 |
| answer‑token probe | 0.799 |
| **probe + reasoning** | **0.827** |

## 7.2 Reasoning helps — much more significantly than before

| | 300 docs (old) | 994 docs (now) |
|---|---|---|
| Best improvement | +0.023 | **+0.036** |
| p‑value | 0.044 (borderline) | **0.000017** |
| Survives correction? | No | **Yes** (0.00012) |

## 7.3 The controls, and what they showed

| Variant | Improvement | Significant? |
|---|---:|---|
| **fused_decomposed** (ours) | **+0.0364** | ✅ *** |
| ctrl_docmean | +0.0263 | ✅ *** |
| ctrl_tracemean | +0.0257 | ✅ *** |
| fused_attr | +0.0227 | ✅ * |
| ctrl_centered | +0.0124 | ✗ |
| ctrl_shuffled | −0.0083 | ✗ |
| ctrl_random | −0.0065 | ✗ |

**Read it like this:**
- shuffled and random do **nothing** → not a "more inputs" artifact ✅
- docmean does **as well as** field‑localized → the effect is document‑wide, not
  per‑field ⚠️
- tracemean does **as well as** value‑mention pooling → which words we pool
  doesn't matter ⚠️

## 7.4 The geometry

- within‑document similarity between fields' attribution vectors: **0.61**
- field part size relative to document part: **0.79**

The vectors genuinely **do** differ per field — so localization isn't impossible
in principle. The per-field differences just aren't very informative about errors.

## 7.5 The practical payoff

At a **20 % regeneration budget**:

| | Errors caught |
|---|---:|
| **our probe** | **49.0 %** |
| answer‑only probe | 46.5 % |
| best free baseline | 40.0 % |
| random | 19.8 % |

Error rate falls **33.2 % → 22.0 %** — a **34 % relative reduction** for a 20 %
spend. Break‑even repair rate **0.01**.

## 7.6 The interpretable finding

| Value's presence in the reasoning | Error rate |
|---|---:|
| Fully mentioned | 28.9 % |
| Never mentioned | 44.3 % |
| **Only partially mentioned** | **83.3 %** |

p = 0.000000000037. Our example fits exactly: "1969" never appears in the trace
(which says 1968), and that field is indeed wrong.

**Caveat to state yourself:** used alone these features reach only 0.717 — *below*
the log‑prob baseline. They're an explanation, not a detector.

---

# PART 8 — Questions he will ask, and your answers

**Q1. "Last time you said field-level attribution works. Now you say it doesn't.
Did you make a mistake?"**

> Two things changed. First, we scaled from 300 to 994 documents, and the extra
> statistical power showed that a control we'd read as null — the document mean —
> actually performs just as well as field-level attribution. Second, we realised
> our original comparison couldn't have detected a field-level effect even if one
> existed, because the two components were mixed into a single input block and
> the small one was regularized away. We fixed that. The overall finding is
> stronger than before; the mechanism is different from what we proposed.

**Q2. "So the earlier result was wrong?"**

> The *effect* was real and is now far more significant. The *explanation* was
> wrong. We only know because we ran controls that most probing work doesn't.

**Q3. "Why is the effect only +0.036? That's small."**

> It is small, and I won't oversell it. Three points: it's highly significant
> (p = 0.00002) and survives correction; it holds at both layers we tested; and in
> practical terms it moves errors-caught from 46.5 % to 49.0 %. The bigger story
> is the gap over the free baseline — +0.074 AUROC, which is what a deployment
> would actually feel.

**Q4. "Is 0.83 AUROC good?"**

> For per-field error detection with no access to the correct answer, yes. The
> model's own confidence gives 0.75, and a perfect oracle is 1.0, so we close
> about a third of that gap.

**Q5. "Why leave-one-document-out instead of normal cross-validation?"**

> Random splits put fields from the same document in both train and test, so the
> probe memorises document quirks. On 300 documents that inflated our score by
> about 0.04. LODO removes it.

**Q6. "Why layer 19?"**

> It was pre-committed from the 300-document pilot. On the full corpus the peak
> is actually layer 21 at 0.806 versus 0.804 for 19 — a difference well inside
> the standard deviation. We deliberately did not re-pick, because choosing the
> best layer on the final data is the post-hoc selection pre-commitment exists to
> prevent.

**Q7. "How do you know it isn't just adding more input features?"**

> That's exactly what the controls test. Random noise of identical size gives
> −0.007. Shuffling the vectors across fields gives −0.008. Both null. The
> content is what matters, not the width.

**Q8. "What does 'document-level effect' actually mean?"**

> The reasoning vector appears to encode something like "this document was hard"
> or "the model was hedging here" — a property of the whole record. Mechanically,
> it acts as a difficulty covariate that lets the answer-token weights calibrate
> themselves. It is not evidence about any individual field.

**Q9. "Why is the error rate 42 %? Isn't that very high?"**

> It's a genuinely hard multi-hop dataset — the model has to combine facts across
> passages. It was originally measuring 95 % because of a scoring bug we fixed.
> And for training a detector, a balanced error rate is ideal; 5 % errors would
> give us almost nothing to learn from.

**Q10. "Could your labels just be wrong?"**

> Possible, which is why we report three matching strictnesses — 46.2 / 42.9 /
> 42.7 %. The conclusions don't depend on which we pick.

**Q11. "What's the practical use?"**

> Selective regeneration. At a 20 % budget we catch 49 % of errors and cut the
> error rate from 33 % to 22 %. The break-even repair rate is 1 %, so it pays for
> itself almost regardless of how good the re-generation is.

**Q12. "How does this relate to Ali's work?"**

> Same framework, different axis. He works on Qwen with ExtractBench; I work on a
> reasoning model with SOB. His future-work slide lists exactly this. My addition
> is the reasoning-trace question, which only exists for a model that thinks out
> loud.

**Q13. "Is this publishable?"**

> I think so, but on rigour rather than effect size. The contributions are: the
> structure-aware labeling correction; a controlled demonstration that reasoning
> traces carry document-level error signal; three well-supported negative results
> about what *doesn't* work; and a demonstration that a 300-document study
> produced two wrong conclusions that 1,000 documents corrected. That last point
> is a methodological warning for a field that routinely uses small corpora.

**Q14. "What's next?"**

> One more experiment: we currently pool the words where a value is *stated*,
> which is usually where the model writes its conclusion. Pooling a window
> *around* the mention targets the reasoning instead. The geometry says
> field-specific structure exists, so that's the remaining lever. It's CPU-only,
> about a day. Then the write-up.

**Q15. "Why DeepSeek-R1 and not a bigger model?"**

> It's the largest reasoning model that fits our GPU allocation with hidden-state
> capture, which is memory-heavy. Scaling the corpus mattered more for the
> statistics than scaling the model.

**Q16. "What would prove you wrong?"**

> If the shuffled or random controls had matched our method, the result would
> have been a dimensionality artifact. They didn't. And if window pooling gives a
> significant field-level gain, that would revise the document-level conclusion —
> which is why we're running it.

---

# PART 9 — How to open the meeting

## 9.1 The first two minutes — say roughly this

> "Since we last met I've been on exams, so this covers about five weeks of work
> compressed into the last week and a half of compute.
>
> Three headlines. First, I scaled the corpus from 300 to nearly 1,000
> documents. Second, the main result got much stronger — reasoning traces improve
> per-field error detection by 0.036 AUROC, p equals 1.7 times 10 to the minus 5,
> and it survives multiple-comparison correction. Previously that was 0.044,
> borderline.
>
> Third, and I want to be upfront about this: the controls I built changed our
> explanation. We thought the gain came from localizing the reasoning to each
> field. It doesn't — it's a document-level effect. I'll walk through how we
> established that, because it's the most interesting part.
>
> I also completed the cost-quality analysis that was missing. At a 20 %
> regeneration budget the signal catches 49 % of errors versus 40 % for the free
> baseline, cutting the field error rate from 33 % to 22 %."

## 9.2 Then walk through, in this order

1. **The scale-up** — 300 → 994, and why it mattered
2. **The main result** — the table in 7.1 and 7.2
3. **The controls** — 7.3, and the two reversals
4. **Why the earlier comparison couldn't have worked** — 4.11, the mixed-block
   problem. This is your strongest technical moment: it shows you found a flaw
   in your own method and fixed it
5. **The geometry** — 7.4, which rules out the trivial explanation
6. **The practical payoff** — 7.5
7. **The interpretable finding** — 7.6, with the caveat
8. **Next steps** — window pooling, then write-up

## 9.3 On the figures

> "I have the figures generated but I want to refine them before showing them —
> I'll bring them next time."

## 9.4 Three things to be honest about, unprompted

Volunteering weaknesses builds far more credibility than being caught on them:

1. The effect is modest (+0.036).
2. The mention statistics are interpretable but *not* a competitive detector
   (0.717, below the 0.753 baseline).
3. Single model, single dataset.

## 9.5 If you get stuck on a question

> "I don't have that number in my head — let me check it and send it to you
> today."

Never guess a number. He will remember it, and every real number you have is
already verified.

---

# PART 10 — The five numbers to memorise

If you remember nothing else:

1. **0.827** — our AUROC (vs **0.753** for the model's own confidence)
2. **+0.036, p = 0.000017** — the reasoning improvement, survives correction
3. **974 documents, 5,071 fields, 33 % errors** — the scale
4. **49 % of errors caught at a 20 % budget**, error rate **33 % → 22 %**
5. **83 %** — error rate when a value is only *partially* mentioned in the
   reasoning
