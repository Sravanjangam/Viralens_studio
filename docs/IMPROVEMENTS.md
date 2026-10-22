# Technical Improvement Notes

Assessment of the developer setup, architecture, and scoring behaviour, with
recommendations ordered by evidence. This document records findings; it does not
claim that the issues are fixed.

## How this assessment was done

**Method**

- Compared the README and configuration files with the actual project structure
  and imports.
- Traced the `/full/full/` request path through routers, engines, and scoring.
- Inspected the scoring implementation and the existing tests.
- Ran setup commands, the benchmark, and the scoring scripts. Only reproducible
  results are treated as evidence.

**Engineering decisions**

- **Developer onboarding first.** Accurate install, configuration, startup, and
  testing instructions are required before contributors can work on anything else.
- **Document before refactoring.** Verified structural concerns are recorded
  before code changes are proposed.
- **Measure before optimising.** Performance work is recommended only where a
  measurement shows a meaningful bottleneck, and output quality is checked
  alongside speed.
- **Understand scoring before changing it.** Signal contributions are inspected
  before any weight or formula change.
- **Test behaviour and failure cases.** Tests should protect user-facing
  behaviour, not implementation details.

**Decision criteria**

Each recommendation states:

1. the observed problem,
2. the supporting evidence,
3. the expected benefit,
4. the potential trade-off,
5. the next action.

**Classification**

- **Prioritize** — verified issue with meaningful impact.
- **Investigate** — plausible concern that needs more evidence.
- **Defer** — benefit does not justify the cost or risk yet.

Effort: **S** ≤ 1 day, **M** ≤ 1 week, **L** > 1 week.

## What the review found

These findings come from the code and from reproducible checks, not from user
research:

1. **Missing signals score 3.6, perfect signals score 96.4.**
   `ViralityScorer` inverts clutter (`1 - clutter`) and defaults absent keys to
   `0`. An empty input still earns `0.2` geometry; `clutter = 1` caps geometry at
   `0.8`. Both are documented by the tests in PR #2.
2. **Brightness and contrast dominate the "visual" score.**
   `aesthetic = 0.4·brightness + 0.6·contrast` is not an independent signal, and
   `color_light` uses the same two values again. Expanding the weights,
   brightness + contrast account for ~76% of the visual subscore — and the visual
   subscore is 60% of the final score — while rule of thirds contributes ~12%.
3. **The trend bank contains one image.**
   `cv_engine/viral_bank/meta.json` lists a single sample (`bus.jpg`, label
   `example`). Trend similarity is the maximum cosine to that one vector, yet it
   carries 15% of the final score.
4. **"Aesthetic" in the API is a lighting heuristic.**
   `full_router.py` sets `aesthetic_score = 0.4·brightness + 0.6·contrast`.
   `cv_engine/aesthetic.py` (CLIP-based) is never called by the API.
5. **The impact claim is unmeasured.**
   "Improves content optimization efficiency by ~40–55%" has no experiment
   behind it. Replace it with a measured metric or label it as a hypothesis.
6. **The API contract and the response disagree.**
   `backend/models/full_model.py` describes `objects` and `virality_score`, while
   `/full/full/` returns `dominant_colors`, `virality`, and no `response_model`.

## P0 — Before wider use

| ID | Improvement | Problem and evidence | Benefit | Trade-off | Class / effort |
| --- | --- | --- | --- | --- | --- |
| P0-1 | Downscale to 1024 px before all CV work | 4K CV is **8.96 s**, of which 8.70 s is KMeans; the same pipeline at 1024 px is **0.34 s** (measured) | ~26× faster CV, with no meaningful loss for these heuristics | Discards fine detail; validate on a golden image set | Prioritize · S |
| P0-2 | One shared model registry | YOLO, CLIP, and RoBERTa are each loaded twice at startup (`full_router.py:21-24`) | Faster cold start, lower memory, viable multi-worker setup | New lifecycle wiring; tests must construct engines explicitly | Prioritize · S |
| P0-3 | Move CPU inference off the event loop | `async def` handlers call blocking code; one request blocks all others | Concurrent requests stop serialising behind one analysis | Threadpool overhead; CPU capacity is still the ceiling | Prioritize · S |
| P0-4 | Stop leaking temp files | `tmp_*` / `temp_*` files are written to the working directory and never removed | No unbounded disk growth | In-memory buffers raise peak RAM; pair with an upload cap | Prioritize · S |
| P0-5 | Load the chat engine lazily | A missing `GROQ_API_KEY` currently prevents the whole API from booting | Analysis works without chat credentials | Chat failures move to request time; return a clear 503 | Prioritize · S |
| P0-6 | Validate uploads | No size, MIME, or pixel limits; client filenames feed filesystem paths | Prevents memory/CPU exhaustion and unsafe paths | Limits can reject legitimate large images; make them configurable | Prioritize · M |
| P0-7 | Pin dependencies | `requirements.txt` uses floors only; library upgrades can change scores | Reproducible installs and scores | Slower security updates; use a lockfile plus scheduled upgrades | Prioritize · S |
| P0-8 | Align schemas and declare `response_model` | `/full/full/` has no response model; `full_model.py` describes a different payload | Accurate OpenAPI and contract tests | Renaming fields can break the current UI; migrate both together | Prioritize · M |

## P1 — Before a public launch

| ID | Improvement | Problem and evidence | Benefit | Trade-off | Class / effort |
| --- | --- | --- | --- | --- | --- |
| P1-1 | Content-hash result cache | The UI encourages repeated runs on the same image; recomputation is pure waste | Repeated runs near-instant; less CPU | Stale results if the key omits model/weights version; invalidation cost | Prioritize · M |
| P1-2 | Per-stage metrics and tracing | There are no request IDs, timings, or metrics; p95 cannot be managed | Bottlenecks and regressions become visible | Small overhead and operational work; avoid logging captions/images | Prioritize · M |
| P1-3 | ONNX / int8 inference | CPU inference cost is high, but export accuracy and speed are not yet measured | Lower CPU latency and cost | Accuracy drift and export maintenance; benchmark on target hardware first | Investigate · M |
| P1-4 | Real test suite + CI | Current tests are scripts with no assertions; regressions are invisible | Regressions caught before merge | CI time and maintenance; keep network/heavy tests separate | Prioritize · M |
| P1-5 | Restrict CORS, add rate limits | Wildcard CORS with credentials is invalid for browser auth and invites abuse | Safer public deployment and cost control | Wrong limits block real users; make them configurable and observable | Prioritize · S |
| P1-6 | Warmup and readiness probe | The first request after deploy is the slowest; "healthy" does not mean "models loaded" | First request is no longer the worst; readiness is truthful | Startup takes longer and holds memory earlier | Prioritize · S |
| P1-7 | Stream or stage results | A single long spinner gives no progress or partial recovery | Perceived latency drops; partial failures are recoverable | More complex client/API contract | Investigate · M |
| P1-8 | Calibrate the score against outcomes | Weights are hand-set; a 0–100 score implies precision the model lacks | The score becomes defensible and comparable | Requires outcome data and a stable evaluation set; scores will shift | Prioritize · L |

## P2 — When usage justifies it

| ID | Improvement | Problem and evidence | Benefit | Trade-off | Class / effort |
| --- | --- | --- | --- | --- | --- |
| P2-1 | Inference worker + queue | API and models scale together today; every worker would load every model | Independent scaling and batching | Operational complexity; only pays off under real load | Defer · L |
| P2-2 | Vector index for the trend bank | A single reference image makes 15% of the score noise | Makes a larger trend corpus queryable | Infrastructure adds nothing until curated content exists | Defer · L |
| P2-3 | Platform / niche scoring profiles | Instagram, TikTok, and Shorts reward different visuals, but no per-platform data is available in the repo | More relevant scoring per segment | Sparse data and overfitting risk; keep `general` as default | Investigate · L |
| P2-4 | History, batch, A/B compare | Creators iterate across variants, but the app handles one image at a time | Fits real iteration workflows | Storage, privacy, and UI scope | Defer · M |
| P2-5 | Container + deployment pipeline | The README implied HF Spaces support that does not exist | Reproducible deployment | Maintenance; choose the target platform first | Defer · M |
| P2-6 | Auth, quotas, billing | Paid endpoints are open, but positioning is unvalidated | Enables paid usage and abuse control | Auth/ops burden before demand is proven | Defer · L |

## Product improvements

### 1. Make the score honest and calibrated

**Problem:** weights are hand-set (`0.60 / 0.25 / 0.15`, then `0.40 / 0.30 /
0.30`), the empty-input floor is 3.6, the ceiling is 96.4, and no confidence is
shown. The word "virality" implies predictive validity the model does not have.

**Suggestions:**

- Rename the output to "Virality Potential (heuristic)" until it is validated.
- Return a confidence band and a percentile within a niche, not just a point
  score: `72 ± 8, top 30% of food posts`.
- Show the top two drivers and the top limiter ("clutter and low contrast hold
  this back"), with the contribution of each signal.
- Publish the scoring formula and version it (`scoring_version` in responses).
- Collect post outcomes (opt-in) and track rank correlation between predicted
  percentile and observed engagement. Re-fit weights only against that data.
- Report calibration on a holdout set; never tune and evaluate on the same posts.

**Trade-off:** Requires outcome data and a fixed evaluation set; scores
will shift as weights change, so version the formula.

**Success metric:** monotonic improvement in rank correlation and calibration
error across releases, tracked on a fixed evaluation set.

### 2. Turn the trend bank into a real asset

**Problem:** one reference image means "trend similarity" is similarity to a bus
photo. With 15% of the final score, this actively misleads users.

**Suggestions:**

- Build a curated corpus per platform and niche (start with 1–5k licensed images
  per niche, not millions).
- Store provenance (source, date, platform, engagement bucket) and refresh on a
  schedule; trend relevance decays.
- Index embeddings in FAISS/pgvector; return top-k neighbours as evidence
  ("similar to these 5 posts"), not just a scalar.
- Version the bank and include the version in cached results.
- When the bank is too small, return a neutral score plus an explicit
  `low_confidence` flag instead of a confident-looking number.

**Trade-off:** Curation, licensing, and refresh work; until the bank is
large enough, trend must be reported as low-confidence.

**Success metric:** bank size/freshness per niche, and whether trend score changes
actually correlate with engagement.

### 3. Separate lighting from aesthetics

**Problem:** the "aesthetic" signal is brightness/contrast, duplicating
`color_light`. Brightness and contrast end up influencing most of the visual
score through three paths.

**Suggestions:**

- Either wire the existing CLIP aesthetic scorer into `/full` (cost: one more
  model) or remove the aesthetic term and rename the field to
  `lighting_quality`.
- Document the effective weight of every raw signal, not just the top-level ones.
- If CLIP aesthetics stay, validate them against human ratings on a small set.

**Trade-off:** Wiring CLIP adds a model and latency; removing the term
changes existing scores.

**Success metric:** no signal appears twice under different names; each has a
documented, non-overlapping contribution.

### 4. Fix caption edge-case behaviour

**Problem:** an empty caption returns `sentiment_score = 0.5` but
`overall_caption_score = 0.0`. A neutral sentiment is combined with "no hook, no
CTA, no length, no emoji", so "no caption" scores as strongly negative without
saying so.

**Suggestions:**

- Add an explicit `has_caption` flag and decide the product behaviour: neutral
  score, or a visible "caption missing" penalty.
- Explain the length heuristic (8–30 words) in the UI.
- Show the detected hook/CTA phrases so users understand the binary scores.
- Distil or ONNX-export the sentiment model; it is the heaviest NLP dependency.

**Trade-off:** Changing the empty-caption rule changes existing scores; ship it
behind a version bump.

### 5. Add platform and niche context

A single global weighting cannot serve a beauty flat-lay, a talking-head clip
thumbnail, and a meme. Add optional `platform` and `niche` inputs, start with
`general`, and only introduce specific profiles when there is data to support
them. Do not silently mix them into one score.

**Trade-off:** More inputs and sparse data per segment; keep `general` as the
default until a segment has enough samples.

### 6. Make recommendations actionable

The chat endpoint can explain a score, but the main result should stand alone:

- top 3 changes with expected score deltas (counterfactuals);
- caption alternatives aligned to the detected weakness;
- a one-click "apply and re-score" loop;
- export as a shareable report for clients.

**Trade-off:** Counterfactuals require extra scoring passes; label them as
estimates.

### 7. Close the feedback loop

Add an opt-in "I posted this" flow with the observed metrics (reach, saves,
watch time, follows). Use it for calibration and for aggregate benchmarks ("your
composition is above 80% of similar posts"). Be explicit about retention and
privacy; do not log captions or images by default.

**Trade-off:** Privacy and retention obligations; opt in and aggregate only.

### 8. Workflow and mobile

Creators iterate on phones. Prioritise a responsive web UI, batch upload, A/B
comparison of variants, history, and a compact report. A native app is not needed
until the web loop is validated.

**Trade-off:** Scope creep before the single-image loop is fast and trustworthy.

### 9. Validate the positioning before building billing

Possible directions: creator subscription, agency/API tier, white-label reports.
Each implies different features (history and branding vs. throughput and SLAs).
Interview 10–15 creators or agencies before investing in auth, quotas, or billing.

**Trade-off:** Billing adds auth/ops cost; validate demand first.

## Quality and testing

- Keep PR #2's scoring tests; add tests for caption edge cases, geometry bounds,
  and the API contract.
- Add a small golden-image set (10–20 licensed images) with expected score ranges,
  not exact floats. Store hashes and thresholds, not just snapshots.
- Add API contract tests that fail if `/full/full/` stops matching its declared
  response model.
- Add a latency smoke test in CI with a generous budget (for example, 1080p CV
  under 1 s on the CI runner) so a resolution or KMeans change cannot regress
  silently.
- Add load tests only after the P0/P1 work; before that they measure the wrong thing.
- Run unit tests on Linux and macOS, Python 3.10 and current stable.
- Verify the chat path end-to-end: `ChatEngine` calls the OpenAI **Responses**
  API (`client.responses.create`) against Groq's OpenAI-compatible endpoint,
  whose documented surface is centred on Chat Completions. Fall back to
  `chat.completions.create` if the call fails.
- Clean up `test_gemini.py`: it uses the legacy `gemini-pro` model id, is
  unrelated to the app, and its `google-genai` dependency is not in
  `requirements.txt`.

## Security and cost

- **Uploads:** cap bytes and pixels, allowlist MIME types, verify the decoded
  image, never derive filesystem paths from the client filename.
- **CORS:** replace `allow_origins=["*"]` with the real frontend origin;
  wildcard + credentials is not a valid credentialed-browser configuration.
- **Abuse:** rate-limit `/chatbot/chat/` (paid Groq tokens) and `/image/improve/`
  (external generation) before exposing them publicly.
- **Secrets:** keep `.env` untracked, rotate the Groq key if it was ever shared,
  and use platform secret stores in deployment.
- **Privacy:** do not retain uploads or captions by default; if you do, document
  retention and deletion.
- **Dependencies:** pin and scan (Dependabot/Renovate). Model and library upgrades
  can change scores, so re-run the evaluation set on upgrade.
- **Cost:** the current architecture loads every model for every deployment.
  Track cost per request and cache aggressively before adding GPU capacity.

## Metrics to instrument first

| Metric | Why |
| --- | --- |
| p50/p95 latency per stage | Find the real bottleneck instead of guessing |
| Cache hit rate | Prove the cache earns its complexity |
| Error rate by endpoint | Detect broken model/API paths |
| Score distribution | Catch weight changes and degenerate inputs |
| Calibration (rank correlation) | The only evidence the score means anything |
| Trend bank size and age | Prevent the one-image problem from returning |
| Cost per request | Keep chat and image generation sustainable |

## Scope and limitations

This document records the findings of the repository assessment; setup guidance
lives in the README. It does not claim to fix the identified application issues.
Findings, benchmarks, and verification results are included only where they are
supported by the actual code or by a reproducible check.

The benchmark covers the computer-vision stages only, on one machine, with
synthetic images. Model inference, network transfer, and real-photo variation are
not measured here; those numbers are planning targets, not results.

## Open questions for the owner

1. Which platform and niche should the first calibrated version target?
2. Is there any historical post/engagement data that can be used for calibration?
3. Where will this run: Hugging Face Spaces, a VM, or serverless?
4. Are uploads allowed to be stored for evaluation, or must they be ephemeral?
5. Is the goal a creator-facing tool, an agency API, or a portfolio demo?
