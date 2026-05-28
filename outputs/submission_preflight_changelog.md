# Submission preflight changelog

Author: Tina Koziol. Branch: `claude/submission-prep`.
Generated 2026-05-26.

This changelog records the submission-prep pass on the manuscript
`paper/blocked_ownership_broken_mainstreams.tex`. The pass addresses
reviewer-risk points identified in the submission checklist while
preserving the paper's argument and refraining from inventing
unsupported claims.

## 0. Restoration of lost PR #21 content (`b81d70b`)

The session began by surfacing and fixing a previously-undetected git
bug: commit `c3c468d` ("Paper: keep local abstract rewrite on top of
PR #21 integration", 2026-05-26) silently reverted nearly all PR #21
manuscript content, despite the commit message claiming preservation.
Root cause: a `git checkout --theirs` during stash-pop conflict
resolution took the stashed working tree (local abstract on a
pre-PR21 base) rather than merging onto the PR-merged version.

Restored from `a8269fe`:

- Section 6.4 retitle "Reversal and censoring diagnostics by leakage
  profile" with `sec:leakage` label and two-panel Table 9 (Panel A
  cross-leakage half-life; Panel B 50-seed central-leakage deep
  diagnostic).
- Section 9.1 K-R channel-collapse reframe (+0.280 → +0.007 cleavage,
  +0.209 → +0.009 AfD; Scenario E peak-and-decay does not survive).
- Appendix "Provenance of Section 9 K-R numbers" (replaces "Known
  reproducibility gap").
- Section 6.2, Section 7, conclusion, and Appendix A.3 prose updates.
- Appendix script list entries for `scripts/run_kr_robustness.py` and
  `scripts/scenario_e_robustness_n50.py`.

Preserved from current master (added by user after `c3c468d`):

- Multi-paragraph abstract rewrite.
- Section 7 expansion with subsections "Why an agent-based model?"
  and "Structural necessity of the state space" (the latter now
  retitled, see below).

## 1. Koszegi-Rabin reference-rule diagnostic

**Note on initial spec:** the submission checklist asked for a K-R
script that reproduces the *previously-reported* K-R numbers
(cleavage ≈ +0.18, AfD ≈ +0.16, "Scenario E peak-and-decay
survives but is attenuated"). Those target numbers correspond to an
uncommitted K-R variant that no committed code reproduces. After
clarification, the user directed: treat the old target numbers as
superseded; keep the current reproducible K-R path; do not invent or
tune numbers.

**Current state:** `scripts/run_kr_robustness.py` evaluates the
SMM-optimum parameter vector under the Koszegi-Rabin
rational-expectations anchor (AR(1) on own realised outcome at the
same rho as AY; the cleanest reduced-form contrast). The script
writes `outputs/kr_robustness.json` with an AY-vs-K-R side-by-side
moment table and an honesty-check verdict. Stable output:

- K-R within-region renter-owner cleavage = +0.007 (vs +0.279 AY)
- K-R aggregate AfD share = +0.009 (vs +0.209 AY)
- Scenario E peak-and-decay does not survive under K-R

Section 9.1 has been further refined to frame this as a
**specification stress test that supports the AY modelling choice**
(rather than as a robustness exercise). The collapse supports the
modelling choice that the housing aspiration in the baseline is an
Akerlof-Yellen-style slow-moving normative anchor rather than a
rapidly adapting own-outcome reference point.

**Files added/modified for this item:**
- `scripts/run_kr_robustness.py` — exists (08a2a98), unchanged this pass
- `outputs/kr_robustness.json` — regenerated
- `src/abmhp/config.py` — `VotingConfig.reference_rule` flag (exists)
- `src/abmhp/voting.py` — K-R branch (exists)
- `tests/test_reference_rule.py` — regression test (exists)
- `paper/blocked_ownership_broken_mainstreams.tex` Section 9.1 — refined framing
- `README.md` — added run_kr_robustness.py + scenario_e_robustness_n50.py to the reproduction list

## 2. Appendix provenance

Appendix paragraph rewritten in PR #21, restored in this pass. The
appendix now says "Provenance of Section 9 K-R numbers" (replacing
the old "Known reproducibility gap") and points referees to the
script and JSON output, with transparent acknowledgment that earlier
drafts reported numbers the committed script does not produce.

## 3. Section 7 softening

Categorical phrases replaced with hedged scope-specific wording.

- Section heading changed from "Methodological necessity of
  agent-based heterogeneity" to "State-space requirements of the
  present research design."
- Subsection 2 heading changed from "Structural necessity of the
  state space" to "State-space requirements for the present
  estimands."
- "The research question requires regional, tenure, and channel
  state" → "The estimands the paper relies on depend on regional,
  tenure, and channel state being present in the model
  specification."
- Three "cannot be identified by … collapses to zero by construction"
  sentences rephrased as "within the present research design …
  not recoverable under specifications that aggregate away … for
  the moments studied here it would be mechanically zero …
  the relevant contrast would be unavailable."
- "These are structural-necessity claims" → "These are scope claims
  about the state space the present paper's estimands depend on."
- "The answer is no" → "Within the present research design and for
  the moments studied here, the wedge is not recoverable without
  that state."
- The corresponding paragraph in the introduction (line 83) and the
  roadmap line in the introduction's last paragraph also updated to
  match.

The core structural claim is preserved; the framing avoids categorical
universalism that could draw reviewer pushback.

## 4. Equal-cost (E-light / C-plus) visibility

The abstract's equal-cost paragraph now explicitly names Scenarios
E-light and C-plus and states the two-sided necessity (channel
coverage AND non-trivial fiscal intensity) rather than just one side.
A new paragraph in the introduction (before the policy-implication
paragraph) describes the equal-cost decomposition as a direct response
to the "Scenario E works because it is fiscally larger" objection.

The full Section 6.2 development of E-light / C-plus was unchanged
this pass (it was already developed in depth there).

## 5. Empirical target provenance

Tables 1–3 were already mostly transparent (explicit "ranking-check
proxy" label, "$\pm 0.05$" reference intervals, sign-only flags).
Two targeted swaps:

- Table 3 caption: "Approximate survey-based comparators are
  flagged" → "Survey-implied central values and sign-only stress-test
  targets are flagged in the source column."
- Table 3 rows for DE/UK renter-owner gap: "$\pm 0.050$ (approx.)"
  → "survey-implied $\pm 0.050$."

Appendix A.4 left unchanged: it already uses "ranking-check target,"
"qualitative (sign and shape) rather than a point estimate," and
"sign-only check."

## 6. GitHub URL handling + anonymized version

A compile-time `\ifanonymized` flag was added at the top of the
`.tex`. The same source file produces two PDFs:

- **Named (default)**: `paper/blocked_ownership_broken_mainstreams.pdf`
  Shows author name, correspondence email, GitHub URL (3 sites), and
  full Acknowledgments paragraph.
- **Anonymized** (compile-time): `paper/blocked_ownership_broken_mainstreams_anon.pdf`
  Title page reads "Author identification suppressed for double-blind
  review"; PDF metadata `pdfauthor` is empty; the three GitHub URL
  sites read "through the anonymized review portal"; Acknowledgments
  reads "Acknowledgments omitted for double-blind review."

Build helper: `paper/build_paper.sh` builds either or both (`bash
paper/build_paper.sh both`). README updated with the build commands.

Verification (via `pdftotext`):

- Named PDF title page contains "Tina Koziol" + correspondence email.
- Anonymized PDF title page contains the suppression text; no
  occurrences of "Koziol", "tinak", "t1nak", "Frankfurt School", or
  "Cape Town" anywhere in the rendered text.
- "Frankfurt am Main" in Section 3 (describing regional structure) is
  not an author identifier and is retained in both versions.

## 7. AI disclosure

Both the manuscript declaration and the README disclosure rewritten
to use generic model naming and a fuller scope statement:

- "Anthropic Claude Opus 4.7" → "Anthropic Claude (via Claude Code)".
- Scope expanded to name both manuscript-side use (drafting,
  structural editing, prose refinement) and code-side use (simulation
  engine, SMM estimation, robustness scripts, replication
  infrastructure).
- "reviewed and edited" → "reviewed, edited, and verified."
- Responsibility statement unchanged ("takes full responsibility").
- No overclaiming of AI capabilities.

## Files changed

```
paper/blocked_ownership_broken_mainstreams.tex     — multiple edits
paper/blocked_ownership_broken_mainstreams.pdf     — rebuilt (named)
paper/blocked_ownership_broken_mainstreams_anon.pdf — new (anonymized)
paper/build_paper.sh                                — new (helper)
README.md                                           — repro list + AI disclosure
outputs/kr_robustness.json                          — regenerated
outputs/submission_preflight_changelog.md           — this file
```

## Tests / checks run

- `bash paper/build_paper.sh both` — both PDFs build cleanly,
  no LaTeX errors or undefined references. Named: 58 pages, 1.58 MB.
  Anonymized: 60 pages, 1.58 MB (extra pages from the conditional
  layout). Pre-existing typography warnings (overfull/underfull
  hboxes) in appendix sections unchanged.
- `python scripts/run_kr_robustness.py` — runs in 13.6s, produces
  the documented K-R numbers (cleavage +0.007, AfD +0.009),
  reproducibility check passes against the pre-existing AY anchors,
  honesty-check verdict is DISCREPANCY against any hypothetical
  paper-claimed band — the script does not (and is not expected to)
  reproduce the superseded +0.18 / +0.16 numbers.
- `pdftotext` audit of both PDFs confirmed no author-identifying
  content leaks into the anonymized PDF.
- The repo's unit-test environment is in a pre-existing broken state
  (`pytest` under the user's Python 3.13 can't import `numpy`), so
  `pytest tests/` was not run. The new K-R contract test
  `tests/test_reference_rule.py` was previously verified manually via
  a pickle-and-diff bit-identity check at the simulator level (PR
  #21, commit `08a2a98`).
- `scripts/counterfactual_material_security.py` and
  `scripts/run_smm.py` were not re-run in this pass (no code changes
  in this branch that affect their output; K-R diagnostic is the
  only new compute).

## Remaining limitations and unresolved issues

1. **K-R numbers do not match the original draft.** The committed
   K-R operationalisation (AR(1) on own outcome) collapses the
   dissatisfaction channel rather than reproducing the previously
   reported +0.18 / +0.16 split. Section 9.1 reframes this as a
   specification stress test supporting the AY anchor. A reader who
   prefers the original framing would need to identify the K-R variant
   that produced the older numbers; no such variant is currently
   reproducible from the committed code.
2. **Repository availability statement.** The README still says
   "available on request from the author for replication and review
   purposes." For a double-blind submission this would need to be
   replaced with "available through the anonymized review portal" or
   the journal's prescribed wording at submission time.
3. **Test environment.** Per-PR CI would need the repo's `pyproject.toml`
   dev dependencies properly installed (`uv sync` or `pip install -e .[dev]`)
   to run `tests/test_reference_rule.py` and the rest of the test suite.
   This is a pre-existing repo-state issue, not introduced by this pass.
4. **`scripts/compare_abm_hank.py`** continues to be flagged as a
   development artefact "not invoked by the main paper" (appendix
   script list). If the HANK comparator is in scope for the
   submission, this wording should be revisited.
