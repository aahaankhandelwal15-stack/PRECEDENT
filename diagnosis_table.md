# Phase 10 Diagnosis — Failing Tests & Trace-Helper Errors

## 6 Failing Tests

| Test | Error | Actual behavior | Expected behavior | Root cause | Likely origin | Independent/downstream |
|------|-------|----------------|------------------|------------|---------------|-----------------------|
| `test_fm3_conflict_no_override` | AssertionError | `outcome=refused_low_confidence_generation` | `outcome in ("refused_conflicting_evidence", "refused_no_evidence")` | `compute_confidence_from_inputs()` produces below-threshold confidence (≈0.08) when `claim_verifications` is empty; `_determine_refusal_category()` returns `low_confidence` when `final_confidence < DEFAULT_CONFIDENCE_THRESHOLD` (0.5). Conflict between policy_a (₹50K, dept: all) and policy_b (₹75K, dept: management) with no supersedes metadata should produce `refused_conflicting_evidence`, but confidence penalty drops score below refusal threshold. | `confidence_scorer.py`: `×0.5` penalty at line 140 for empty `claim_verifications`; `_determine_refusal_category()` at `retrieval_phase2.py:394` uses `final_confidence` to choose refusal category. | Downstream of confidence scoring; conflict detection is correct but overridden by low-confidence refusal. |
| `test_fm5_prompt_injection` | AttributeError | `resp.status_status` raises AttributeError | `resp.status_code == 200` | Test code typo: uses `status_status` instead of `status_code` at `test_phase10.py:379`. The `FilteredRetrievalResponse` Pydantic model has `status_code`, not `status_status`. | Test code at `test_phase10.py:379` — pure test bug, no application involvement. | Independent (test code error, not app logic). |
| `test_fm7a_empty_corpus` | AssertionError | `outcome=refused_low_confidence_generation` | `outcome in ("refused_no_evidence", "refused_low_confidence")` | Empty corpus short-circuit at `retrieval_phase2.py:237-244` calls `_determine_refusal_category(found=False, empty_corpus=True, ...)` which returns `refusal_category="low_confidence"` because `final_confidence=0.0 < DEFAULT_CONFIDENCE_THRESHOLD`. Test expects `refused_no_evidence` or `refused_low_confidence` (without `_generation` suffix), but `_determine_refusal_category` returns `low_confidence` which maps to `refused_low_confidence_generation`. | `_determine_refusal_category()` in `retrieval_phase2.py:394` — refusal category determination depends on `final_confidence` value, which is 0.0 for empty corpus. | Independent (empty-corpus short-circuit path; not downstream of other failures). |
| `test_fm7c_irrelevant_evidence_only` | AssertionError | `outcome=refused_low_confidence_generation` | `outcome in ("refused_no_evidence", "refused_inapplicable_evidence")` | Query "parking lot policies" retrieves only irrelevant chunks (parking, authority_level: departmental). `compute_confidence_from_inputs()` applies `×0.5` penalty for empty `claim_verifications` (line 140) and additional penalties from authority_flags; final confidence falls below 0.5 threshold, producing `refused_low_confidence_generation` instead of `refused_inapplicable_evidence`. | `confidence_scorer.py`: empty `claim_verifications` ×0.5 penalty at line 140; `authority_flags` with `authority_level="departmental"` may trigger further penalties. `_determine_refusal_category()` returns `low_confidence` when confidence < 0.5. | Downstream of confidence scoring; the irrelevant-evidence classification may be correct but overridden by low-confidence refusal. |
| `test_fm8_simple_direct_fact` | AssertionError | `outcome=refused_low_confidence_generation` | `outcome == "answered"` | Query "What is the reimbursement cap?" retrieves 5 applicable chunks (current policy) with `evidence_sufficiency=1.0`. However `claim_verifications` is empty (no generation has occurred yet per FR-5.3 pipeline ordering). `compute_confidence_from_inputs()` applies `×0.5` penalty at line 140 for empty `claim_verifications`, dropping confidence from ~0.8 to ~0.4 below the 0.5 threshold. `_determine_refusal_category()` then returns `low_confidence`, mapping to `refused_low_confidence_generation`. Sufficient evidence exists but confidence penalty causes false refusal. | `compute_confidence_from_inputs()` at `confidence_scorer.py:138-140`: `×0.5` penalty when `claim_verifications` is empty, with no downward adjustment for strong other inputs. `_determine_refusal_category()` at `retrieval_phase2.py:394`. | Downstream of confidence scoring; the evidence is sufficient but the confidence pipeline penalizes missing claim verifications. |
| `test_fm8_multi_sentence_rule` | AssertionError | `outcome=refused_low_confidence_generation` | `outcome == "answered"` | Query "What are the eligibility requirements and approval process?" retrieves 2 applicable chunks (eligibility + approval sentences). `evidence_sufficiency = min(2/5, 1.0) = 0.4`. `claim_verifications` empty → `×0.5` penalty at line 140. Final confidence ≈ 0.4 × 0.5 = 0.2, well below 0.5 threshold. `_determine_refusal_category()` returns `low_confidence` → `refused_low_confidence_generation`. Sufficient multi-sentence evidence exists but confidence penalty causes false refusal. | `compute_confidence_from_inputs()`: `evidence_sufficiency = min(2/5, 1.0) = 0.4` + `×0.5` for empty `claim_verifications` → final ≈0.2. `_determine_refusal_category()`. | Downstream of confidence scoring; same root as FM8 simple fact but with lower evidence_sufficiency due to fewer chunks. |

## 6 Trace-Helper Errors (pytest fixture issues)

All 6 `test_trace_assertions_*` errors are identical in nature: the functions are regular Python functions (not pytest fixtures) with parameter names that pytest attempts to resolve as fixture names, causing "fixture not found" errors.

| Error Helper | Error | Root cause |
|-------------|-------|------------|
| `test_trace_assertions_outcome` | `fixture 'outcome' not found` | Function at `test_phase10.py:736` takes `outcome: str, expected: str` as parameters but has no `@pytest.fixture` decorator and no `yield` statement. Pytest treats `outcome` as an expected fixture name. |
| `test_trace_assertions_refusal_category` | `fixture 'refusal_category' not found` | Function at `test_phase10.py:741` takes `refusal_category: str, expected: str` as parameters but is not a pytest fixture. |
| `test_trace_assertions_claims_has_citations` | `fixture 'claims' not found` | Function at `test_phase10.py:746` takes `claims: list, expected_min: int = 1` as parameters but is not a pytest fixture. |
| `test_trace_assertions_failures_records_failures` | `fixture 'failures' not found` | Function at `test_phase10.py:753` takes `failures: list` as parameter but is not a pytest fixture. |
| `test_trace_assertions_budget_present` | `fixture 'execution_budget' not found` | Function at `test_phase10.py:763` takes `execution_budget: dict` as parameter but is not a pytest fixture. |
| `test_trace_assertions_no_secrets` | `fixture 'trace_text' not found` | Function at `test_phase10.py:772` takes `trace_text: str` as parameter but is not a pytest fixture. |

All 6 are regular functions intended as audit/verification helpers (called directly in test bodies or meant to be injected as fixtures). They lack the `@pytest.fixture` decorator or `yield` pattern, so pytest cannot resolve them as fixtures.

## Root-Cause Grouping Summary

**Same-root group 1 — Empty `claim_verifications` penalty** (4 tests):
- `test_fm3_conflict_no_override` — conflict detected but confidence penalty suppresses refusal
- `test_fm8_simple_direct_fact` — sufficient evidence, `×0.5` penalty for empty claim_verifications
- `test_fm8_multi_sentence_rule` — same pattern, lower evidence_sufficiency compounds the issue
- `test_fm7c_irrelevant_evidence_only` — irrelevant evidence processed but confidence too low

**Same-root group 2 — Refusal category determination** (2 tests):
- `test_fm7a_empty_corpus` — empty corpus short-circuit returns `low_confidence` category; test expects `no_evidence` or bare `low_confidence` (without `_generation` suffix)

**Independent**:
- `test_fm5_prompt_injection` — pure test code typo (`status_status` vs `status_code`)

**Trace-helper errors** — 6 pytest fixture defects, independent of application logic.