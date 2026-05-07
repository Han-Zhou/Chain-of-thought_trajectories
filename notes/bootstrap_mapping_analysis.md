# Bootstrap to Full Trajectory Mapping Analysis

## Table 1: Question Set Matching (Same Order)

This table shows whether bootstrap trajectories contain the exact same questions in the exact same order as sample_0 from full trajectories.

| Bootstrap Directory | Full Directory | Dataset | Total | Matches | Status |
|---------------------|----------------|---------|-------|---------|--------|
| bootstrap-bc-503-qwen | MERGED-cots-501-qwen-bc-8 | bigbench_causal | 187 | 187 | ✅ Perfect Match |
| bootstrap-bc-504-llama | cots-420-llama-bc-32 | bigbench_causal | 187 | 187 | ✅ Perfect Match |
| bootstrap-bfcl-503-qwen | MERGED-cots-503-qwen-bfcl-8 | bfcl | 500 | 2 | ❌ Mismatch |
| bootstrap-bfcl-504-llama | cots-420-llama-bfcl-32 | bfcl | 500 | 500 | ✅ Perfect Match |
| bootstrap-bm-504-llama | cots-420-llama-bm-32 | bigbench_movie | 250 | 250 | ✅ Perfect Match |
| bootstrap-codeqa-503-qwen | MERGED-cots-503-qwen-codeqa-8 | codeqa | 500 | 1 | ❌ Mismatch |
| bootstrap-codeqa-504-llama | cots-422-llama-codeqa-32 | codeqa | 500 | 1 | ❌ Mismatch |
| bootstrap-cs1qa-span-503-llama | cots-427-llama-cs1qa_span-32 | cs1qa_span | 500 | 0 | ❌ Mismatch |
| bootstrap-hotpotqa-503-llama | cots-427-llama-hotpotqa-32 | hotpotqa | 500 | 0 | ❌ Mismatch |
| bootstrap-hotpotqa-503-qwen | MERGED-cots-503-qwen-hotpotqa-8 | hotpotqa | 500 | 0 | ❌ Mismatch |
| bootstrap-logiqa-503-qwen | MERGED-cots-503-qwen-logiqa-8 | logiqa | 500 | 0 | ❌ Mismatch |
| bootstrap-logiqa-504-llama | cots-420-llama-logiqa-32 | logiqa | 500 | 500 | ✅ Perfect Match |
| bootstrap-m500-504-llama | cots-420-llama-m500-32 | math500 | 498 | 498 | ✅ Perfect Match |
| bootstrap-math500-503-qwen | MERGED-cots-502-qwen-m500-8 | math500 | 498 | 498 | ✅ Perfect Match |

**Summary:** 8/14 perfect matches, 6/14 mismatches

---

## Table 2: Question Set Matching (Regardless of Order)

This table shows whether bootstrap and full directories contain the same set of questions, even if in different order.

| Bootstrap Directory | Full Directory | Dataset | Total | Overlap | Only Bootstrap | Only Full | Status |
|---------------------|----------------|---------|-------|---------|----------------|-----------|--------|
| bootstrap-bc-503-qwen | MERGED-cots-501-qwen-bc-8 | bigbench_causal | 187 | 185 | 0 | 0 | ✅ Same Set |
| bootstrap-bc-504-llama | cots-420-llama-bc-32 | bigbench_causal | 187 | 185 | 0 | 0 | ✅ Same Set |
| bootstrap-bfcl-503-qwen | MERGED-cots-503-qwen-bfcl-8 | bfcl | 500 | 270 | 226 | 203 | ❌ Different Sets |
| bootstrap-bfcl-504-llama | cots-420-llama-bfcl-32 | bfcl | 500 | 473 | 0 | 0 | ✅ Same Set |
| bootstrap-bm-504-llama | cots-420-llama-bm-32 | bigbench_movie | 250 | 250 | 0 | 0 | ✅ Same Set |
| bootstrap-codeqa-503-qwen | MERGED-cots-503-qwen-codeqa-8 | codeqa | 500 | 3 | 314 | 382 | ❌ Different Sets |
| bootstrap-codeqa-504-llama | cots-422-llama-codeqa-32 | codeqa | 500 | 3 | 314 | 382 | ❌ Different Sets |
| bootstrap-cs1qa-span-503-llama | cots-427-llama-cs1qa_span-32 | cs1qa_span | 500 | 268 | 222 | 218 | ❌ Different Sets |
| bootstrap-hotpotqa-503-llama | cots-427-llama-hotpotqa-32 | hotpotqa | 500 | 33 | 467 | 467 | ❌ Different Sets |
| bootstrap-hotpotqa-503-qwen | MERGED-cots-503-qwen-hotpotqa-8 | hotpotqa | 500 | 33 | 467 | 467 | ❌ Different Sets |
| bootstrap-logiqa-503-qwen | MERGED-cots-503-qwen-logiqa-8 | logiqa | 500 | 363 | 95 | 103 | ❌ Different Sets |
| bootstrap-logiqa-504-llama | cots-420-llama-logiqa-32 | logiqa | 500 | 466 | 0 | 0 | ✅ Same Set |
| bootstrap-m500-504-llama | cots-420-llama-m500-32 | math500 | 498 | 498 | 0 | 0 | ✅ Same Set |
| bootstrap-math500-503-qwen | MERGED-cots-502-qwen-m500-8 | math500 | 498 | 498 | 0 | 0 | ✅ Same Set |

**Summary:** 8/14 same question sets, 6/14 different question sets

---

## Table 3: Comparison with Pickle Data

This table shows whether bootstrap and full directories contain questions from the canonical pickle files.

| Bootstrap Directory | Full Directory | Dataset | Pickle File | Bootstrap in Pickle | Full in Pickle | Bootstrap Status | Full Status |
|---------------------|----------------|---------|-------------|---------------------|----------------|------------------|-------------|
| bootstrap-bc-503-qwen | MERGED-cots-501-qwen-bc-8 | bigbench_causal | bigbench_causal.pkl | 185/185 (100%) | 185/185 (100%) | ✅ Valid | ✅ Valid |
| bootstrap-bc-504-llama | cots-420-llama-bc-32 | bigbench_causal | bigbench_causal.pkl | 185/185 (100%) | 185/185 (100%) | ✅ Valid | ✅ Valid |
| bootstrap-bfcl-503-qwen | MERGED-cots-503-qwen-bfcl-8 | bfcl | bfcl.pkl | 270/496 (54%) | 473/473 (100%) | ❌ Invalid | ✅ Valid |
| bootstrap-bfcl-504-llama | cots-420-llama-bfcl-32 | bfcl | bfcl.pkl | 473/473 (100%) | 473/473 (100%) | ✅ Valid | ✅ Valid |
| bootstrap-bm-504-llama | cots-420-llama-bm-32 | bigbench_movie | bigbench_movie.pkl | 250/250 (100%) | 250/250 (100%) | ✅ Valid | ✅ Valid |
| bootstrap-codeqa-503-qwen | MERGED-cots-503-qwen-codeqa-8 | codeqa | codeqa.pkl | 3/317 (1%) | 385/385 (100%) | ❌ Invalid | ✅ Valid |
| bootstrap-codeqa-504-llama | cots-422-llama-codeqa-32 | codeqa | codeqa.pkl | 3/317 (1%) | 385/385 (100%) | ❌ Invalid | ✅ Valid |
| bootstrap-cs1qa-span-503-llama | cots-427-llama-cs1qa_span-32 | cs1qa_span | cs1qa_span.pkl | 268/490 (55%) | 486/486 (100%) | ❌ Invalid | ✅ Valid |
| bootstrap-hotpotqa-503-llama | cots-427-llama-hotpotqa-32 | hotpotqa | hotpotqa.pkl | 33/500 (7%) | 500/500 (100%) | ❌ Invalid | ✅ Valid |
| bootstrap-hotpotqa-503-qwen | MERGED-cots-503-qwen-hotpotqa-8 | hotpotqa | hotpotqa.pkl | 33/500 (7%) | 500/500 (100%) | ❌ Invalid | ✅ Valid |
| bootstrap-logiqa-503-qwen | MERGED-cots-503-qwen-logiqa-8 | logiqa | logiqa.pkl | 363/458 (79%) | 466/466 (100%) | ❌ Invalid | ✅ Valid |
| bootstrap-logiqa-504-llama | cots-420-llama-logiqa-32 | logiqa | logiqa.pkl | 466/466 (100%) | 466/466 (100%) | ✅ Valid | ✅ Valid |
| bootstrap-m500-504-llama | cots-420-llama-m500-32 | math500 | math500.pkl | 498/498 (100%) | 498/498 (100%) | ✅ Valid | ✅ Valid |
| bootstrap-math500-503-qwen | MERGED-cots-502-qwen-m500-8 | math500 | math500.pkl | 498/498 (100%) | 498/498 (100%) | ✅ Valid | ✅ Valid |

**Summary:** 
- Bootstrap: 8/14 valid (100% in pickle), 6/14 invalid (contain questions NOT in pickle)
- Full: 14/14 valid (all are 100% subsets of pickle data)

---

## Key Findings

1. **Correct Mappings (8/14):** These bootstrap directories correctly correspond to their mapped full trajectory directories:
   - bootstrap-bc-503-qwen, bootstrap-bc-504-llama (bigbench_causal)
   - bootstrap-bfcl-504-llama (bfcl)
   - bootstrap-bm-504-llama (bigbench_movie)
   - bootstrap-logiqa-504-llama (logiqa)
   - bootstrap-m500-504-llama, bootstrap-math500-503-qwen (math500)

2. **Incorrect Mappings (6/14):** These bootstrap directories contain questions from a different data source:
   - bootstrap-bfcl-503-qwen (bfcl)
   - bootstrap-codeqa-503-qwen, bootstrap-codeqa-504-llama (codeqa)
   - bootstrap-cs1qa-span-503-llama (cs1qa_span)
   - bootstrap-hotpotqa-503-llama, bootstrap-hotpotqa-503-qwen (hotpotqa)
   - bootstrap-logiqa-503-qwen (logiqa)

3. **All full trajectory directories are valid:** Every full trajectory directory contains questions that are 100% subsets of the canonical pickle data.

4. **Pattern:** The invalid bootstrap directories appear to be from different dataset samples or versions that don't match the canonical pickle files used for the full trajectories.
