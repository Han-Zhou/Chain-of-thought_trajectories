"""
Dataset loading utilities for CoT trajectory generation.

Each loader returns a list of dicts with the following normalized fields:
  id        str        unique identifier
  question  str        the question / prompt text
  answer    str        ground-truth answer (JSON-encoded list for BFCL)
  context   str|None   background passage, code snippet, etc.
  choices   list|None  ["A. …", "B. …", …] for multiple-choice tasks
  metadata  dict       dataset-specific extras

Supported dataset names (pass to load()):
  bfcl               Berkeley Function Calling Leaderboard v4 (local files)
  bigbench_movie     BigBench Movie Recommendation
  bigbench_causal    BigBench Causal Judgement
  logiqa             LogiQA
  codeqa             CodeQA
  cs1qa              CS1QA
  hotpotqa           HotPotQA
  college_math       College Math Test  (MMLU-Pro, math category)
  olympiadbench      OlympiadBench
  math500            MATH-500
  hle                Humanity's Last Exam
"""

import json
from pathlib import Path
from typing import Any

# ── paths ─────────────────────────────────────────────────────────────────────

BFCL_DIR = Path("/storage/backup/han/cot/bfcl")

# ── helpers ───────────────────────────────────────────────────────────────────

def _load_json_or_jsonl(path: Path) -> list[dict]:
    """Read a file that is either a JSON array or JSONL."""
    with open(path, encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            return [json.loads(line) for line in f if line.strip()]


def _entry(
    id_: str,
    question: str,
    answer: Any,
    *,
    context: str | None = None,
    choices: list | None = None,
    **metadata,
) -> dict:
    """Build a normalized dataset entry."""
    return {
        "id": id_,
        "question": question,
        "answer": answer,
        "context": context,
        "choices": choices,
        "metadata": metadata,
    }


def _hf(repo: str, split: str, *, name: str | None = None) -> Any:
    """Load a HuggingFace dataset split (no trust_remote_code needed)."""
    from datasets import load_dataset
    kwargs: dict = {}
    if name:
        kwargs["name"] = name
    return load_dataset(repo, split=split, **kwargs)


# ── BFCL (local) ──────────────────────────────────────────────────────────────

def _load_bfcl_type(bfcl_type: str) -> list[dict]:
    """
    Load one BFCL v4 category from local disk.

    File-naming note: for parallel_multiple the question/answer files are
    stored swapped relative to the other categories:
      v4_parallel_multiple.json         → ground truth  (answers)
      v4_parallel_multiple_answer.json  → questions + function specs
    All other types follow:
      v4_{type}.json         → questions + function specs
      v4_{type}_answer.json  → ground truth
    """
    if bfcl_type == "parallel_multiple":
        q_path = BFCL_DIR / "v4_parallel_multiple_answer.json"
        a_path = BFCL_DIR / "v4_parallel_multiple.json"
    else:
        q_path = BFCL_DIR / f"v4_{bfcl_type}.json"
        a_path = BFCL_DIR / f"v4_{bfcl_type}_answer.json"

    questions = _load_json_or_jsonl(q_path)
    answers = {
        row["id"]: row["ground_truth"]
        for row in _load_json_or_jsonl(a_path)
    }

    entries = []
    for q in questions:
        qid = q["id"]
        # question is a list of conversation-turn lists; extract the user turn
        turns = q["question"][0] if q["question"] else []
        user_msg = next(
            (t["content"] for t in turns if t.get("role") == "user"), ""
        )
        ground_truth = answers.get(qid, [])
        entries.append(_entry(
            id_=qid,
            question=user_msg,
            answer=json.dumps(ground_truth),
            functions=q.get("function", []),
            ground_truth=ground_truth,
            bfcl_type=bfcl_type,
        ))
    return entries


def load_bfcl() -> list[dict]:
    """Load all four BFCL v4 categories and return them as one list."""
    all_entries: list[dict] = []
    for bfcl_type in ("simple_python", "multiple", "parallel", "parallel_multiple"):
        entries = _load_bfcl_type(bfcl_type)
        print(f"  [bfcl/{bfcl_type}] {len(entries)} entries")
        all_entries.extend(entries)
    print(f"[bfcl] {len(all_entries)} total entries")
    return all_entries


# ── BigBench ──────────────────────────────────────────────────────────────────

def load_bigbench_movie() -> list[dict]:
    """BigBench Movie Recommendation (tasksource/bigbench)."""
    ds = _hf("tasksource/bigbench", "validation", name="movie_recommendation")
    entries = []
    for i, row in enumerate(ds):
        targets = row.get("targets") or []
        mc = row.get("multiple_choice_targets") or []
        entries.append(_entry(
            id_=f"bigbench_movie_{row.get('idx', i)}",
            question=row["inputs"],
            answer=targets[0] if targets else "",
            choices=mc or None,
            source="bigbench/movie_recommendation",
        ))
    print(f"[bigbench_movie] {len(entries)} entries")
    return entries


def load_bigbench_causal() -> list[dict]:
    """BigBench Causal Judgement (tasksource/bigbench)."""
    # The config name on HF is 'causal_judgment' (no trailing 'e')
    ds = _hf("tasksource/bigbench", "validation", name="causal_judgment")
    entries = []
    for i, row in enumerate(ds):
        targets = row.get("targets") or []
        mc = row.get("multiple_choice_targets") or []
        entries.append(_entry(
            id_=f"bigbench_causal_{row.get('idx', i)}",
            question=row["inputs"],
            answer=targets[0] if targets else "",
            choices=mc or None,
            source="bigbench/causal_judgment",
        ))
    print(f"[bigbench_causal] {len(entries)} entries")
    return entries


# ── LogiQA ────────────────────────────────────────────────────────────────────

def load_logiqa() -> list[dict]:
    """
    LogiQA – Chinese logical-reasoning benchmark (English version).

    Tries multiple HF repos in order; raises RuntimeError if none are
    accessible.  If you have a local copy, set LOGIQA_PATH env var:
      export LOGIQA_PATH=/path/to/logiqa_test.json
    The JSON should be a list of objects with keys:
      context, query, options (list[str]), correct_option (int 0-3)
    """
    import os, pathlib

    local = os.environ.get("LOGIQA_PATH")
    if local:
        data = _load_json_or_jsonl(pathlib.Path(local))
        return _parse_logiqa_rows(data)

    # HF repos to try in order (loading-script repos are blocked by datasets>=2.x)
    repos_to_try = [
        ("Zhufeng1993/LogiQA2.0", "test", None),
        ("McGill-NLP/logiqa2", "test", None),
        ("datatune/logiqa_en", "test", None),
    ]
    for repo, split, name in repos_to_try:
        try:
            ds = _hf(repo, split, name=name)
            rows = list(ds)
            return _parse_logiqa_rows(rows)
        except Exception:
            continue

    raise RuntimeError(
        "Could not load LogiQA from HuggingFace. "
        "Set LOGIQA_PATH=/path/to/local_file.json to use a local copy. "
        "The file should be JSON/JSONL with fields: "
        "context, query, options (list), correct_option (0-based int)."
    )


def _parse_logiqa_rows(rows: list[dict]) -> list[dict]:
    _LETTERS = "ABCD"
    entries = []
    for i, row in enumerate(rows):
        options = row.get("options") or row.get("candidates") or []
        correct = row.get("correct_option", row.get("answer", 0))
        # correct may be int index or letter string
        if isinstance(correct, str) and correct.upper() in _LETTERS:
            answer = correct.upper()
        else:
            answer = _LETTERS[int(correct)] if int(correct) < len(_LETTERS) else str(correct)
        choices = [f"{_LETTERS[j]}. {opt}" for j, opt in enumerate(options)]
        entries.append(_entry(
            id_=str(row.get("id", i)),
            question=row.get("query", row.get("question", "")),
            answer=answer,
            context=row.get("context", None),
            choices=choices or None,
            source="logiqa",
        ))
    print(f"[logiqa] {len(entries)} entries")
    return entries


# ── CodeQA ────────────────────────────────────────────────────────────────────

def load_codeqa() -> list[dict]:
    """
    CodeQA – question answering over source code.

    Uses lissadesu/codeqa_v2 on HuggingFace (fields: question, code, answer).
    Set CODEQA_PATH env var to point to a local JSON/JSONL file with those
    same fields if the HF dataset is unavailable.
    """
    import os, pathlib

    local = os.environ.get("CODEQA_PATH")
    if local:
        rows = _load_json_or_jsonl(pathlib.Path(local))
    else:
        ds = _hf("lissadesu/codeqa_v2", "train")   # only split available
        rows = list(ds)

    entries = []
    for i, row in enumerate(rows):
        entries.append(_entry(
            id_=str(row.get("id", i)),
            question=row["question"],
            answer=row.get("answer", ""),
            context=row.get("code") or row.get("code_processed") or None,
            source="codeqa",
            question_type=row.get("questionType", ""),
        ))
    print(f"[codeqa] {len(entries)} entries")
    return entries


# ── CS1QA ─────────────────────────────────────────────────────────────────────

def load_cs1qa() -> list[dict]:
    """
    CS1QA – QA dataset for introductory programming courses (Lee et al., 2022).

    The canonical dataset is not yet on HuggingFace. Download it from:
      https://github.com/cs1qa/cs1qa
    and set CS1QA_PATH to the local JSON/JSONL file.

    Expected fields: id, question, answer, code (optional)
    """
    import os, pathlib

    local = os.environ.get("CS1QA_PATH")
    if not local:
        raise RuntimeError(
            "CS1QA is not available on HuggingFace. "
            "Download the dataset from https://github.com/cs1qa/cs1qa "
            "and set CS1QA_PATH=/path/to/cs1qa_test.json"
        )

    rows = _load_json_or_jsonl(pathlib.Path(local))
    entries = []
    for i, row in enumerate(rows):
        entries.append(_entry(
            id_=str(row.get("id", i)),
            question=row.get("question", ""),
            answer=str(row.get("answer", "")),
            context=row.get("code", None),
            source="cs1qa",
        ))
    print(f"[cs1qa] {len(entries)} entries")
    return entries


# ── HotPotQA ──────────────────────────────────────────────────────────────────

def load_hotpotqa() -> list[dict]:
    """HotPotQA full-wiki validation split (hotpotqa/hotpot_qa)."""
    ds = _hf("hotpotqa/hotpot_qa", "validation", name="fullwiki")
    entries = []
    for i, row in enumerate(ds):
        # context: {"title": [str, …], "sentences": [[str, …], …]}
        ctx_dict = row.get("context") or {}
        titles = ctx_dict.get("title") or []
        sent_lists = ctx_dict.get("sentences") or []
        context = "\n\n".join(
            f"{t}: {' '.join(s)}" for t, s in zip(titles, sent_lists)
        ) or None

        entries.append(_entry(
            id_=row.get("id", f"hotpotqa_{i}"),
            question=row["question"],
            answer=row["answer"],
            context=context,
            answer_type=row.get("type", ""),
            level=row.get("level", ""),
            source="hotpotqa/fullwiki",
        ))
    print(f"[hotpotqa] {len(entries)} entries")
    return entries


# ── College Math Test ─────────────────────────────────────────────────────────

def load_college_math() -> list[dict]:
    """
    College Math Test – math questions from MMLU-Pro (TIGER-Lab/MMLU-Pro),
    filtered to the 'math' and 'mathematics' categories.
    """
    ds = _hf("TIGER-Lab/MMLU-Pro", "test")
    entries = []
    for i, row in enumerate(ds):
        cat = (row.get("category") or "").lower()
        if cat not in ("math", "mathematics"):
            continue
        options = row.get("options") or []
        choices = [f"{chr(65 + j)}. {opt}" for j, opt in enumerate(options)]
        entries.append(_entry(
            id_=str(row.get("question_id", i)),
            question=row["question"],
            answer=str(row.get("answer", "")),
            choices=choices or None,
            source="mmlu_pro/math",
            category=row.get("category", ""),
            cot_content=row.get("cot_content", ""),
        ))
    print(f"[college_math] {len(entries)} entries")
    return entries


# ── OlympiadBench ─────────────────────────────────────────────────────────────

def load_olympiadbench() -> list[dict]:
    """
    OlympiadBench – English math Olympiad problems (lmms-lab/OlympiadBench).
    Uses the test_en split (open-ended English problems).
    """
    ds = _hf("lmms-lab/OlympiadBench", "test_en")
    entries = []
    for i, row in enumerate(ds):
        # final_answer is a list (possibly multi-part); join as semicolon-separated
        raw_ans = row.get("final_answer") or []
        answer = "; ".join(str(a) for a in raw_ans) if raw_ans else ""
        entries.append(_entry(
            id_=str(row.get("question_id", i)),
            question=row["question"],
            answer=answer,
            context=row.get("context") or None,
            source="olympiadbench",
            subfield=row.get("subfield", ""),
            answer_type=row.get("answer_type", ""),
            is_multiple_answer=row.get("is_multiple_answer", False),
            unit=row.get("unit", ""),
        ))
    print(f"[olympiadbench] {len(entries)} entries")
    return entries


# ── MATH-500 ──────────────────────────────────────────────────────────────────

def load_math500() -> list[dict]:
    """MATH-500 benchmark (HuggingFaceH4/MATH-500)."""
    ds = _hf("HuggingFaceH4/MATH-500", "test")
    entries = []
    for i, row in enumerate(ds):
        entries.append(_entry(
            id_=str(row.get("unique_id", i)),
            question=row["problem"],
            answer=row.get("answer", row.get("solution", "")),
            source="math500",
            solution=row.get("solution", ""),
            level=row.get("level", ""),
            subject=row.get("subject", ""),
        ))
    print(f"[math500] {len(entries)} entries")
    return entries


# ── HLE ───────────────────────────────────────────────────────────────────────

def load_hle() -> list[dict]:
    """
    Humanity's Last Exam (cais/hle).

    This is a gated dataset – you must accept the terms on HuggingFace and set
    a HF_TOKEN environment variable before loading:
      export HF_TOKEN=hf_...
    """
    import os
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HLE (Humanity's Last Exam) is a gated dataset. "
            "Accept the terms at https://huggingface.co/datasets/cais/hle "
            "and set HF_TOKEN=<your_token>."
        )

    from datasets import load_dataset
    ds = load_dataset("cais/hle", split="test", token=token)
    entries = []
    for i, row in enumerate(ds):
        choices = None
        if row.get("answer_type") == "multiple_choice":
            raw = row.get("choices")
            choices = raw if isinstance(raw, list) else None
        entries.append(_entry(
            id_=str(row.get("id", i)),
            question=row["question"],
            answer=str(row.get("answer", "")),
            choices=choices,
            source="hle",
            category=row.get("category", ""),
            answer_type=row.get("answer_type", ""),
        ))
    print(f"[hle] {len(entries)} entries")
    return entries


_LOADERS: dict[str, Any] = {
    "bfcl":            load_bfcl,
    "bigbench_movie":  load_bigbench_movie,
    "bigbench_causal": load_bigbench_causal,
    "logiqa":          load_logiqa,
    "codeqa":          load_codeqa,
    "cs1qa":           load_cs1qa,
    "hotpotqa":        load_hotpotqa,
    "college_math":    load_college_math,
    "olympiadbench":   load_olympiadbench,
    "math500":         load_math500,
    "hle":             load_hle,
}


def load(dataset_name: str) -> list[dict]:
    """
    Load a dataset by name and return a normalized list of entries.
    """
    name = dataset_name.strip().lower()
    loader = _LOADERS.get(name)
    if loader is None:
        valid = ", ".join(sorted(_LOADERS))
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Valid names: {valid}"
        )
    return loader()
