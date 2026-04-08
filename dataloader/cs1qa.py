"""CS1QA dataloaders.

Provides four loaders:
- CS1QADataLoader      — raw rows (legacy, one entry per JSONL line)
- CS1QATypeDataLoader   — deduplicated, answer = questionType label
- CS1QASpanDataLoader   — deduplicated, answer = list of (startLine, endLine) spans
- CS1QAAnswerDataLoader — deduplicated, answer = free-text answer
"""

from pathlib import Path

from dataloader.base import BaseBenchmarkDataset

_CS1QA_DIR = Path("/storage/backup/han/cot/cs1qa")


def _dedup_key(row: dict) -> tuple:
    """Canonical key for deduplicating CS1QA rows by unique question."""
    return (
        row.get("labNo"),
        row.get("taskNo"),
        row.get("questioner"),
        row.get("question"),
        row.get("code"),
    )


def _load_and_group(path: Path) -> dict[tuple, list[dict]]:
    """Load JSONL and group rows by dedup key."""
    import json
    with open(path, encoding="utf-8") as f:
        try:
            rows = json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            rows = [json.loads(line) for line in f if line.strip()]

    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = _dedup_key(row)
        groups.setdefault(key, []).append(row)
    return groups


def _add_line_numbers(code: str) -> str:
    """Prepend 1-based line numbers to each line of code."""
    lines = code.split("\n")
    return "\n".join(f"{i + 1}: {line}" for i, line in enumerate(lines))


class CS1QADataLoader(BaseBenchmarkDataset):
    """QA dataset for introductory programming courses (Lee et al., 2022).

    Loads the test split from local disk.
    Source: https://github.com/cyoon47/CS1QA
    """

    def _load_all(self) -> list[dict]:
        path = _CS1QA_DIR / "test_cleaned.jsonl"
        rows = self._load_json_or_jsonl(path)
        entries = []
        for i, row in enumerate(rows):
            entries.append(self._entry(
                id_=str(i),
                question=row.get("question", ""),
                answer=str(row.get("answer", "")),
                context=row.get("code", None),
                source="cs1qa",
                question_type=row.get("questionType", None),
            ))
        print(f"[cs1qa] {len(entries)} entries")
        return entries


class CS1QATypeDataLoader(BaseBenchmarkDataset):
    """CS1QA — type classification subtask. One entry per unique question."""

    def _load_all(self) -> list[dict]:
        groups = _load_and_group(_CS1QA_DIR / "test_cleaned.jsonl")
        entries = []
        for i, (key, rows) in enumerate(groups.items()):
            rep = rows[0]
            entries.append(self._entry(
                id_=str(i),
                question=rep["question"],
                answer=rep["questionType"],
                context=rep.get("code"),
                source="cs1qa_type",
                question_type=rep.get("questionType"),
            ))
        print(f"[cs1qa_type] {len(entries)} entries")
        return entries


class CS1QASpanDataLoader(BaseBenchmarkDataset):
    """CS1QA — span prediction subtask. answer = list of (startLine, endLine) tuples."""

    def _load_all(self) -> list[dict]:
        groups = _load_and_group(_CS1QA_DIR / "test_cleaned.jsonl")
        entries = []
        for i, (key, rows) in enumerate(groups.items()):
            rep = rows[0]
            spans = list({(r["startLine"], r["endLine"]) for r in rows})
            code = rep.get("code", "")
            entries.append(self._entry(
                id_=str(i),
                question=rep["question"],
                answer=spans,
                context=_add_line_numbers(code),
                source="cs1qa_span",
                question_type=rep.get("questionType"),
                raw_code=code,
            ))
        print(f"[cs1qa_span] {len(entries)} entries")
        return entries


class CS1QAAnswerDataLoader(BaseBenchmarkDataset):
    """CS1QA — answer retrieval subtask (text generation). answer = free-text answer."""

    def _load_all(self) -> list[dict]:
        groups = _load_and_group(_CS1QA_DIR / "test_cleaned.jsonl")
        entries = []
        for i, (key, rows) in enumerate(groups.items()):
            rep = rows[0]
            answer = rep.get("answer", "")
            if not answer:
                continue
            entries.append(self._entry(
                id_=str(i),
                question=rep["question"],
                answer=answer,
                context=rep.get("code"),
                source="cs1qa_answer",
                question_type=rep.get("questionType"),
            ))
        print(f"[cs1qa_answer] {len(entries)} entries")
        return entries
