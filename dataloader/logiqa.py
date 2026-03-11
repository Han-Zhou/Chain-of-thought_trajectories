"""LogiQA dataloader.

Parsing logic adapted from the HuggingFace dataset script by lucasmccabe/logiqa.
Source: https://github.com/lgw863/LogiQA-dataset
"""

import re
import tempfile
import urllib.request

import datasets as hf_datasets

from dataloader.base import BaseBenchmarkDataset


_URLS = {
    "en_train": "https://raw.githubusercontent.com/lgw863/LogiQA-dataset/master/Train.txt",
    "en_test":  "https://raw.githubusercontent.com/lgw863/LogiQA-dataset/master/Test.txt",
    "en_eval":  "https://raw.githubusercontent.com/lgw863/LogiQA-dataset/master/Eval.txt",
}

_LETTERS = "ABCD"


# ---------------------------------------------------------------------------
# Raw-file parser (ported from lucasmccabe/logiqa HF script)
# ---------------------------------------------------------------------------

def _process_answer(answer: str) -> str:
    if not any(answer.startswith(x) for x in "ABCD"):
        return answer
    return answer[3:]


def _process_sentences(text: str) -> str:
    text = text.replace("\n", "")
    sents = text.split(".")
    result = ""
    for sent in sents:
        if not sent:
            continue
        if not result:
            result += sent
        elif sent[0].isnumeric():
            result += "." + sent
        else:
            result += ". " + sent
    result = result.replace("  ", " ").replace("\\'", "'")
    result = result.rstrip()
    if re.match(r'^[A-Z][\w\s]+[?.!]$', result) is None:
        result += "."
    result = result.replace("?.", "?").replace("!.", "!").replace("..", ".")
    return result


def _generate_examples(filepath: str):
    with open(filepath, encoding="utf-8") as f:
        logiqa = [_process_sentences(line) for line in f.readlines()]

    for key in range(int(len(logiqa) / 8)):
        row = 8 * key
        correct_answer = logiqa[row + 1].replace(".", "")
        context = logiqa[row + 2]
        query = logiqa[row + 3]
        answers = logiqa[row + 4 : row + 8]
        yield key, {
            "context": context,
            "query": query,
            "options": [_process_answer(answers[i]) for i in range(4)],
            "correct_option": "abcd".index(correct_answer),
        }


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------

class LogiQADataLoader(BaseBenchmarkDataset):
    """Load the LogiQA English test split directly from GitHub."""

    def __init__(self, split: str = "en_test") -> None:
        self.split = split
        super().__init__()

    def _load_all(self) -> list[dict]:
        url = _URLS[self.split]
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".txt", delete=False) as tmp:
            tmp_path = tmp.name
        urllib.request.urlretrieve(url, tmp_path)

        rows = [ex for _, ex in _generate_examples(tmp_path)]
        

        entries = self._parse_rows(rows)


        print(f"[logiqa] {len(entries)} entries")
        return entries

    def _parse_rows(self, rows: list[dict]) -> list[dict]:
        entries = []
        for i, row in enumerate(rows):
            options = row.get("options") or row.get("candidates") or []
            correct = row.get("correct_option", row.get("answer", 0))
            if isinstance(correct, str) and correct.upper() in _LETTERS:
                answer = correct.upper()
            else:
                answer = _LETTERS[int(correct)] if int(correct) < len(_LETTERS) else str(correct)
            choices = [f"{_LETTERS[j]}. {opt}" for j, opt in enumerate(options)]
            entries.append(self._entry(
                id_=str(row.get("id", i)),
                question=row.get("query", row.get("question", "")),
                answer=answer,
                context=row.get("context", None),
                choices=choices or None,
                source="logiqa",
            ))
        return entries
