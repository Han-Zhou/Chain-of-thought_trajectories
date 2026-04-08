"""BigBench dataloaders (Movie Recommendation, Causal Judgement)."""

import json
import urllib.request

from dataloader.base import BaseBenchmarkDataset

_BBH_URL = "https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/main/bbh/{}.json"


def _fetch_bbh(config: str) -> list[dict]:
    """Download a BBH config JSON and return the examples list."""
    url = _BBH_URL.format(config)
    with urllib.request.urlopen(url) as resp:
        return json.loads(resp.read())["examples"]


class BigBenchMovieDataLoader(BaseBenchmarkDataset):
    """BigBench Movie Recommendation."""

    def _load_all(self) -> list[dict]:
        rows = _fetch_bbh("movie_recommendation")
        entries = []
        for i, row in enumerate(rows):
            entries.append(self._entry(
                id_=f"bigbench_movie_{i}",
                question=row["input"],
                answer=row["target"],
                source="bigbench/movie_recommendation",
            ))
        print(f"[bigbench_movie] {len(entries)} entries")
        return entries


class BigBenchCausalDataLoader(BaseBenchmarkDataset):
    """BigBench Causal Judgement."""

    def _load_all(self) -> list[dict]:
        rows = _fetch_bbh("causal_judgement")
        entries = []
        for i, row in enumerate(rows):
            entries.append(self._entry(
                id_=f"bigbench_causal_{i}",
                question=row["input"],
                answer=row["target"],
                source="bigbench/causal_judgment",
            ))
        print(f"[bigbench_causal] {len(entries)} entries")
        return entries
