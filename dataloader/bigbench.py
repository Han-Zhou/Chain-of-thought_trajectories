"""BigBench dataloaders (Movie Recommendation, Causal Judgement)."""

from dataloader.base import BaseBenchmarkDataset


class BigBenchMovieDataLoader(BaseBenchmarkDataset):
    """BigBench Movie Recommendation (tasksource/bigbench)."""

    def _load_all(self) -> list[dict]:
        ds = self._hf("tasksource/bigbench", "validation", name="movie_recommendation")
        entries = []
        for i, row in enumerate(ds):
            targets = row.get("targets") or []
            mc = row.get("multiple_choice_targets") or []
            mc_scores = row.get("multiple_choice_scores") or []

            # Convert text answer to MCQ letter (A/B/C/D)
            answer_letter = ""
            for j, score in enumerate(mc_scores):
                if score == 1:
                    answer_letter = chr(ord("A") + j)
                    break
            if not answer_letter:
                answer_letter = targets[0] if targets else ""

            entries.append(self._entry(
                id_=f"bigbench_movie_{row.get('idx', i)}",
                question=row["inputs"],
                answer=answer_letter,
                choices=mc or None,
                source="bigbench/movie_recommendation",
            ))
        print(f"[bigbench_movie] {len(entries)} entries")
        return entries


class BigBenchCausalDataLoader(BaseBenchmarkDataset):
    """BigBench Causal Judgement (tasksource/bigbench)."""

    def _load_all(self) -> list[dict]:
        # The config name on HF is 'causal_judgment' (no trailing 'e')
        ds = self._hf("tasksource/bigbench", "validation", name="causal_judgment")
        entries = []
        for i, row in enumerate(ds):
            targets = row.get("targets") or []
            mc = row.get("multiple_choice_targets") or []
            entries.append(self._entry(
                id_=f"bigbench_causal_{row.get('idx', i)}",
                question=row["inputs"],
                answer=targets[0] if targets else "",
                choices=mc or None,
                source="bigbench/causal_judgment",
            ))
        print(f"[bigbench_causal] {len(entries)} entries")
        return entries
