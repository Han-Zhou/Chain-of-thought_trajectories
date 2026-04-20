"""METEOR scoring helper for CodeQA evaluation.

This ports the original CodeQA METEOR wrapper closely enough for local
evaluation while being a bit more defensive about missing external
dependencies. The scorer expects a Java runtime plus a `meteor-1.5.jar`.
"""

from __future__ import annotations

import atexit
import logging
import os
import shutil
import subprocess
import threading

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency
    psutil = None


class MeteorUnavailableError(RuntimeError):
    """Raised when the external METEOR runtime cannot be started."""


def _candidate_jar_paths() -> list[str]:
    here = os.path.dirname(os.path.abspath(__file__))
    env_paths = [
        os.environ.get("CODEQA_METEOR_JAR"),
        os.environ.get("METEOR_JAR"),
    ]
    repo_local = os.path.join(here, "vendor", "meteor-1.5.jar")
    sibling_codeqa = os.path.abspath(
        os.path.join(here, "..", "..", "codeqa", "codeBERT", "code", "eval", "meteor", "meteor-1.5.jar")
    )
    return [p for p in [*env_paths, repo_local, sibling_codeqa] if p]


def resolve_meteor_jar(jar_path: str | None = None) -> str:
    """Resolve the METEOR jar path or raise a clear error."""
    candidates = [jar_path] if jar_path is not None else _candidate_jar_paths()
    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)

    searched = "\n".join(f"  - {os.path.abspath(path)}" for path in candidates)
    raise MeteorUnavailableError(
        "Could not find meteor-1.5.jar. Set CODEQA_METEOR_JAR or METEOR_JAR "
        f"to the jar path.\nSearched:\n{searched}"
    )


def _enc(text: str) -> bytes:
    return text.encode("utf-8")


def _dec(blob: bytes) -> str:
    return blob.decode("utf-8")


class Meteor:
    """Thread-safe wrapper around the METEOR Java jar."""

    def __init__(self, jar_path: str | None = None, java_bin: str = "java"):
        self.lock = threading.Lock()
        self.jar_path = resolve_meteor_jar(jar_path)

        java_path = shutil.which(java_bin)
        if java_path is None:
            raise MeteorUnavailableError(
                "Could not find a Java runtime for METEOR. Install `java` or set "
                "PATH so the `java` executable is available."
            )

        mem = "2G"
        if psutil is not None:
            mem_available_gb = psutil.virtual_memory().available / 1e9
            if mem_available_gb < 2:
                logging.warning(
                    "There is less than 2GB of available memory. "
                    "Falling back to a 1GB METEOR heap."
                )
                mem = "1G"

        meteor_cmd = [
            java_path,
            f"-Xmx{mem}",
            "-jar",
            self.jar_path,
            "-",
            "-",
            "-stdio",
            "-l",
            "en",
            "-norm",
        ]
        env = os.environ.copy()
        env["LC_ALL"] = "C"

        try:
            self.meteor_p = subprocess.Popen(
                meteor_cmd,
                cwd=os.path.dirname(self.jar_path),
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as exc:
            raise MeteorUnavailableError(
                f"Failed to start METEOR using jar at {self.jar_path}: {exc}"
            ) from exc

        atexit.register(self.close)

    def close(self):
        with self.lock:
            if getattr(self, "meteor_p", None):
                self.meteor_p.kill()
                self.meteor_p.wait()
                self.meteor_p = None
        if atexit is not None and getattr(atexit, "unregister", None) is not None:
            try:
                atexit.unregister(self.close)
            except Exception:  # pragma: no cover - best effort cleanup
                pass

    def _stat(self, hypothesis: str, references: list[str]) -> str:
        hypothesis = hypothesis.replace("|||", "").replace("  ", " ")
        score_line = " ||| ".join(("SCORE", " ||| ".join(references), hypothesis))
        assert self.meteor_p.stdin is not None
        assert self.meteor_p.stdout is not None
        self.meteor_p.stdin.write(_enc(score_line))
        self.meteor_p.stdin.write(_enc("\n"))
        self.meteor_p.stdin.flush()
        return _dec(self.meteor_p.stdout.readline()).strip()

    def compute_score(self, references: dict[int, list[str]], hypotheses: dict[int, list[str]]):
        assert references.keys() == hypotheses.keys()
        ids = list(references.keys())
        scores = []

        eval_line = "EVAL"
        with self.lock:
            for key in ids:
                assert len(hypotheses[key]) == 1
                stat = self._stat(hypotheses[key][0], references[key])
                eval_line += f" ||| {stat}"

            assert self.meteor_p.stdin is not None
            assert self.meteor_p.stdout is not None
            self.meteor_p.stdin.write(_enc(f"{eval_line}\n"))
            self.meteor_p.stdin.flush()

            for _ in ids:
                value = self.meteor_p.stdout.readline()
                try:
                    scores.append(float(_dec(value.strip())))
                except Exception as exc:
                    raise RuntimeError(
                        f"Error parsing METEOR output {value!r} for eval line {eval_line!r}"
                    ) from exc

            score = float(_dec(self.meteor_p.stdout.readline()).strip())

        return score, scores

    def __del__(self):  # pragma: no cover - destructor timing is interpreter-dependent
        self.close()


_SHARED_METEOR: Meteor | None = None
_SHARED_LOCK = threading.Lock()


def get_shared_meteor() -> Meteor:
    """Reuse one METEOR subprocess across many CodeQA examples."""
    global _SHARED_METEOR
    with _SHARED_LOCK:
        if _SHARED_METEOR is None:
            _SHARED_METEOR = Meteor()
        return _SHARED_METEOR


def close_shared_meteor():
    """Close the cached METEOR subprocess, if one exists."""
    global _SHARED_METEOR
    with _SHARED_LOCK:
        if _SHARED_METEOR is not None:
            _SHARED_METEOR.close()
            _SHARED_METEOR = None


def compute_meteor_score(prediction: str, ground_truths: list[str]) -> float:
    """Compute a sentence-level METEOR score in the 0-1 range."""
    meteor = get_shared_meteor()
    score, _ = meteor.compute_score({0: list(ground_truths)}, {0: [prediction]})
    return score
