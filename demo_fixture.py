"""Render the three charts from a synthetic trajectory - no API keys, no spend.

Useful for iterating on visualizer.py without paying for a research run:
    python demo_fixture.py
"""

from __future__ import annotations

from config import Verdict
from evaluator import StepEvaluation, build_report
import visualizer

V, X, N = Verdict.COMPLIANT, Verdict.VIOLATED, Verdict.NOT_APPLICABLE

# A hand-built decay: uncertainty-acknowledgement collapses mid-run and trips
# the breaker at step 10, which is the case the charts exist to show.
SCRIPT = {
    "REASONING_FIRST":         [V, V, V, V, V, X, V, V, X, V, V, X],
    "NO_CONSECUTIVE_TOOLS":    [N, V, V, X, V, X, X, V, X, X, V, V],
    "CITE_SOURCES":            [N, N, V, V, V, X, V, X, X, X, V, X],
    "ACKNOWLEDGE_UNCERTAINTY": [V, V, V, V, X, X, V, X, X, X, X, V],
    "SUMMARIZE_FINDINGS":      [N, V, N, V, V, V, X, V, N, X, V, V],
}


def main() -> None:
    evaluations = [
        StepEvaluation(
            step_index=i + 1,
            verdicts={key: seq[i] for key, seq in SCRIPT.items()},
            reasons={key: "Synthetic fixture verdict." for key in SCRIPT},
        )
        for i in range(12)
    ]
    report = build_report(
        evaluations,
        breaker_events=[{"step": 10, "rolling": 0.4, "continued": True}],
        aborted=False,
    )
    print(f"synthetic ILS: {report.overall_ils:.1f}")
    for path in visualizer.render_all(report):
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
