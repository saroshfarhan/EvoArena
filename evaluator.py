"""
Fitness computation for EvoArena LLM agents.

fitness = 0.40 * task_accuracy
        + 0.20 * efficiency
        + 0.20 * robustness
        + 0.20 * strategic_score

  task_accuracy  : mean correctness across coding, planning, game_theory (0–1)
  efficiency     : fraction of step budget saved (0–1)
  robustness     : consistency across tasks — 1 minus normalised std-dev (0–1)
  strategic_score: game-theory cooperation quality + verification usage (0–1)
"""

from __future__ import annotations

import math


def compute_fitness(tasks: dict) -> dict:
    """
    tasks: dict keyed by task name, each with at least:
      accuracy    float 0–1
      steps_used  int
      max_steps   int

    game_theory task additionally:
      cooperation_rate  float 0–1

    Returns dict with fitness and component breakdown.
    """
    TASK_NAMES = ("coding", "planning", "game_theory")

    accuracies: list[float] = []
    efficiencies: list[float] = []

    for name in TASK_NAMES:
        t = tasks.get(name, {})
        if not t or "error" in t:
            accuracies.append(0.0)
            efficiencies.append(0.0)
            continue

        acc = float(t.get("accuracy", 0.0))
        steps = int(t.get("steps_used", 0))
        max_s = int(t.get("max_steps", 1))

        eff = max(0.0, 1.0 - steps / max_s) if max_s > 0 else 0.0
        accuracies.append(min(1.0, max(0.0, acc)))
        efficiencies.append(min(1.0, eff))

    mean_acc = sum(accuracies) / len(accuracies) if accuracies else 0.0
    mean_eff = sum(efficiencies) / len(efficiencies) if efficiencies else 0.0

    # Robustness: penalise high variance across tasks
    if len(accuracies) > 1:
        var = sum((a - mean_acc) ** 2 for a in accuracies) / len(accuracies)
        robustness = max(0.0, 1.0 - math.sqrt(var) * 2)
    else:
        robustness = 1.0

    # Strategic score: game_theory cooperation quality
    gt = tasks.get("game_theory", {})
    coop_rate = float(gt.get("cooperation_rate", 0.0)) if gt and "error" not in gt else 0.0
    # Verification bonus: did the coding agent actually run tests?
    coding = tasks.get("coding", {})
    used_verify = "run_tests" in coding.get("tool_calls", []) if coding else False
    verify_bonus = 0.3 if used_verify else 0.0
    strategic_score = min(1.0, coop_rate * 0.7 + verify_bonus)

    fitness = (
        0.40 * mean_acc
        + 0.20 * mean_eff
        + 0.20 * robustness
        + 0.20 * strategic_score
    )

    return {
        "fitness": round(fitness, 4),
        "mean_accuracy": round(mean_acc, 4),
        "mean_efficiency": round(mean_eff, 4),
        "robustness": round(robustness, 4),
        "strategic_score": round(strategic_score, 4),
        "task_accuracies": {n: round(accuracies[i], 4) for i, n in enumerate(TASK_NAMES)},
        "task_efficiencies": {n: round(efficiencies[i], 4) for i, n in enumerate(TASK_NAMES)},
    }
