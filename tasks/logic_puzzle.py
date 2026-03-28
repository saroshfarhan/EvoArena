"""
Task 2 — Hill-climbing Optimisation
====================================
Tests reasoning_steps, exploration_rate, and verification_level.

Maximise f(x1..x5) = 100 − (Σxi − 25)² − Σ|xi − 5|
  xi ∈ {0..9}.  Optimum at xi=5 for all i → f = 100.

Genome strategy mapping:
  reasoning_steps  → n_iters = steps * 10  (more iters → closer to optimal)
  exploration_rate → search radius         (too big = random walk, too small = stuck)
  verification_level > 0.5 → final ±1 polish pass

Sweet spot: reasoning_steps≈5, exploration_rate≈0.3, verification_level>0.5
"""

import random


def f(xs: list[int]) -> float:
    return 100.0 - (sum(xs) - 25) ** 2 - sum(abs(x - 5) for x in xs)


def solve(reasoning_steps: int = 5,
          exploration_rate: float = 0.3,
          verification_level: float = 0.6,
          seed: int = 77) -> dict:
    rng = random.Random(seed)
    n_iters = reasoning_steps * 10

    best_xs = [rng.randint(0, 9) for _ in range(5)]
    best_val = f(best_xs)

    for _ in range(n_iters):
        radius = max(1, round(exploration_rate * 4 + 0.5))
        candidate = [max(0, min(9, x + rng.randint(-radius, radius)))
                     for x in best_xs]
        val = f(candidate)
        if val > best_val:
            best_val = val
            best_xs = candidate

    if verification_level > 0.5:
        for i in range(5):
            for delta in (-1, 1):
                c = best_xs[:]
                c[i] = max(0, min(9, c[i] + delta))
                val = f(c)
                if val > best_val:
                    best_val = val
                    best_xs = c

    return {"xs": best_xs, "value": best_val, "optimal": 100,
            "accuracy": best_val / 100}


if __name__ == "__main__":
    for rs in [1, 3, 5, 10]:
        r = solve(reasoning_steps=rs)
        print(f"steps={rs:2d} → value={r['value']:6.1f}  acc={r['accuracy']:.3f}")
