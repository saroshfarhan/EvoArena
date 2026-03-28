"""
Task 1 — 0/1 Knapsack
=====================
Tests planning_depth and verification_level.

Genome strategy mapping:
  planning_depth >= 4  → Dynamic Programming   (exact, high step cost)
  planning_depth 2-3   → Greedy (value/weight) (near-optimal, cheap)
  planning_depth 1     → Random search         (poor, variable)

The fitness trade-off means depth=2-3 (greedy) often beats depth=5 (DP)
because efficiency weighs 35% of fitness.
"""

import random

ITEMS = [(2, 6), (2, 10), (3, 12), (5, 13), (5, 15),
         (7, 10), (8, 14), (10, 8), (4, 9), (6, 11)]
CAPACITY = 20


def solve_dp() -> int:
    n, W = len(ITEMS), CAPACITY
    dp = [[0] * (W + 1) for _ in range(n + 1)]
    for i, (w, v) in enumerate(ITEMS):
        for c in range(W + 1):
            dp[i + 1][c] = dp[i][c]
            if c >= w:
                dp[i + 1][c] = max(dp[i + 1][c], dp[i][c - w] + v)
    return dp[n][W]


def solve_greedy() -> int:
    items = sorted(ITEMS, key=lambda x: x[1] / x[0], reverse=True)
    cap, val = CAPACITY, 0
    for w, v in items:
        if cap >= w:
            cap -= w
            val += v
    return val


def solve_random(trials: int = 20, seed: int = 42) -> int:
    rng = random.Random(seed)
    best = 0
    n = len(ITEMS)
    for _ in range(trials):
        perm = rng.sample(range(n), n)
        cap, val = CAPACITY, 0
        for idx in perm:
            w, v = ITEMS[idx]
            if cap >= w:
                cap -= w
                val += v
        best = max(best, val)
    return best


OPTIMAL = solve_dp()


if __name__ == "__main__":
    print(f"DP (optimal) : {solve_dp()}")
    print(f"Greedy       : {solve_greedy()}")
    print(f"Random(20)   : {solve_random()}")
    print(f"Optimal      : {OPTIMAL}")
