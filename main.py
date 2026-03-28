"""
EvoArena — Agentic Evolutionary Multi-Agent Problem Solving.

LLM-driven agents run inside parallel Daytona sandboxes.
Each agent's genome shapes its system prompt, changing how Claude behaves.
Evolution finds better strategies across generations.
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from daytona_sdk import AsyncDaytona

import anthropic

from evaluator import compute_fitness
from genome import Genome, evolve_population, initialize_population
from orchestrator import run_generation
from sandbox_worker import NegotiationEnv, genome_to_system_prompt, run_task

# ── Config ────────────────────────────────────────────────────────────────────
POPULATION_SIZE = 12
GENERATIONS     = 3
RESULTS_DIR     = Path("results")
W               = 68   # display width

GENOME_TRAITS = [
    "planning_depth", "reasoning_steps", "cooperation_bias",
    "exploration_rate", "verification_level", "risk_bias", "tool_usage_bias",
]


# ── Display helpers ───────────────────────────────────────────────────────────

def banner(text: str) -> None:
    print("\n" + "═" * W)
    print(f"  {text}")
    print("═" * W)


def section(text: str) -> None:
    print(f"\n{'─' * W}")
    print(f"  {text}")
    print("─" * W)


def _bar(val: float, lo: float, hi: float, w: int = 18) -> str:
    n = round((val - lo) / max(hi - lo, 1e-9) * w)
    return "█" * n + "░" * (w - n)


def print_leaderboard(ranked: list[tuple[Genome, float, dict]], gen: int) -> None:
    section(f"LEADERBOARD — Generation {gen}")
    hdr = f"{'Rank':>4}  {'Agent':8}  {'Fitness':>7}  {'Acc':>6}  {'Eff':>6}  {'Robust':>7}  {'Strat':>6}  Lineage"
    print(hdr)
    print("─" * W)
    for rank, (g, fitness, data) in enumerate(ranked, 1):
        fit = compute_fitness(data.get("tasks", {})) if data else {}
        lineage = "seed" if not g.parent_ids else (
            "+".join(p[:6] for p in g.parent_ids[:2])
        )
        print(
            f"{rank:>4}  {g.agent_id:8}  {fitness:>7.4f}"
            f"  {fit.get('mean_accuracy',0):>6.3f}"
            f"  {fit.get('mean_efficiency',0):>6.3f}"
            f"  {fit.get('robustness',0):>7.3f}"
            f"  {fit.get('strategic_score',0):>6.3f}"
            f"  {lineage}"
        )


def print_genome_profile(g: Genome) -> None:
    vals = [
        ("planning_depth",    g.planning_depth,    1.0, 5.0),
        ("reasoning_steps",   g.reasoning_steps,   1,   7  ),
        ("cooperation_bias",  g.cooperation_bias,  0.0, 1.0),
        ("exploration_rate",  g.exploration_rate,  0.0, 1.0),
        ("verification_level",g.verification_level,0.0, 1.0),
        ("risk_bias",         g.risk_bias,         0.0, 1.0),
        ("tool_usage_bias",   g.tool_usage_bias,   0.0, 1.0),
    ]
    for label, val, lo, hi in vals:
        bar = _bar(val, lo, hi)
        print(f"  {label:<22} [{bar}]  {val}")


def print_task_summary(data: dict) -> None:
    tasks = data.get("tasks", {})
    for name in ("coding", "planning", "game_theory"):
        t = tasks.get(name, {})
        if not t:
            continue
        acc = t.get("accuracy", 0)
        steps = t.get("steps_used", 0)
        maxs = t.get("max_steps", 1)
        calls = " → ".join(t.get("tool_calls", []))
        extra = ""
        if name == "coding":
            extra = f"  tests={t.get('passed_tests',0)}/{t.get('total_tests',5)}"
        elif name == "planning":
            extra = f"  value={t.get('best_value',0)}/{t.get('optimal_value',32)}"
        elif name == "game_theory":
            extra = f"  coop={t.get('cooperation_rate',0):.0%}  score={t.get('score',0)}/{t.get('max_score',24)}"
        print(f"    {name:<12} acc={acc:.3f}  steps={steps}/{maxs}{extra}")
        if calls:
            print(f"               tools: {calls}")


def print_fitness_progress(history: list[float]) -> None:
    print("\n  Fitness trajectory:")
    for i, f in enumerate(history, 1):
        bar = "█" * round(f * 30)
        print(f"    Gen {i}: {bar:<30} {f:.4f}")


# ── Evolution loop ────────────────────────────────────────────────────────────

async def main() -> None:
    # Pre-flight checks
    if not os.getenv("DAYTONA_API_KEY") and not os.getenv("DAYTONA_API_URL"):
        print("Warning: DAYTONA_API_KEY not found in environment.")
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set. Add it to .env or export it.")
        sys.exit(1)

    RESULTS_DIR.mkdir(exist_ok=True)

    banner("EvoArena — LLM Agents Evolving in Daytona Sandboxes")
    print(f"  Population : {POPULATION_SIZE} agents")
    print(f"  Generations: {GENERATIONS}")
    print(f"  Model      : claude-haiku-4-5-20251001  (inside each sandbox)")
    print(f"  Tasks      : coding | planning | game_theory")
    print(f"  Started    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    population = initialize_population(POPULATION_SIZE)
    print(f"\n  Initialised {POPULATION_SIZE} random genomes.")

    history_by_gen: list[dict] = []
    best_fitness_per_gen: list[float] = []
    overall_best: tuple[Genome, float, dict] | None = None
    lineage: list[dict] = []

    async with AsyncDaytona() as daytona:
        for gen in range(1, GENERATIONS + 1):
            section(f"GENERATION {gen} / {GENERATIONS}")
            print(f"  Launching {len(population)} sandboxes (max {4} concurrent)...\n")

            t0 = time.time()
            outcomes = await run_generation(daytona, population)
            elapsed = time.time() - t0

            print(f"\n  All sandboxes finished in {elapsed:.1f}s")

            # Attach fitness to genome objects and sort
            ranked = sorted(outcomes, key=lambda x: x[1], reverse=True)

            print_leaderboard(ranked, gen)

            best_genome, best_fit, best_data = ranked[0]
            best_fitness_per_gen.append(best_fit)

            if overall_best is None or best_fit > overall_best[1]:
                overall_best = ranked[0]

            # Show best agent detail
            print(f"\n  Top agent: {best_genome.agent_id}  (fitness {best_fit:.4f})")
            print_genome_profile(best_genome)
            print(f"\n  Task breakdown:")
            print_task_summary(best_data)

            # Fitness stats
            all_fits = [f for _, f, _ in ranked]
            avg_fit = sum(all_fits) / len(all_fits)
            print(f"\n  Gen {gen} stats: best={best_fit:.4f}  avg={avg_fit:.4f}  worst={all_fits[-1]:.4f}")

            if len(best_fitness_per_gen) > 1:
                delta = best_fitness_per_gen[-1] - best_fitness_per_gen[-2]
                pct = delta / best_fitness_per_gen[-2] * 100 if best_fitness_per_gen[-2] else 0
                print(f"  Progress vs last gen: {delta:+.4f}  ({pct:+.1f}%)")

            # Record lineage
            for g, fit, data in ranked:
                lineage.append({
                    "generation": gen,
                    "agent_id": g.agent_id,
                    "fitness": fit,
                    "parent_ids": g.parent_ids,
                })

            # Save generation file
            gen_data = {
                "generation": gen,
                "agents": [
                    {
                        "genome": g.to_dict(),
                        "mean_accuracy":  compute_fitness(d.get("tasks", {})).get("mean_accuracy", 0),
                        "mean_efficiency": compute_fitness(d.get("tasks", {})).get("mean_efficiency", 0),
                    }
                    for g, _, d in ranked
                ],
            }
            history_by_gen.append(gen_data)
            (RESULTS_DIR / f"generation_{gen}.json").write_text(json.dumps(gen_data, indent=2))

            # Evolve (skip after final generation)
            if gen < GENERATIONS:
                for g, fit, _ in ranked:
                    g.fitness = fit
                population = evolve_population(ranked_as_genomes(ranked), POPULATION_SIZE)
                print(f"\n  Evolved → {POPULATION_SIZE} agents for generation {gen + 1}")

    # ── Final summary ──────────────────────────────────────────────────────────
    banner("FINAL RESULTS")
    print_fitness_progress(best_fitness_per_gen)

    if overall_best:
        best_g, best_f, best_d = overall_best
        print(f"\n  Best genome: {best_g.agent_id}  fitness={best_f:.4f}  gen={best_g.generation}")
        print_genome_profile(best_g)

    # ── Generalization holdout test ────────────────────────────────────────────
    holdout_result: dict = {}
    if overall_best:
        banner("GENERALIZATION TEST — Negotiation (never seen during evolution)")
        best_g, _, _ = overall_best
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        try:
            client = anthropic.Anthropic(api_key=api_key)
            system_prompt = genome_to_system_prompt(best_g.to_dict())
            budget = NegotiationEnv.ROUNDS + 3
            env = NegotiationEnv()
            result = run_task(client, system_prompt, env, budget, "negotiation")
            holdout_result = {
                "agent_id": best_g.agent_id,
                "genome": best_g.to_dict(),
                "negotiation": result,
            }
            score = result["accuracy"]
            baseline = 0.5
            beat = score > baseline
            print(f"  Agent     : {best_g.agent_id}")
            print(f"  Score     : {score:.4f}  (random baseline ≈ {baseline:.2f})")
            print(f"  Earned    : {result.get('total_earned', '?')}/{result.get('max_score', 30)}")
            print(f"  Tools used: {' → '.join(result.get('tool_calls', []))}")
            print(f"  Beat baseline? {'YES — generalised!' if beat else 'NO — did not generalise'}")
        except Exception as e:
            print(f"  Holdout failed: {e}")
            holdout_result = {"error": str(e)}

        (RESULTS_DIR / "holdout_result.json").write_text(json.dumps(holdout_result, indent=2))

    # Save artefacts
    final = {
        "run_at": datetime.now().isoformat(),
        "config": {
            "population_size": POPULATION_SIZE,
            "generations": GENERATIONS,
            "max_concurrency": 4,
            "model": "claude-haiku-4-5-20251001",
        },
        "fitness_trajectory": best_fitness_per_gen,
        "history": history_by_gen,
        "best_agent": overall_best[0].to_dict() if overall_best else {},
    }
    (RESULTS_DIR / "final_results.json").write_text(json.dumps(final, indent=2))
    (RESULTS_DIR / "best_agent.json").write_text(
        json.dumps(overall_best[0].to_dict() if overall_best else {}, indent=2)
    )
    (RESULTS_DIR / "lineage.json").write_text(json.dumps(lineage, indent=2))

    print(f"\n  Saved to {RESULTS_DIR}/")
    for fname in ("final_results.json", "best_agent.json", "lineage.json",
                  "holdout_result.json",
                  *(f"generation_{g}.json" for g in range(1, GENERATIONS + 1))):
        print(f"    • {fname}")

    banner("Done.")


def ranked_as_genomes(ranked: list[tuple[Genome, float, dict]]) -> list[Genome]:
    for g, fit, _ in ranked:
        g.fitness = fit
    return [g for g, _, _ in ranked]


if __name__ == "__main__":
    asyncio.run(main())
