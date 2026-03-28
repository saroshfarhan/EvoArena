"""
Agent Runtime — local reference implementation of the LLM agent loop.

This mirrors the logic inside sandbox_worker.py but runs locally,
making it easy to test a single agent without Daytona.

Usage:
    uv run agent_runtime.py --genome '{"agent_id":"test","planning_depth":3,...}'
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import json
import os
import sys
from pathlib import Path

# Re-use the environment classes and helpers from sandbox_worker
sys.path.insert(0, str(Path(__file__).parent))
from sandbox_worker import (
    CodingEnv,
    GameTheoryEnv,
    PlanningEnv,
    genome_to_system_prompt,
    run_task,
)


def run_agent_locally(genome: dict, api_key: str | None = None) -> dict:
    """Run all three tasks locally and return the same JSON as sandbox_worker."""
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    system_prompt = genome_to_system_prompt(genome)
    rs = max(3, min(7, int(genome.get("reasoning_steps", 4))))

    tasks: dict = {}
    total_calls = 0

    for env_cls, name, budget in [
        (CodingEnv,      "coding",      rs),
        (PlanningEnv,    "planning",    rs),
        (GameTheoryEnv,  "game_theory", GameTheoryEnv.ROUNDS + 3),
    ]:
        try:
            env = env_cls()
            result = run_task(client, system_prompt, env, budget, name)
            tasks[name] = result
            total_calls += result["steps_used"]
        except Exception as e:
            tasks[name] = {"error": str(e), "accuracy": 0.0,
                           "steps_used": 0, "max_steps": budget, "tool_calls": []}

    return {
        "agent_id": genome.get("agent_id", "local"),
        "genome": genome,
        "tasks": tasks,
        "total_llm_calls": total_calls,
    }


if __name__ == "__main__":
    import argparse
    from evaluator import compute_fitness

    parser = argparse.ArgumentParser(description="Run a single EvoArena agent locally")
    parser.add_argument("--genome", type=str, help="Genome JSON string")
    args = parser.parse_args()

    if args.genome:
        g = json.loads(args.genome)
    else:
        # Default test genome
        g = {
            "agent_id": "local_test",
            "planning_depth": 3.0,
            "reasoning_steps": 4,
            "cooperation_bias": 0.8,
            "exploration_rate": 0.3,
            "verification_level": 0.7,
            "risk_bias": 0.4,
            "tool_usage_bias": 0.5,
        }

    print(f"Running agent {g['agent_id']} locally...")
    result = run_agent_locally(g)
    fit = compute_fitness(result["tasks"])

    print(f"\nFitness: {fit['fitness']:.4f}")
    print(f"  accuracy:        {fit['mean_accuracy']:.4f}")
    print(f"  efficiency:      {fit['mean_efficiency']:.4f}")
    print(f"  robustness:      {fit['robustness']:.4f}")
    print(f"  strategic_score: {fit['strategic_score']:.4f}")
    print(f"  total LLM calls: {result['total_llm_calls']}")

    for task_name, task_data in result["tasks"].items():
        print(f"\n  [{task_name}]")
        print(f"    accuracy  : {task_data.get('accuracy', 0):.4f}")
        print(f"    steps     : {task_data.get('steps_used', 0)}/{task_data.get('max_steps', 0)}")
        print(f"    tool_calls: {task_data.get('tool_calls', [])}")
        if "error" in task_data:
            print(f"    error     : {task_data['error']}")
