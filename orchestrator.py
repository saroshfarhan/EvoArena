"""
Orchestrator — spawns Daytona sandboxes and gathers agent results.

For each agent in the population:
  1. Create an ephemeral Daytona sandbox
  2. Install anthropic inside the sandbox
  3. Upload sandbox_worker.py via base64
  4. Execute the worker (passes GENOME + ANTHROPIC_API_KEY as env vars)
  5. Parse JSON output
  6. Stop the sandbox

Up to MAX_CONCURRENCY sandboxes run in parallel (asyncio.Semaphore).
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time
from pathlib import Path

from daytona_sdk import AsyncDaytona, CreateSandboxFromSnapshotParams

from evaluator import compute_fitness
from genome import Genome

MAX_CONCURRENCY = 4
INSTALL_TIMEOUT = 120   # seconds — pip install anthropic
RUN_TIMEOUT     = 300   # seconds — full agent run
WORKER_FILE     = Path(__file__).parent / "sandbox_worker.py"


async def _run_one(
    daytona: AsyncDaytona,
    genome: Genome,
    worker_b64: str,
    api_key: str,
    sem: asyncio.Semaphore,
    idx: int,
    total: int,
) -> tuple[Genome, float, dict]:
    async with sem:
        # ── Create sandbox ────────────────────────────────────────────
        try:
            sandbox = await daytona.create(
                CreateSandboxFromSnapshotParams(language="python", ephemeral=True)
            )
        except Exception as e:
            print(f"  [{idx:>2}/{total}] {genome.agent_id} | create FAILED: {e}")
            return genome, 0.0, {"error": str(e), "tasks": {}}

        print(f"  [{idx:>2}/{total}] {genome.agent_id} | sandbox up → installing deps...")

        try:
            # ── Install anthropic ─────────────────────────────────────
            await sandbox.process.exec(
                "python -m pip install anthropic -q --disable-pip-version-check",
                timeout=INSTALL_TIMEOUT,
            )

            # ── Upload worker via base64 ──────────────────────────────
            await sandbox.process.exec(
                "python3 -c \""
                "import base64; "
                f"open('/tmp/worker.py','wb').write(base64.b64decode('{worker_b64}'))"
                "\"",
                timeout=30,
            )

            # ── Run agent ─────────────────────────────────────────────
            t0 = time.time()
            run_result = await sandbox.process.exec(
                "python /tmp/worker.py",
                env={
                    "GENOME": json.dumps(genome.to_dict()),
                    "ANTHROPIC_API_KEY": api_key,
                },
                timeout=RUN_TIMEOUT,
            )
            elapsed = time.time() - t0

            raw = (run_result.result or "").strip()

            # Parse the last JSON line (guards against stray pip output)
            data: dict | None = None
            for line in reversed(raw.splitlines()):
                line = line.strip()
                if line.startswith("{"):
                    try:
                        data = json.loads(line)
                        break
                    except json.JSONDecodeError:
                        continue

            if data is None:
                raise ValueError(f"No JSON found in output: {raw[:400]!r}")

            fit = compute_fitness(data.get("tasks", {}))
            genome.fitness = fit["fitness"]

            print(
                f"  [{idx:>2}/{total}] {genome.agent_id} | "
                f"fitness={genome.fitness:.4f}  "
                f"acc={fit['mean_accuracy']:.3f}  "
                f"eff={fit['mean_efficiency']:.3f}  "
                f"llm_calls={data.get('total_llm_calls',0)}  "
                f"({elapsed:.0f}s)"
            )
            return genome, genome.fitness, data

        except Exception as e:
            print(f"  [{idx:>2}/{total}] {genome.agent_id} | ERROR: {e}")
            return genome, 0.0, {"error": str(e), "tasks": {}}

        finally:
            try:
                await sandbox.stop()
            except Exception:
                pass


async def run_generation(
    daytona: AsyncDaytona,
    population: list[Genome],
) -> list[tuple[Genome, float, dict]]:
    """Launch all agents in parallel and return (genome, fitness, raw_data) tuples."""
    worker_code = WORKER_FILE.read_text()
    worker_b64 = base64.b64encode(worker_code.encode()).decode()
    api_key = os.getenv("ANTHROPIC_API_KEY", "")

    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    coros = [
        _run_one(daytona, genome, worker_b64, api_key, sem, i + 1, len(population))
        for i, genome in enumerate(population)
    ]

    results = await asyncio.gather(*coros, return_exceptions=True)

    # Unwrap any unexpected exceptions from gather
    cleaned: list[tuple[Genome, float, dict]] = []
    for genome, item in zip(population, results):
        if isinstance(item, Exception):
            cleaned.append((genome, 0.0, {"error": str(item), "tasks": {}}))
        else:
            cleaned.append(item)

    return cleaned
