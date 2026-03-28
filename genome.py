"""
Genome definition for EvoArena LLM agents.

Seven traits shape both the system prompt and the agent's decision style:

  planning_depth    float 1–5   how deeply to plan before acting
  reasoning_steps   int   1–8   tool-call budget per task (hard constraint)
  cooperation_bias  float 0–1   cooperative vs competitive in game theory
  exploration_rate  float 0–1   creative vs safe approaches
  verification_level float 0–1  how much to verify before finishing
  risk_bias         float 0–1   willingness to attempt risky moves
  tool_usage_bias   float 0–1   preference for re-using tools vs moving on
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field


@dataclass
class Genome:
    agent_id: str
    planning_depth: float       # 1.0–5.0
    reasoning_steps: int        # 1–8   (tool-call budget per task)
    cooperation_bias: float     # 0–1
    exploration_rate: float     # 0–1
    verification_level: float   # 0–1
    risk_bias: float            # 0–1
    tool_usage_bias: float      # 0–1
    generation: int = 0
    fitness: float = 0.0
    parent_ids: list = field(default_factory=list)

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "planning_depth": round(self.planning_depth, 3),
            "reasoning_steps": self.reasoning_steps,
            "cooperation_bias": round(self.cooperation_bias, 4),
            "exploration_rate": round(self.exploration_rate, 4),
            "verification_level": round(self.verification_level, 4),
            "risk_bias": round(self.risk_bias, 4),
            "tool_usage_bias": round(self.tool_usage_bias, 4),
            "generation": self.generation,
            "fitness": round(self.fitness, 4),
            "parent_ids": self.parent_ids,
        }

    @staticmethod
    def from_dict(d: dict) -> "Genome":
        return Genome(
            agent_id=d["agent_id"],
            planning_depth=float(d.get("planning_depth", 2.5)),
            reasoning_steps=int(d.get("reasoning_steps", 4)),
            cooperation_bias=float(d.get("cooperation_bias", 0.5)),
            exploration_rate=float(d.get("exploration_rate", 0.5)),
            verification_level=float(d.get("verification_level", 0.5)),
            risk_bias=float(d.get("risk_bias", 0.5)),
            tool_usage_bias=float(d.get("tool_usage_bias", 0.5)),
            generation=int(d.get("generation", 0)),
            fitness=float(d.get("fitness", 0.0)),
            parent_ids=list(d.get("parent_ids", [])),
        )

    @staticmethod
    def random_genome(generation: int = 0) -> "Genome":
        return Genome(
            agent_id=uuid.uuid4().hex[:8],
            planning_depth=round(random.uniform(1.0, 5.0), 3),
            reasoning_steps=random.randint(2, 7),
            cooperation_bias=random.random(),
            exploration_rate=random.random(),
            verification_level=random.random(),
            risk_bias=random.random(),
            tool_usage_bias=random.random(),
            generation=generation,
        )

    def mutate(self, rate: float = 0.4) -> "Genome":
        rng = random
        pd = self.planning_depth
        rs = self.reasoning_steps
        cb = self.cooperation_bias
        er = self.exploration_rate
        vl = self.verification_level
        rb = self.risk_bias
        tub = self.tool_usage_bias

        def clamp(v, lo, hi):
            return max(lo, min(hi, v))

        if rng.random() < rate:
            pd = clamp(pd + rng.gauss(0, 0.6), 1.0, 5.0)
        if rng.random() < rate:
            rs = clamp(rs + rng.choice([-1, 0, 1]), 2, 7)
        if rng.random() < rate:
            cb = clamp(cb + rng.gauss(0, 0.15), 0.0, 1.0)
        if rng.random() < rate:
            er = clamp(er + rng.gauss(0, 0.15), 0.0, 1.0)
        if rng.random() < rate:
            vl = clamp(vl + rng.gauss(0, 0.15), 0.0, 1.0)
        if rng.random() < rate:
            rb = clamp(rb + rng.gauss(0, 0.15), 0.0, 1.0)
        if rng.random() < rate:
            tub = clamp(tub + rng.gauss(0, 0.15), 0.0, 1.0)

        return Genome(
            agent_id=uuid.uuid4().hex[:8],
            planning_depth=round(pd, 3),
            reasoning_steps=int(rs),
            cooperation_bias=round(cb, 4),
            exploration_rate=round(er, 4),
            verification_level=round(vl, 4),
            risk_bias=round(rb, 4),
            tool_usage_bias=round(tub, 4),
            generation=self.generation + 1,
            parent_ids=[self.agent_id],
        )


def crossover(a: Genome, b: Genome) -> Genome:
    """Uniform crossover — each trait independently from either parent."""
    pick = random.choice
    return Genome(
        agent_id=uuid.uuid4().hex[:8],
        planning_depth=round(pick([a.planning_depth, b.planning_depth]), 3),
        reasoning_steps=pick([a.reasoning_steps, b.reasoning_steps]),
        cooperation_bias=round(pick([a.cooperation_bias, b.cooperation_bias]), 4),
        exploration_rate=round(pick([a.exploration_rate, b.exploration_rate]), 4),
        verification_level=round(pick([a.verification_level, b.verification_level]), 4),
        risk_bias=round(pick([a.risk_bias, b.risk_bias]), 4),
        tool_usage_bias=round(pick([a.tool_usage_bias, b.tool_usage_bias]), 4),
        generation=max(a.generation, b.generation) + 1,
        parent_ids=[a.agent_id, b.agent_id],
    )


def evolve_population(population: list[Genome], pop_size: int = 12) -> list[Genome]:
    ranked = sorted(population, key=lambda g: g.fitness, reverse=True)
    n_elite = max(2, round(len(ranked) * 0.30))
    elites = ranked[:n_elite]

    next_gen: list[Genome] = []

    # Elites carry over
    for e in elites:
        carried = Genome.from_dict(e.to_dict())
        carried.generation += 1
        next_gen.append(carried)

    # Crossover children
    n_cross = (pop_size - n_elite) // 2
    for _ in range(n_cross):
        a, b = random.sample(elites, min(2, len(elites)))
        next_gen.append(crossover(a, b))

    # Mutated copies fill remainder
    while len(next_gen) < pop_size:
        next_gen.append(random.choice(elites).mutate())

    return next_gen[:pop_size]


def initialize_population(pop_size: int = 12) -> list[Genome]:
    return [Genome.random_genome(generation=1) for _ in range(pop_size)]
