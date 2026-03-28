# EvoArena

Evolutionary multi-agent problem solving powered by Claude AI and [Daytona](https://www.daytona.io/) sandboxes.

Each agent carries a **genome** — a set of personality traits — that shapes how Claude (the AI model) reasons and makes decisions. Agents compete across three tasks inside isolated Daytona sandboxes. The fittest survive, reproduce, and mutate. Over generations, the population evolves toward strategies that solve problems better.

Think of it as **natural selection for AI agents**: instead of DNA, agents inherit numerical traits. Instead of survival in nature, they survive by scoring well on tasks.

---

## The Big Picture

```
┌─────────────────────────────────────────────────────────────┐
│                      EVOARENA LOOP                          │
│                                                             │
│  Random genomes → Claude agents → Daytona sandboxes         │
│       ↓                                                     │
│  Each agent solves 3 tasks (Claude decides every move)      │
│       ↓                                                     │
│  Fitness score computed (accuracy + efficiency + strategy)  │
│       ↓                                                     │
│  Top agents selected → crossover + mutation → next gen      │
│       ↓                                                     │
│  Repeat for N generations                                   │
└─────────────────────────────────────────────────────────────┘
```

**Key insight:** The genome doesn't directly solve the task — it changes the *system prompt* sent to Claude. A high `cooperation_bias` makes Claude more inclined to cooperate in a game. A high `planning_depth` makes Claude reason more carefully before acting. Evolution finds which combination of traits produces the best Claude behavior.

---

## How the Tweaking Actually Works

This is the part that makes EvoArena more than a typical evolutionary algorithm — the genome numbers don't directly control logic. They control **what Claude is told to do**, and Claude is an LLM making real decisions. Here is exactly what happens at each stage.

### Stage 1 — Numbers become words (genome → system prompt)

The raw numbers are thresholded into plain English instructions that get injected into Claude's system prompt before each task. Claude never sees the number `4.2` — it sees the sentence it maps to:

```
planning_depth = 4.2  →  "DEEP PLANNER: Think through every step before acting.
                           Map the full solution before your first tool call."

planning_depth = 1.3  →  "FAST ACTOR: Act immediately.
                           Skip lengthy planning — first instinct, then go."

cooperation_bias = 0.9  →  "COOPERATIVE: In multi-agent or game scenarios,
                             prefer cooperation and mutual benefit."

cooperation_bias = 0.1  →  "COMPETITIVE: Prioritise your own score."

verification_level = 0.8  →  "HIGH VERIFICATION: Always verify your work with a
                               test/evaluate tool before finishing. Double-check."
```

The `reasoning_steps` trait works differently — it is a **hard cap** on how many tool calls Claude is allowed per task. An agent with `reasoning_steps=3` gets cut off after 3 moves regardless of what it wants to do.

### Stage 2 — Claude plays the tasks

Claude receives the system prompt + task description and decides which tool to call on every turn. The behavioral differences are real and observable:

- A high `cooperation_bias` agent cooperates every round in the Prisoner's Dilemma, scoring 3 pts/round vs Tit-for-Tat
- A high `verification_level` agent calls `run_tests` before `finish` in the coding task, catching bugs
- A low `planning_depth` agent jumps straight to proposing a knapsack solution without reasoning about item ratios, often missing the optimum

### Stage 3 — Scores drive evolution (fitness → selection → genetic operators)

After all tasks complete, a fitness score is computed from the results:

```
fitness = 0.40 × mean_accuracy      ← did it solve the problems correctly?
        + 0.20 × mean_efficiency    ← did it do it in few steps?
        + 0.20 × robustness         ← was it consistent across all 3 tasks?
        + 0.20 × strategic_score    ← did it cooperate? did it verify its code?
```

Then three genetic operators physically change the numbers for the next generation:

**Selection** — only the top 30% of agents survive to reproduce. Agents with poor fitness are discarded entirely. Their numbers never appear in the next generation.

**Crossover** — two surviving parents produce a child by randomly inheriting each trait from one parent:
```
parent A:  cooperation_bias=0.9,  planning_depth=4.2,  exploration_rate=0.2
parent B:  cooperation_bias=0.3,  planning_depth=2.1,  exploration_rate=0.7
child:     cooperation_bias=0.9,  planning_depth=2.1,  exploration_rate=0.2
                              ↑ from A              ↑ from B             ↑ from A
```

**Mutation** — each trait has a 40% chance of a small random Gaussian nudge (±0.15 on average for 0–1 traits):
```
cooperation_bias = 0.85  →  0.85 + gauss(0, 0.15)  →  0.91  (small drift up)
cooperation_bias = 0.85  →  0.85 + gauss(0, 0.15)  →  0.72  (small drift down)
```

This introduces variation so the population doesn't get stuck — children can explore slightly different instructions than their parents.

### The feedback loop in one sentence

> Numbers shape what Claude is told → Claude's decisions produce scores → scores select which numbers survive → surviving numbers are recombined and nudged → the process repeats

Evolution never touches Claude's model weights. It only changes the **instructions Claude receives**. Selection pressure pushes those instructions toward whatever combination makes Claude perform best across all three tasks.

---

## How it works

### 1. The Genome

Every agent is defined by 7 numerical traits. These traits are injected into Claude's system prompt as behavioral instructions:

| Trait | Range | What it tells Claude |
|---|---|---|
| `planning_depth` | 1.0 – 5.0 | How carefully to analyse the problem before acting. High = reason step by step. Low = act fast. |
| `reasoning_steps` | 2 – 7 | How many tool calls Claude is allowed. More = thorough but expensive. |
| `cooperation_bias` | 0 – 1 | In game theory, how strongly to prefer cooperation over defection. |
| `exploration_rate` | 0 – 1 | How willing Claude is to try unconventional solutions. High = creative but risky. |
| `verification_level` | 0 – 1 | How often Claude checks its own work before finishing. |
| `risk_bias` | 0 – 1 | Willingness to attempt aggressive solutions that might fail but score high. |
| `tool_usage_bias` | 0 – 1 | How proactively Claude calls available tools vs reasoning internally. |

### 2. The Tasks

Each sandbox runs all three tasks. Claude controls every action via tool calls.

**Coding Task**
- Problem: write a Python function `sum_evens(lst)` that passes 5 test cases
- Claude's tools: `write_code`, `run_tests`, `inspect_errors`, `finish`
- What makes an agent good here: `verification_level` (runs tests before finishing), `planning_depth` (reasons about edge cases)
- Score: fraction of test cases passed × step efficiency

**Planning Task (0/1 Knapsack)**
- Problem: pick items from a list to maximise total value without exceeding weight capacity 10. Optimal value = 32.
- Claude's tools: `propose_solution`, `evaluate_solution`, `modify_solution`, `finish`
- What makes an agent good here: `planning_depth` (reasons about value/weight ratios), `verification_level` (evaluates before finishing)
- Score: `best_value_achieved / 32`

**Game Theory Task (Iterated Prisoner's Dilemma)**
- Problem: 8 rounds against a Tit-for-Tat opponent. Cooperate → 3 pts each. You defect, they cooperate → 5 pts once, then they punish you every round.
- Claude's tools: `play_move` (cooperate/defect), `reflect`, `finish`
- What makes an agent good here: `cooperation_bias` (cooperating consistently = 3 pts/round = 24 total), `planning_depth` (understands TfT dynamics)
- Score: `total_points / 24`

### 3. Fitness Formula

```
fitness = 0.40 × mean_accuracy
        + 0.20 × mean_efficiency
        + 0.20 × robustness
        + 0.20 × strategic_score
```

- **mean_accuracy** — average correctness across all 3 tasks (0–1)
- **mean_efficiency** — how few steps were used relative to the budget. Using 3 steps when given 7 is more efficient than using all 7.
- **robustness** — reward for being consistently good across all tasks. An agent that scores 0.9/0.9/0.9 beats one that scores 1.0/1.0/0.0 even if the average is similar.
- **strategic_score** — bonus for strategic behavior: high game theory cooperation rate + verifying code before finishing

### 4. Evolution

```
Generation 1: 12 completely random agents (random genome values)
                 ↓
              All run in parallel Daytona sandboxes
                 ↓
              Fitness computed and ranked
                 ↓
Generation 2: Top 30% (elite) survive unchanged
              Remaining slots filled by:
                - Crossover: two parents randomly swap traits
                - Mutation: small random shifts to trait values
                 ↓
              Repeat
                 ↓
Generation 3: Population has converged toward fitter strategies
```

The evolution process mirrors biological natural selection: beneficial trait combinations are preserved and propagated, while poor strategies die out.

---

## What Good Results Look Like

### Fitness trajectory (improving over generations)
```
Gen 1: ████████████░░░░░░░░░░░░░░░░░░  0.42   ← random agents, highly variable
Gen 2: ████████████████████░░░░░░░░░░  0.65   ← selection filtered out worst
Gen 3: ████████████████████████████░░  0.78   ← crossover found good combinations
```

A **rising fitness trajectory** across generations confirms evolution is working. If Gen 3 fitness equals Gen 1, the population likely converged too early (low mutation) or the tasks are too easy to differentiate agents.

### Leaderboard interpretation

| Column | What to look for |
|---|---|
| `Fitness` | Overall score 0–1. Above 0.7 is strong. Above 0.85 is excellent. |
| `Acc` | Mean accuracy. 1.0 = solved all tasks correctly. |
| `Eff` | Mean efficiency. 1.0 = solved everything in minimal steps. |
| `Robust` | Consistency. Close to 1.0 = performed well on every task, not just one. |
| `Strat` | Strategic score. High = cooperated in game theory + verified code. |
| `Lineage` | `seed` = original random genome. `a3f2c1+b7e9a0` = crossover of those two parents. |

### Genome profile interpretation

When you see the bar chart of a top agent's genome, here is what a typically good genome looks like:

```
planning_depth      [████████████████░░]  3.8   ← high: careful reasoning
reasoning_steps     [████████████░░░░░░]  4     ← moderate: thorough but not excessive
cooperation_bias    [████████████████░░]  0.85  ← high: cooperates with TfT opponent
exploration_rate    [████░░░░░░░░░░░░░░]  0.2   ← low: stays focused, not random
verification_level  [████████████████░░]  0.8   ← high: checks work before finishing
risk_bias           [████████░░░░░░░░░░]  0.4   ← moderate: calculated risks
tool_usage_bias     [████████████░░░░░░]  0.6   ← moderate: uses tools when needed
```

**Red flags in a genome** — traits that tend to hurt performance:
- `exploration_rate > 0.7`: too random, hurts game theory and planning
- `cooperation_bias < 0.3`: defects in prisoner's dilemma, loses to Tit-for-Tat
- `verification_level < 0.2`: never checks work, submits wrong answers
- `planning_depth < 1.5`: acts impulsively, misses optimal solutions

### Task breakdown interpretation

```
coding       acc=1.000  steps=3/4   tests=5/5   ← perfect: all tests pass, used 3 of 4 steps
planning     acc=0.875  steps=4/4   value=28/32  ← near-optimal: 28/32 value found
game_theory  acc=1.000  steps=11/11 coop=100%  score=24/24  ← perfect: cooperated every round
```

- **coding acc=1.0**: Claude wrote a correct function on the first try
- **planning value=28/32**: found 87.5% of the optimal knapsack value — very good, not perfect
- **game_theory coop=100%**: Claude cooperated every round, scoring maximum 24 points vs TfT

---

## Streamlit Dashboard

```bash
uv run streamlit run app.py
```

The dashboard has 5 tabs:

| Tab | What it shows |
|---|---|
| **Overview** | Fitness trajectory across generations + population stats |
| **Leaderboard** | Ranked table of all agents with fitness breakdown |
| **Genome Analysis** | Scatter plots showing which trait values correlate with high fitness |
| **Lineage** | Which agents are parents of which, tracking evolutionary heritage |
| **Best Agent** | Deep dive into the highest-scoring genome and its task performance |

**What to look for in Genome Analysis:** if you see a clear upward trend between `cooperation_bias` and fitness, it confirms cooperation is a winning strategy. A flat or noisy scatter means that trait doesn't strongly predict success.

---

## Project Structure

```
.
├── main.py              # Evolution loop — orchestrates all generations
├── genome.py            # Genome dataclass, mutation, crossover, selection
├── sandbox_worker.py    # Self-contained worker that runs INSIDE each Daytona sandbox
├── orchestrator.py      # Launches parallel sandboxes and collects results
├── evaluator.py         # Fitness computation from task results
├── agent_runtime.py     # Local test runner (no Daytona needed)
├── app.py               # Streamlit dashboard
├── tasks/
│   ├── coding.py        # Coding task reference implementation
│   ├── planning.py      # Planning (knapsack) reference implementation
│   └── game_theory.py   # Game theory (prisoner's dilemma) reference implementation
└── results/             # Per-generation JSON + final_results.json + best_agent.json
```

---

## Setup

**Prerequisites:** Python 3.11+, [uv](https://github.com/astral-sh/uv), a [Daytona](https://app.daytona.io/) account, an [Anthropic](https://console.anthropic.com/) API key.

```bash
# 1. Clone and install
git clone <repo>
cd daytona_hack
uv sync

# 2. Add API keys to .env
echo "DAYTONA_API_KEY=your_daytona_key" >> .env
echo "ANTHROPIC_API_KEY=your_anthropic_key" >> .env

# 3. Run the evolution loop
uv run main.py

# 4. (Optional) View results in the dashboard
uv run streamlit run app.py
```

Get your Daytona key from [app.daytona.io](https://app.daytona.io/).
Get your Anthropic key from [console.anthropic.com](https://console.anthropic.com/).

---

## Test a Single Agent Locally

You can run one agent locally without Daytona (useful for debugging):

```bash
uv run agent_runtime.py
# or with a custom genome:
uv run agent_runtime.py --genome '{"agent_id":"test","planning_depth":4.0,"reasoning_steps":5,"cooperation_bias":0.9,"exploration_rate":0.2,"verification_level":0.8,"risk_bias":0.3,"tool_usage_bias":0.6}'
```

---

## Configuration

Edit the constants at the top of `main.py`:

| Constant | Default | Description |
|---|---|---|
| `POPULATION_SIZE` | `12` | Agents per generation. More = better exploration, higher cost. |
| `GENERATIONS` | `3` | Evolution cycles. More = more convergence, higher cost. |
| `MAX_CONCURRENCY` | `4` | Max parallel Daytona sandboxes (free-tier limit). |

---

## Results Files

After each run, `results/` contains:

| File | Contents |
|---|---|
| `generation_N.json` | Full leaderboard, genomes, and fitness breakdown for generation N |
| `final_results.json` | Complete run history, fitness trajectory, and config |
| `best_agent.json` | Genome of the highest-scoring agent across all generations |
| `lineage.json` | Parent-child relationships across all generations |
