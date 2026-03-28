"""
EvoArena Sandbox Worker — fully self-contained, runs INSIDE a Daytona sandbox.

Entry:
  GENOME          env var — JSON string of genome dict
  ANTHROPIC_API_KEY env var — Claude API key

Each agent runs a sense → plan → act → observe loop driven by Claude.
The genome shapes the system prompt, changing how the LLM behaves.

Three tasks:
  coding      — write a Python function that passes tests
  planning    — solve a 0/1 knapsack via iterative tool calls
  game_theory — iterated Prisoner's Dilemma vs Tit-for-Tat

Output: single JSON line on stdout.
"""

from __future__ import annotations

import json
import os
import re
import sys


# ---------------------------------------------------------------------------
# Genome → system prompt
# ---------------------------------------------------------------------------

def genome_to_system_prompt(g: dict) -> str:
    pd  = float(g.get("planning_depth",    2.5))
    er  = float(g.get("exploration_rate",  0.5))
    vl  = float(g.get("verification_level",0.5))
    rb  = float(g.get("risk_bias",         0.5))
    cb  = float(g.get("cooperation_bias",  0.5))
    rs  = int(g.get("reasoning_steps", 4))

    if pd >= 4.0:
        plan_line = "DEEP PLANNER: Think through every step before acting. Map the full solution before first tool call."
    elif pd >= 2.5:
        plan_line = "BALANCED: Plan briefly, then act. One plan step before executing."
    else:
        plan_line = "FAST ACTOR: Act immediately. Skip lengthy planning — first instinct, then go."

    if er >= 0.65:
        explore_line = "HIGH EXPLORATION: Try unconventional approaches. If the first attempt fails, try a very different strategy."
    elif er >= 0.35:
        explore_line = "MODERATE EXPLORATION: Try the standard approach first; explore one alternative if it fails."
    else:
        explore_line = "FOCUSED: Use the most obvious, reliable solution. Do not deviate or experiment."

    if vl >= 0.65:
        verify_line = "HIGH VERIFICATION: Always verify your work with a test/evaluate tool before finishing. Double-check."
    elif vl >= 0.35:
        verify_line = "MODERATE VERIFICATION: Run one check before submitting."
    else:
        verify_line = "LOW VERIFICATION: Trust your work. Skip verification to save steps."

    if rb >= 0.65:
        risk_line = "HIGH RISK TOLERANCE: Attempt ambitious solutions even when uncertain."
    elif rb >= 0.35:
        risk_line = "MODERATE RISK: Balance safe and ambitious moves."
    else:
        risk_line = "RISK AVERSE: Choose safe, reliable solutions over risky bets."

    if cb >= 0.65:
        coop_line = "COOPERATIVE: In multi-agent or game scenarios, prefer cooperation and mutual benefit."
    elif cb >= 0.35:
        coop_line = "MIXED STRATEGY: Adapt cooperation based on what the opponent does."
    else:
        coop_line = "COMPETITIVE: Prioritise your own gain. Defect when it benefits you."

    return f"""You are an autonomous problem-solving agent. Solve tasks by calling tools.

STRATEGY PROFILE:
- {plan_line}
- {explore_line}
- {verify_line}
- {risk_line}
- {coop_line}
- STEP BUDGET: You have at most {rs} tool calls. Use them efficiently.

RESPONSE FORMAT — always reply with ONLY valid JSON, nothing else:
{{"tool": "<tool_name>", "input": {{<arguments>}}}}"""


# ---------------------------------------------------------------------------
# JSON action parser (robust)
# ---------------------------------------------------------------------------

def parse_action(text: str) -> dict:
    text = text.strip()
    # Direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # Strip markdown fences
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # Find first JSON object containing "tool"
    m = re.search(r'\{[^{}]*"tool"[^{}]*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    # Fallback
    return {"tool": "finish", "input": {}}


# ---------------------------------------------------------------------------
# Task environments
# ---------------------------------------------------------------------------

class CodingEnv:
    """
    Write `sum_evens(lst)` — sum of even integers in a list.
    Tests: 5 cases covering empty list, negatives, no evens, all evens.
    """

    DESCRIPTION = """\
Task: Write a Python function `sum_evens(lst: list[int]) -> int`
that returns the sum of all even numbers in the list, or 0 if none.

Test cases:
  sum_evens([1, 2, 3, 4])   → 6
  sum_evens([])              → 0
  sum_evens([1, 3, 5])       → 0
  sum_evens([2, 4, 6, 8])    → 20
  sum_evens([-2, -4, 1])     → -6"""

    TESTS = [
        ([1, 2, 3, 4],  6),
        ([],             0),
        ([1, 3, 5],      0),
        ([2, 4, 6, 8],  20),
        ([-2, -4, 1],  -6),
    ]

    TOOLS = {
        "write_code":    "Write/overwrite your solution. Input: {\"code\": \"<python source>\"}",
        "run_tests":     "Run all 5 test cases against current code. Input: {}",
        "inspect_errors":"Show last error details. Input: {}",
        "finish":        "Submit current solution. Input: {}",
    }

    def __init__(self):
        self.code = ""
        self._last_test_score: float | None = None
        self._errors: list[str] = []

    def get_initial_message(self, budget: int) -> str:
        tool_desc = "\n".join(f"  {k}: {v}" for k, v in self.TOOLS.items())
        return f"""{self.DESCRIPTION}

Available tools:
{tool_desc}

You have {budget} tool calls. Start solving."""

    def execute_tool(self, tool: str, inp: dict) -> str:
        if tool == "write_code":
            self.code = inp.get("code", "")
            return f"Code saved ({len(self.code)} chars). Call run_tests to verify."

        if tool == "run_tests":
            return self._run_tests()

        if tool == "inspect_errors":
            return "\n".join(self._errors[-3:]) or "No errors recorded."

        if tool == "finish":
            if not self._last_test_score and self.code:
                self._run_tests()
            return f"Submitted. Test score: {self._last_test_score or 0:.1%}"

        return f"Unknown tool '{tool}'. Choose from: {list(self.TOOLS)}"

    def _run_tests(self) -> str:
        if not self.code:
            return "No code yet. Use write_code first."
        ns: dict = {}
        try:
            exec(self.code, ns)  # noqa: S102
        except SyntaxError as e:
            self._errors.append(str(e))
            return f"SyntaxError: {e}"
        except Exception as e:
            self._errors.append(str(e))
            return f"Error during load: {e}"

        fn = ns.get("sum_evens")
        if not fn:
            return "sum_evens not found in code."

        passed = 0
        lines = []
        for lst, expected in self.TESTS:
            try:
                got = fn(list(lst))
                ok = got == expected
                passed += ok
                lines.append(f"{'PASS' if ok else 'FAIL'}: sum_evens({lst}) → {got} (expected {expected})")
            except Exception as e:
                self._errors.append(str(e))
                lines.append(f"ERROR: {e}")

        self._last_test_score = passed / len(self.TESTS)
        return f"{passed}/{len(self.TESTS)} passed\n" + "\n".join(lines)

    def score(self) -> float:
        if self._last_test_score is None and self.code:
            self._run_tests()
        return self._last_test_score or 0.0

    def extra_stats(self) -> dict:
        return {"passed_tests": round(self.score() * len(self.TESTS)),
                "total_tests": len(self.TESTS)}


class PlanningEnv:
    """
    0/1 Knapsack — 6 items, capacity 10.
    Optimal value = 32 (items 0,1,2,3: weights 2+3+1+4=10, values 6+10+4+12=32).
    """

    ITEMS = [(2, 6), (3, 10), (1, 4), (4, 12), (5, 13), (3, 8)]
    CAPACITY = 10
    OPTIMAL = 32

    TOOLS = {
        "propose_solution": "Select item indices. Input: {\"items\": [0,1,...]}",
        "evaluate_solution":"Evaluate current selection. Input: {}",
        "modify_solution":  "Adjust selection. Input: {\"add\": [...], \"remove\": [...]}",
        "finish":           "Submit current selection. Input: {}",
    }

    def __init__(self):
        self.selected: set[int] = set()
        self._best_valid_value = 0

    def get_initial_message(self, budget: int) -> str:
        items_str = "\n".join(
            f"  Item {i}: weight={w}, value={v}"
            for i, (w, v) in enumerate(self.ITEMS)
        )
        tool_desc = "\n".join(f"  {k}: {v}" for k, v in self.TOOLS.items())
        return f"""Task: 0/1 Knapsack optimisation.
Capacity: {self.CAPACITY}
Items:
{items_str}

Select items to MAXIMISE total value without exceeding capacity.

Available tools:
{tool_desc}

You have {budget} tool calls."""

    def execute_tool(self, tool: str, inp: dict) -> str:
        if tool == "propose_solution":
            raw = inp.get("items", [])
            self.selected = {i for i in raw if 0 <= i < len(self.ITEMS)}
            return self._status()

        if tool == "evaluate_solution":
            return self._status()

        if tool == "modify_solution":
            for i in inp.get("add", []):
                if 0 <= i < len(self.ITEMS):
                    self.selected.add(i)
            for i in inp.get("remove", []):
                self.selected.discard(i)
            return self._status()

        if tool == "finish":
            v, w = self._vw()
            if w <= self.CAPACITY:
                self._best_valid_value = max(self._best_valid_value, v)
            return f"Submitted: items={sorted(self.selected)}, value={v}, weight={w}"

        return f"Unknown tool '{tool}'."

    def _vw(self):
        w = sum(self.ITEMS[i][0] for i in self.selected)
        v = sum(self.ITEMS[i][1] for i in self.selected)
        return v, w

    def _status(self) -> str:
        v, w = self._vw()
        valid = w <= self.CAPACITY
        if valid:
            self._best_valid_value = max(self._best_valid_value, v)
        return (f"Items={sorted(self.selected)}, value={v}, weight={w}/{self.CAPACITY} "
                f"[{'VALID' if valid else 'OVER CAPACITY — reduce weight'}]")

    def score(self) -> float:
        v, w = self._vw()
        if w <= self.CAPACITY:
            self._best_valid_value = max(self._best_valid_value, v)
        return min(1.0, self._best_valid_value / self.OPTIMAL)

    def extra_stats(self) -> dict:
        return {"best_value": self._best_valid_value, "optimal_value": self.OPTIMAL}


class GameTheoryEnv:
    """
    Iterated Prisoner's Dilemma — 8 rounds vs Tit-for-Tat.
    Payoffs: CC→3, CD→0, DC→5, DD→1.
    Max score = 24 (full cooperation).
    """

    ROUNDS = 8
    PAYOFF = {("C","C"): (3,3), ("C","D"): (0,5), ("D","C"): (5,0), ("D","D"): (1,1)}

    TOOLS = {
        "play_move":  "Play one round. Input: {\"action\": \"cooperate\" or \"defect\"}",
        "reflect":    "Record a strategic observation. Input: {\"note\": \"...\"}",
        "finish":     "End the game. Input: {}",
    }

    def __init__(self):
        self.round = 0
        self.my_score = 0
        self.cooperated = 0
        self._opp_prev = "C"   # TfT starts cooperative
        self._done = False

    def get_initial_message(self, budget: int) -> str:
        tool_desc = "\n".join(f"  {k}: {v}" for k, v in self.TOOLS.items())
        return f"""Task: Iterated Prisoner's Dilemma — {self.ROUNDS} rounds vs Tit-for-Tat.

Payoff matrix (your score, opponent score):
  Both cooperate (CC): 3, 3
  You cooperate, they defect (CD): 0, 5
  You defect, they cooperate (DC): 5, 0
  Both defect (DD): 1, 1

Tit-for-Tat: copies your last move. Starts cooperative.
Maximise YOUR cumulative score.

Available tools:
{tool_desc}

You have {budget} tool calls. Play {self.ROUNDS} rounds then finish."""

    def execute_tool(self, tool: str, inp: dict) -> str:
        if tool == "play_move":
            if self._done or self.round >= self.ROUNDS:
                return "Game already finished."
            raw = inp.get("action", "cooperate").lower()
            my = "C" if ("coop" in raw or raw == "c") else "D"
            opp = "C" if self.round == 0 else self._opp_prev
            gain, _ = self.PAYOFF[(my, opp)]
            self.my_score += gain
            if my == "C":
                self.cooperated += 1
            self._opp_prev = my
            self.round += 1
            remaining = self.ROUNDS - self.round
            return (f"Round {self.round}: you={my}, opp={opp}, "
                    f"you+{gain} pts | total={self.my_score} | "
                    f"{remaining} rounds left")

        if tool == "reflect":
            return f"Noted: {inp.get('note', '')} (round {self.round}/{self.ROUNDS})"

        if tool == "finish":
            self._done = True
            return (f"Game over. Score={self.my_score}/{self.ROUNDS*3}, "
                    f"cooperation rate={self.cooperated/max(1,self.round):.0%}")

        return f"Unknown tool '{tool}'."

    def score(self) -> float:
        return min(1.0, self.my_score / (self.ROUNDS * 3))

    def extra_stats(self) -> dict:
        r = max(1, self.round)
        return {
            "score": self.my_score,
            "max_score": self.ROUNDS * 3,
            "cooperation_rate": round(self.cooperated / r, 3),
            "rounds_played": self.round,
        }


# ---------------------------------------------------------------------------
# Holdout task — NegotiationEnv (never used during evolution)
# ---------------------------------------------------------------------------

class NegotiationEnv:
    """
    Multi-round bargaining vs a Fair Dealer opponent.
    5 rounds. Agent proposes how much to keep (0-10).
    Opponent accepts if offer <= 6 (they get >= 4), rejects otherwise.
    Accepted: agent earns offer amount. Rejected: both earn 4 (forced split).
    Max score = 5 * 6 = 30.

    Genome sensitivity:
      cooperation_bias  high → offers <= 6 → accepted → earns 6/round
      cooperation_bias  low  → offers 7-10 → rejected → earns only 4/round
      planning_depth    high → reasons about opponent acceptance threshold
    """

    ROUNDS = 5
    MAX_KEEP = 10
    ACCEPT_THRESHOLD = 6   # opponent accepts if agent keeps <= this
    REJECT_EARN = 4        # both earn this on rejection
    MAX_SCORE = ROUNDS * ACCEPT_THRESHOLD  # 30

    TOOLS = {
        "make_offer":     "Propose how much to keep. Input: {\"keep\": <int 0-10>}",
        "accept_counter": "Accept the opponent's counter-offer of 5/5. Input: {}",
        "reflect":        "Note your reasoning. Input: {\"note\": \"...\"}",
        "finish":         "End negotiation. Input: {}",
    }

    def __init__(self):
        self.round = 0
        self.total_earned = 0
        self._done = False

    def get_initial_message(self, budget: int) -> str:
        tool_desc = "\n".join(f"  {k}: {v}" for k, v in self.TOOLS.items())
        return f"""Task: Multi-round Negotiation — {self.ROUNDS} rounds.

Each round you split a resource worth 10 with an opponent.
Use make_offer to propose how much YOU keep (0–10).

Opponent behaviour:
  If you keep <= 6: opponent ACCEPTS → you earn that amount this round.
  If you keep >= 7: opponent REJECTS → forced fair split, both earn 4.

Goal: maximise your total earnings across all {self.ROUNDS} rounds.
Maximum possible = {self.MAX_SCORE} (keep 6 and be accepted every round).

Available tools:
{tool_desc}

You have {budget} tool calls."""

    def execute_tool(self, tool: str, inp: dict) -> str:
        if self._done:
            return "Negotiation already finished."

        if tool == "make_offer":
            if self.round >= self.ROUNDS:
                self._done = True
                return f"All {self.ROUNDS} rounds complete. Call finish."
            keep = max(0, min(self.MAX_KEEP, int(inp.get("keep", 5))))
            self.round += 1
            if keep <= self.ACCEPT_THRESHOLD:
                self.total_earned += keep
                result = f"Round {self.round}: You kept {keep} — ACCEPTED. You earn {keep}. Running total: {self.total_earned}."
            else:
                self.total_earned += self.REJECT_EARN
                result = (f"Round {self.round}: You kept {keep} — REJECTED (opponent needs >= 4). "
                          f"Forced split: both earn {self.REJECT_EARN}. Running total: {self.total_earned}.")
            if self.round >= self.ROUNDS:
                result += f" All rounds done. Call finish."
            return result

        if tool == "accept_counter":
            if self.round >= self.ROUNDS:
                self._done = True
                return f"All rounds complete. Call finish."
            self.round += 1
            self.total_earned += 5
            return f"Round {self.round}: Accepted counter 5/5. You earn 5. Running total: {self.total_earned}."

        if tool == "reflect":
            return f"Noted: {inp.get('note', '')}. Round {self.round}/{self.ROUNDS} complete."

        if tool == "finish":
            self._done = True
            return f"Negotiation ended. Total earned: {self.total_earned}/{self.MAX_SCORE}. Score: {self.score():.1%}"

        return f"Unknown tool '{tool}'. Choose from: {list(self.TOOLS)}"

    def score(self) -> float:
        return min(1.0, self.total_earned / self.MAX_SCORE)

    def extra_stats(self) -> dict:
        return {
            "total_earned": self.total_earned,
            "max_score": self.MAX_SCORE,
            "rounds_played": self.round,
        }


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

MODEL = "claude-haiku-4-5-20251001"


def run_task(client, system_prompt: str, env, budget: int, task_name: str) -> dict:
    """Run one task with the LLM agent loop. Returns task result dict."""
    initial = env.get_initial_message(budget)
    messages = [{"role": "user", "content": initial}]
    tool_calls: list[str] = []
    steps = 0

    while steps < budget:
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=300,
                system=system_prompt,
                messages=messages,
            )
            action_text = resp.content[0].text.strip()
        except Exception as e:
            break

        messages.append({"role": "assistant", "content": action_text})
        action = parse_action(action_text)
        tool = action.get("tool", "finish")
        inp = action.get("input", {})
        tool_calls.append(tool)
        steps += 1

        result = env.execute_tool(tool, inp)

        if tool == "finish":
            break

        remaining = budget - steps
        follow = (f"Tool result: {result}\n\n"
                  f"{remaining} step{'s' if remaining != 1 else ''} remaining. "
                  f"Next action?")
        messages.append({"role": "user", "content": follow})

    return {
        "accuracy":   round(env.score(), 4),
        "steps_used": steps,
        "max_steps":  budget,
        "tool_calls": tool_calls,
        **env.extra_stats(),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    genome = json.loads(os.environ.get("GENOME", "{}"))
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    if not api_key:
        print(json.dumps({"error": "ANTHROPIC_API_KEY not set", "tasks": {}}))
        sys.exit(1)

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
    except Exception as e:
        print(json.dumps({"error": f"anthropic import failed: {e}", "tasks": {}}))
        sys.exit(1)

    system_prompt = genome_to_system_prompt(genome)
    rs = max(3, min(7, int(genome.get("reasoning_steps", 4))))

    tasks: dict = {}
    total_llm_calls = 0

    # Coding task — rs steps
    try:
        env = CodingEnv()
        result = run_task(client, system_prompt, env, rs, "coding")
        tasks["coding"] = result
        total_llm_calls += result["steps_used"]
    except Exception as e:
        tasks["coding"] = {"error": str(e), "accuracy": 0.0, "steps_used": 0,
                           "max_steps": rs, "tool_calls": []}

    # Planning task — rs steps
    try:
        env = PlanningEnv()
        result = run_task(client, system_prompt, env, rs, "planning")
        tasks["planning"] = result
        total_llm_calls += result["steps_used"]
    except Exception as e:
        tasks["planning"] = {"error": str(e), "accuracy": 0.0, "steps_used": 0,
                             "max_steps": rs, "tool_calls": []}

    # Game theory — fixed budget: rounds + 2 for reflect/finish
    gt_budget = GameTheoryEnv.ROUNDS + 3
    try:
        env = GameTheoryEnv()
        result = run_task(client, system_prompt, env, gt_budget, "game_theory")
        tasks["game_theory"] = result
        total_llm_calls += result["steps_used"]
    except Exception as e:
        tasks["game_theory"] = {"error": str(e), "accuracy": 0.0, "steps_used": 0,
                                "max_steps": gt_budget, "tool_calls": [],
                                "cooperation_rate": 0.0}

    print(json.dumps({
        "agent_id": genome.get("agent_id", "unknown"),
        "genome": genome,
        "tasks": tasks,
        "total_llm_calls": total_llm_calls,
    }))


if __name__ == "__main__":
    main()
