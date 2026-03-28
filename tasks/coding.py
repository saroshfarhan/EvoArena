"""
Coding Task — reference implementation.
The agent must write sum_evens(lst) that passes 5 test cases.

Tool flow (typical):
  write_code → run_tests → [write_code → run_tests if failed] → finish

Genome influence:
  verification_level  high → agent always calls run_tests before finish
  exploration_rate    high → agent tries creative solutions
  planning_depth      high → agent designs solution before writing
"""

# This mirrors CodingEnv in sandbox_worker.py — kept here for local testing.
from sandbox_worker import CodingEnv

__all__ = ["CodingEnv"]

if __name__ == "__main__":
    env = CodingEnv()
    print(env.get_initial_message(5))
    print("\n-- write correct code --")
    print(env.execute_tool("write_code", {"code": "def sum_evens(lst):\n    return sum(x for x in lst if x % 2 == 0)"}))
    print(env.execute_tool("run_tests", {}))
    print(f"Score: {env.score():.2f}")
