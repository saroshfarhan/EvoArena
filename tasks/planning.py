"""
Planning Task — reference implementation.
0/1 Knapsack: 6 items, capacity 10, optimal value = 32.

Tool flow (typical):
  propose_solution → evaluate_solution → modify_solution → finish

Genome influence:
  planning_depth      high → agent reasons about item ratios before proposing
  verification_level  high → agent calls evaluate before finish
  risk_bias           high → agent attempts larger item sets (may go over capacity)
"""

from sandbox_worker import PlanningEnv

__all__ = ["PlanningEnv"]

if __name__ == "__main__":
    env = PlanningEnv()
    print(env.get_initial_message(5))
    print("\n-- propose optimal selection --")
    print(env.execute_tool("propose_solution", {"items": [0, 1, 2, 3]}))
    print(env.execute_tool("evaluate_solution", {}))
    print(f"Score: {env.score():.4f}  (optimal={env.OPTIMAL})")
