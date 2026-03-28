"""
Game Theory Task — reference implementation.
Iterated Prisoner's Dilemma: 8 rounds vs Tit-for-Tat.

Tool flow:
  play_move × 8 → finish
  (optionally: reflect between rounds)

Genome influence:
  cooperation_bias  high → LLM instructed to cooperate → better vs TfT (3pts/round)
  planning_depth    high → LLM understands TfT dynamics (defect = opponent defects next)
  exploration_rate  high → random experimentation → hurts against TfT
"""

from sandbox_worker import GameTheoryEnv

__all__ = ["GameTheoryEnv"]

if __name__ == "__main__":
    env = GameTheoryEnv()
    print(env.get_initial_message(11))
    print("\n-- play full cooperative game --")
    for _ in range(8):
        print(env.execute_tool("play_move", {"action": "cooperate"}))
    print(env.execute_tool("finish", {}))
    print(f"Score: {env.score():.4f}  coop_rate={env.extra_stats()['cooperation_rate']}")
