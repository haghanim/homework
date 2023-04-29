"""


"""

import gymnasium as gym
from customagent import Agent

SHOW_ANIMATIONS = True

# https://gymnasium.farama.org/environments/box2d/lunar_lander/
# observation = state of spacecraft - position and velocity
    # The state is an 8-dimensional vector: 
    # 1:2, the coordinates of the lander in x & y, 
    # 3:4 linear velocities in x & y, 
    # 5 angle, 
    # 6:its angular velocity, 
    # 7:8 two booleans that represent whether each leg is in contact with the ground or not.
# action_space = all possible moves spacecraft can make 
    # 0: do nothing
    # 1: fire left orientation engine
    # 2: fire main engine
    # 3: fire right orientation engine

env = gym.make("LunarLander-v2", render_mode="human" if SHOW_ANIMATIONS else "none")
observation, info = env.reset(seed=42)

agent = Agent(
    action_space=env.action_space,
    observation_space=env.observation_space,
)

total_reward = 0
last_n_rewards = []
for _ in range(100000):
    action = agent.act(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    agent.learn(observation, reward, terminated, truncated)
    total_reward += reward

    if terminated or truncated:
        observation, info = env.reset()
        last_n_rewards.append(total_reward)
        n = min(30, len(last_n_rewards))
        avg = sum(last_n_rewards[-n:]) / n
        improvement_emoji = "🔥" if (total_reward > avg) else "😢"
        print(
            f"{improvement_emoji} Finished with reward {int(total_reward)}.\tAverage of last {n}: {int(avg)}"
        )
        if avg > 0:
            print("🎉 Nice work! You're ready to submit the leaderboard! 🎉")
        total_reward = 0

env.close()
