"""Balance an inverted pendulum with reinforcement learning.

Physics: MuJoCo (simulates gravity, joints, and the pole falling over).
Environment interface: Gymnasium (reset / step / reward).
Learning algorithm: PPO from Stable-Baselines3.

Run:  python demo_balance.py
"""

from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO

ENV_NAME = "InvertedPendulum-v5"
TRAIN_STEPS = 100_000
MODEL_PATH = Path(__file__).parent / "models" / "pendulum_ppo"


def train() -> PPO:
    """Train a PPO agent and save it to models/pendulum_ppo.zip."""
    env = gym.make(ENV_NAME)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=TRAIN_STEPS)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    model.save(MODEL_PATH)
    env.close()
    return model


def show(model: PPO, episodes: int = 3) -> None:
    """Run the trained agent in the MuJoCo viewer window."""
    env = gym.make(ENV_NAME, render_mode="human")
    for episode in range(episodes):
        obs, _ = env.reset(seed=episode)
        steps = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            steps += 1
            done = terminated or truncated
        print(f"Episode {episode + 1}: balanced for {steps} steps")
    env.close()


if __name__ == "__main__":
    if MODEL_PATH.with_suffix(".zip").exists():
        print(f"Loading saved model from {MODEL_PATH}.zip")
        trained = PPO.load(MODEL_PATH)
    else:
        print("No saved model found. Training now (a few minutes on a CPU)...")
        trained = train()
    show(trained)
