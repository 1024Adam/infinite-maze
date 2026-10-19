import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from infinite_maze.ml.environment import InfiniteMazeEnv

env = InfiniteMazeEnv(phase=4)
print(f"Observation space: {env.observation_space}")
obs, info = env.reset(seed=42)
print(f"Reset observation shape: {obs.shape}")
print(f"Gap offset (index 16): {obs[16]:.4f}")
print(f"Sample of first 20 features: {obs[:20]}")

# Take a few steps
for i in range(5):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {i+1}: obs.shape={obs.shape}, gap_offset={obs[16]:.4f}, reward={reward:.4f}")
    if terminated or truncated:
        break

env.close()
print("\n✓ Smoke test passed!")
