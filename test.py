"""

### NOTICE ###
You DO NOT need to upload this file

"""

import argparse
import numpy as np
from environment import Environment
import time
from gymnasium.wrappers.monitoring import video_recorder
from tqdm import tqdm

seed = 11037

def test(agent, env, total_episodes=100, record_video=False):
    rewards = []
    env.seed(seed)

    vid = None  # Initialize vid to None to ensure it's accessible outside the if block
    if record_video:
        vid = video_recorder.VideoRecorder(env=env.env, path="test_vid.mp4")
    start_time = time.time()
    
    truncated = False
    for _ in tqdm(range(total_episodes)):
        if truncated:
            env.close()
            env = Environment('BreakoutNoFrameskip-v4', None, atari_wrapper=True, test=True, render_mode=None)
        episode_reward = 0.0
        truncated = False
        for _ in range(5):  # Run each episode for 5 lives
            state = env.reset()
            agent.init_game_setting()
            terminated = False

            # playing one game (1 life)
            while not terminated and not truncated:
                action = agent.make_action(state, test=True)
                state, reward, terminated, truncated, info = env.step(int(action.cpu().numpy()))
                episode_reward += reward
                if truncated:
                    print("see truncated true!")

                if record_video:
                    vid.capture_frame()

            if truncated:
                break

        rewards.append(episode_reward)

    if record_video:
        vid.close()  # Ensure the video recorder is properly closed

    env.close()

    print('Run %d episodes for 5 lives each' % (total_episodes))
    print('Mean:', np.mean(rewards))
    print('rewards', rewards)
    print('running time', time.time()-start_time)