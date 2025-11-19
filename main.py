"""

### NOTICE ###
You DO NOT need to upload this file

"""
import argparse

from atari_wrapper import FrameStackTrajectoryView
from test import test
from environment import Environment
import time

import matplotlib
import matplotlib.pyplot as plt

def parse():
    parser = argparse.ArgumentParser(description="DS551/CS525 RL Project3")
    parser.add_argument('--env_name', default=None, help='environment name')
    parser.add_argument('--train_dqn', action='store_true', help='whether train DQN')
    parser.add_argument('--train_dqn_again', action='store_true', help='whether train DQN again')
    parser.add_argument('--train_episode', type=int, help='continue training from episode #')
    parser.add_argument('--test_dqn', action='store_true', help='whether test DQN')
    parser.add_argument('--record_video', action='store_true', help='whether to record video during testing')
    try:
        from argument import add_arguments
        parser = add_arguments(parser)
    except:
        pass
    args = parser.parse_args()
    return args


def run(args, record_video=False):
    start_time = time.time()
    if args.train_dqn or args.train_dqn_again:
        env_name = args.env_name or 'BreakoutNoFrameskip-v4'
        env = Environment(env_name, args, atari_wrapper=True, test=False)
        # observation = env.reset()
        # print(observation.shape)
        # traview = FrameStackTrajectoryView(env)
        # ob = traview.observation(observation)
        # obs, rew, terminated, truncated, info = env.step(0)
        # print(obs)
        # while True:
        #     obs, rew, terminated, truncated, info = env.step(2)
        #     for i in range(4):
        #         plt.imshow(obs[:, :, i], cmap='gray', interpolation='none')
        #         plt.show()
        #         plt.clf()
        from agent_dqn import Agent_DQN
        agent = Agent_DQN(env, args)
        agent.train()

    if args.test_dqn:
        render_mode_value = "rgb_array" if record_video else None
        env = Environment('BreakoutNoFrameskip-v4', args, atari_wrapper=True, test=True, render_mode=render_mode_value)
        from agent_dqn import Agent_DQN
        agent = Agent_DQN(env, args)
        test(agent, env, total_episodes=100, record_video=record_video)
    print('running time:',time.time()-start_time)

if __name__ == '__main__':
    args = parse()
    run(args, record_video=args.record_video)

