#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
import os
import sys
import random
import copy

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn

from agent import Agent
from dqn_model import DQN

from tqdm import tqdm
from gymnasium.wrappers.monitoring import video_recorder
from environment import Environment
"""
you can import any package and define any extra function as you need
"""

# torch.manual_seed(595)
# np.random.seed(595)
# random.seed(595)

class replay_buffer():
    def __init__(self, size, batch_size, device):
        self.device = device
        self.buffer_pool = [deque([], maxlen=size), deque([], maxlen=size), deque([], maxlen=size), deque([], maxlen=size), deque([], maxlen=size)]
        # self.priority = deque([], maxlen=size)
        self.bs = batch_size
    def push(self, record: tuple, live: int):
        # assert priority >= 0
        self.buffer_pool[live].append(record)
        # self.priority.append(priority)
        # assert len(self.buffer_pool) == len(self.priority)
    def len(self):
        return len(self.buffer_pool[0]), len(self.buffer_pool[1]), len(self.buffer_pool[2]), len(self.buffer_pool[3]), len(self.buffer_pool[4])
    def sample(self):
        # assert len(self.buffer_pool) == len(self.priority)
        # p_sum = sum(self.priority)
        # selected_records = np.random.choice(range(len(self.buffer_pool)), size=self.bs, replace=False, p=np.array(self.priority)/p_sum) # record: (s,a,R,s')
        # selected_s = torch.stack([self.buffer_pool[record][0] for record in selected_records], dim=0).to(self.device)
        # selected_a = torch.stack([self.buffer_pool[record][1] for record in selected_records], dim=0).to(self.device)
        # selected_R = torch.stack([self.buffer_pool[record][2] for record in selected_records], dim=0).to(self.device)
        # selected_s_ = torch.stack([self.buffer_pool[record][3] for record in selected_records], dim=0).to(self.device)
        # selected_dones = torch.stack([self.buffer_pool[record][4] for record in selected_records], dim=0).to(self.device)
        selected_records = random.sample(self.buffer_pool[0], 7)  # record: (s,a,R,s')
        selected_records += random.sample(self.buffer_pool[1], 7)  # record: (s,a,R,s')
        selected_records += random.sample(self.buffer_pool[2], 7)  # record: (s,a,R,s')
        selected_records += random.sample(self.buffer_pool[3], 7)  # record: (s,a,R,s')
        selected_records += random.sample(self.buffer_pool[4], 7)  # record: (s,a,R,s')
        random.shuffle(selected_records)
        selected_s = torch.stack([record[0] for record in selected_records], dim=0).to(self.device)
        selected_a = torch.stack([record[1] for record in selected_records], dim=0).to(self.device)
        selected_R = torch.stack([record[2] for record in selected_records], dim=0).to(self.device)
        selected_s_ = torch.stack([record[3] for record in selected_records], dim=0).to(self.device)
        selected_dones = torch.stack([record[4] for record in selected_records], dim=0).to(self.device)
        return selected_s, selected_a, selected_R, selected_s_, selected_dones

class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize everything you need here.
        For example: 
            paramters for neural network  
            initialize Q net and target Q net
            parameters for repaly buffer
            parameters for q-learning; decaying epsilon-greedy
            ...
        """

        super(Agent_DQN,self).__init__(env)
        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.args = args
        # check agent device.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = env
        self.total_episode = 2000000
        self.train_from_episode = args.train_episode if args.train_dqn_again else 0
        self.max_multistep = 4
        self.gamma = 0.99
        # initialize replay buffer
        self.replay_buffer = replay_buffer(5000, 32, self.device)

        # double Q-Learning
        self.DQN1 = DQN(4, self.env.action_space.n).to(self.device)
        self.DQN2 = DQN(4, self.env.action_space.n).to(self.device)
        
        if args.test_dqn:
            #you can load your model here
            print('loading trained model')
            ###########################
            # YOUR IMPLEMENTATION HERE #
            saved_dict = torch.load('./tested_490.37_482.3_episode_6300.ckpt')
            self.train_from_episode = saved_dict['completed_episodes']
            total_avereward_history = saved_dict['reward_history']
            self.DQN1.load_state_dict(saved_dict['DQN1'])
            self.DQN2.load_state_dict(saved_dict['DQN2'])
            print('saved parameters loaded!')

            

    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary.
        If no parameters need to be initialized, you can leave it as blank.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        
        ###########################
        pass
    
    
    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
        Return:
            action: int
                the predicted action from trained model
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        
        ###########################
        if not test and (self.args.train_dqn or self.args.train_dqn_again):
            if np.random.rand() <= self.epsilon:
                return torch.tensor(np.random.choice(self.env.action_space.n)).to(self.device)
        # decide by NN
        if test and np.random.rand() < 0.01:
            return torch.tensor(np.random.choice(self.env.action_space.n)).to(self.device)
        with torch.no_grad():
            ob = torch.tensor(observation).to(self.device)
            q1 = self.DQN1(ob)
            # q2 = self.DQN2(ob)
            return torch.argmax(q1)

    
    def push(self):
        """ You can add additional arguments as you need. 
        Push new data to buffer and remove the old one if the buffer is full.
        
        Hints:
        -----
            you can consider deque(maxlen = 10000) list
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        
        ###########################
        
        
    def replay_buffer(self):
        """ You can add additional arguments as you need.
        Select batch from buffer.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        
        
        
        ###########################
        return 
        

    def train(self):
        """
        Implement your training algorithm here
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        
        
        ###########################
        self.DQN1.train()
        self.DQN2.train()
        optimizer1 = optim.Adam(self.DQN1.parameters(), lr=1.5e-4)
        optimizer2 = optim.Adam(self.DQN2.parameters(), lr=1.5e-4)
        self.DQN2.load_state_dict(self.DQN1.state_dict())
        multi_step_scale = [1, 1 + 0.98, 1 + 0.98 + pow(0.98, 2), 1 + 0.98 + pow(0.98, 2) + pow(0.98, 3)]
        total_avereward_history = []
        step = 0 # updated times
        training_frequency = 4
        action_step = 0
        self.epsilon = 1
        if self.args.train_dqn_again:
            saved_dict = torch.load('./ave_341.7_episode_4200.ckpt')
            self.train_from_episode = saved_dict['completed_episodes']
            total_avereward_history = saved_dict['reward_history'].numpy().tolist()
            # optimizer1.load_state_dict(saved_dict['DQN1_optimizer'])
            # optimizer2.load_state_dict(saved_dict['DQN2_optimizer'])
            self.DQN1.load_state_dict(saved_dict['DQN1'])
            self.DQN2.load_state_dict(saved_dict['DQN2'])
            self.epsilon = max(self.epsilon - (0.9*100*saved_dict['completed_episodes'])/1000000, 0.1)
            print('saved parameters loaded!')
        truncated = False
        for i in tqdm(range(self.train_from_episode, self.total_episode)):
            # ob = self.env.reset()  # initialize the episode
            # print(step_total)
            if truncated:
                self.env.close()
                self.env = Environment('BreakoutNoFrameskip-v4', None, atari_wrapper=True, test=False)
            truncated = False
            for _ in range(5):
                ob = self.env.reset()  # initialize the episode
                done = False
                while not done and not truncated:  # using "not truncated" as well, when using time_limited wrapper.
                    # multi_step = 1 # random.sample([1,2,3,4], 1)[0]
                    S = copy.deepcopy(ob)
                    action = self.make_action(ob, test=False)
                    ob, reward, done, truncated, info = self.env.step(int(action.cpu().numpy()))
                    # if i <= 5000: vid_train.capture_frame()
                    if done or truncated: reward = -1
                    # if reward == 0: reward = -0.025
                    assert reward in [-1, 0, 1]
                    self.replay_buffer.push((torch.tensor(S), action, torch.tensor(reward), torch.tensor(ob), torch.tensor(done)), _)
                    self.epsilon = max(self.epsilon - (0.9)/1000000, 0.1)
                    action_step += 1
                    if self.replay_buffer.len()[4] >= 1000 and action_step % training_frequency == 0:
                        action_step = 0
                        sampled_s, sampled_a, sampled_R, sampled_s_, sampled_dones = self.replay_buffer.sample()
                        # update DQN1
                        with torch.no_grad():
                            target_eval = self.DQN2(sampled_s_)
                            next_action_est = target_eval.max(1)[0] #(bs, 1)
                            target_reward = self.gamma * (next_action_est*(~sampled_dones)) + sampled_R
                        current_est = self.DQN1(sampled_s).gather(1, sampled_a.unsqueeze(1)).squeeze()
                        optimizer1.zero_grad()
                        loss = nn.functional.smooth_l1_loss(current_est, target_reward)
                        # update para for DQN1
                        loss.backward()
                        # Clip gradients by value
                        nn.utils.clip_grad_value_(self.DQN1.parameters(), clip_value=1)
                        optimizer1.step()
                        step += 1
                        if step % 5000 == 0:
                            self.DQN2.load_state_dict(self.DQN1.state_dict())
                            step = 0
                if truncated: break
            # if i == 300: vid_train.close()
            # one episode completed, test agent after a period.
            if i % 300 == 0:
                test_env = Environment('BreakoutNoFrameskip-v4', self.args, atari_wrapper=True, test=True, render_mode="rgb_array")
                vid = video_recorder.VideoRecorder(env=test_env.env, path="/data/programs_data/class_rl_proj3/sr/video/perepisode_" + str(i) +'.mp4')
                total_reward = 0
                env_dead = False
                epi = 0
                step_inte = 0
                step_inte_rec = [0]
                for t in range(10):
                    # if env_dead: break
                    truncated = False
                    epi += 1
                    for live in range(5):
                        ob = test_env.reset()  # initialize the episode
                        done = False
                        #test_steps = 0
                        while not done and not truncated:  # using "not truncated" as well, when using time_limited wrapper.
                            action = self.make_action(ob, test=True)
                            ob, reward, done, truncated, info = test_env.step(int(action.cpu().numpy()))
                            step_inte += 1
                            #test_steps += 1
                            total_reward += reward
                            if reward > 0:
                                step_inte_rec.append(step_inte)
                                step_inte = 0
                            if t == 0: vid.capture_frame()
                        if truncated:
                            # test_env.close()
                            # test_env = Environment('BreakoutNoFrameskip-v4', self.args, atari_wrapper=True, test=True, render_mode="rgb_array")
                            env_dead = True
                            print("see truncated true!")
                            break
                    if env_dead: break
                print(epi)
                # print(step_inte_rec)
                # print(max(step_inte_rec))
                total_avereward_history.append(total_reward/10 if env_dead == False else -total_reward/epi)
                vid.close()
                test_env.close()
                print('latest_avarage: ', total_avereward_history[-1], 'history: ', total_avereward_history)
                if total_avereward_history[-1] >= 300 or total_avereward_history[-1] <= -300:
                    torch.save({
                        'completed_episodes': i,
                        'DQN1': self.DQN1.state_dict(),
                        'DQN2': self.DQN2.state_dict(),
                        'DQN1_optimizer': optimizer1.state_dict(),
                        'DQN2_optimizer': optimizer2.state_dict(),
                        'reward_history': torch.tensor(total_avereward_history)
                    }, '/data/programs_data/class_rl_proj3/sr/perave_' + str(total_avereward_history[-1]) + '_episode_' + str(i) +'.ckpt')

