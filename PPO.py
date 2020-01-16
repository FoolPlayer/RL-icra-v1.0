import random
from tensorflow.keras import models, layers, optimizers
import sys
sys.path.append("/home/tony/PycharmProjects/ICRA_PPO/ICRA-Simulator")
from play import ver_env
import tensorflow as tf
import numpy as np
import pandas as pd
import os


class PPO(object):
    def __int__(self, batch):
        self.optimizer = tf.keras.optimizers.Adam()
        self.ep = 100  # 训练多少回合
        self.gamma = 0.9
        self.A_LR = 0.0001
        self.C_LR = 0.0002
        self.A_UPDATE_STEPS = 10
        self.C_UPDATE_STEPS = 10
        self.batch = batch
        # self.bound = 3 #当作超参数处理
        self.env = ver_env()
        # ε for ppo2
        self.epsilon = 0.2
        self.build_model()

        self.DIM_C = 100
        self.DIM_A = 100
        self.dim_st = 40
        self.dim_ac = 50

    def _build_critic(self):
        x_critic = tf.keras.layers.Dense(self.DIM_C, input=self.state, activation='relu')
        self.v = tf.keras.layers.Dense(1, input=x_critic)
        self.advantage = self.dr - self.v

    def _build_actor(self, name, trainable):
        x_actor = tf.keras.layers.Dense(self.DIM_A, input=self.states, activation='relu')

        mu = self.bound * tf.keras.layers.Dense(1, x_actor, activation='tanh')
        sigma = tf.keras.layers.Dense(1, input=x_actor, activation='softplus')
        norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def build_model(self):
        self.state = tf.compat.v1.placeholder([None, self.dim_st], 'state')
        self.action = tf.compat.v1.placeholder([None, self.dim_ac], 'action')
        self.adv = tf.compat.v1.placeholder([None, 1], 'advantage')
        self.dr = tf.compat.v1.placeholder([None, 1], 'discount_reward')

        # build model
        self._build_critic()
        nd, pi_params = self._build_actor('actor', trainable=True)
        old_nd, oldpi_params = self._build_actor('old_actor', trainable=False)

        # define ppo loss

        # critic loss
        self.closs = tf.reduce_mean(tf.square(self.advantage))

        # actor loss
        ratio = tf.exp(nd.log_prob(self.action) - old_nd.log_prob(self.action))
        surr = ratio * self.adv

        self.aloss = -tf.reduce_mean(tf.minimum(surr,
                                                tf.clip_by_value(ratio, 1. - self.epsilon,
                                                                 1. + self.epsilon) * self.adv))

        # define Optimizer
        self.ctrain_op = tf.train.AdamOptimizer(self.C_LR).minimize(self.closs)
        self.atrain_op = tf.train.AdamOptimizer(self.A_LR).minimize(self.aloss)

        # sample action
        self.sample_op = tf.squeeze(nd.sample(1), axis=0)

        # update old actor
        self.update_old_actor = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

    def choose_action(self, state):
        state = state[np.newaxis, :]
        action = self.sample_op(self, state)[0]
        return np.clip(action, -self.bound, self.bound)

    def save_history(self, history, name):
        name = os.path.join('history', name)

        df = pd.DataFrame.from_dict(history)
        df.to_csv(name, index=False, encoding='utf-8')

    def train(self):
        """train method.
        """
        tf.reset_default_graph()
        history = {'episode': [], 'Episode_reward': []}

        for i in range(self.ep):
            obs_fri, obs_ene = self.env.reset()

            states_fri, actions_fri, rewards_fri = [], [], []
            states_ene, actions_ene, rewards_ene = [], [], []
            episode_reward_fri = 0
            episode_reward_ene = 0
            j = 0

            while True:
                action_fri = self.choose_action(obs_fri)
                action_ene = self.choose_action(obs_ene)

                next_obs_fri, rew_fri, next_obs_ene, rew_ene, done, _ = self.env.step(action_fri.numpy(),
                                                                                      action_ene.numpy())
                states_fri.append(obs_fri)
                actions_fri.append(action_fri)
                states_ene.append(obs_fri)
                actions_ene.append(action_ene)

                episode_reward_fri += rew_fri
                rewards_fri.append((rew_fri + 8) / 8)
                episode_reward_ene += rew_ene
                rewards_ene.append((rew_ene + 8) / 8)

                obs_fri = next_obs_fri
                obs_ene = next_obs_ene

                if (j + 1) % self.batch == 0:
                    states_fri = np.array(states_fri)
                    actions_fri = np.array(actions_fri)
                    rewards_fri = np.array(rew_fri)
                    d_reward_fri = self.discount_reward(states_fri, rewards_fri, next_obs_fri)
                    states_ene = np.array(states_ene)
                    actions_ene = np.array(actions_ene)
                    rewards_ene = np.array(rewards_ene)
                    d_reward_ene = self.discount_reward(states_ene, rewards_ene, next_obs_ene)

                    self.update(states_fri, actions_fri, d_reward_fri)
                    self.update(states_ene, actions_ene, d_reward_ene)
                    states_fri, actions_fri, rewards_fri = [], [], []
                    states_ene, actions_ene, rewards_ene = [], [], []
                if done:
                    break
                j += 1

            history['episode'].append(i)
            history['Episode_reward_fri'].append(episode_reward_fri)
            history['Episode_reward_ene'].append(episode_reward_ene)
            print(
                'Episode: {} | Episode reward fri: {:.2f} | | Episode reward ene : {:.2f}'.format(i,
                                                                                                  episode_reward_fri,
                                                                                                  episode_reward_ene))

        return history

    def discount_reward(self, states, rewards, next_observation):
        """Compute target value.

        Arguments:
            states: state in episode.
            rewards: reward in episode.
            next_observation: state of last action.

        Returns:
            targets: q targets.
        """
        s = np.vstack([states, next_observation.reshape(-1, 3)])
        q_values = self.get_value(s).flatten()

        targets = rewards + self.gamma * q_values[1:]
        targets = targets.reshape(-1, 1)

        return targets

    def update(self, states, action, dr):
        """update model.

        Arguments:
            states: states.
            action: action of states.
            dr: discount reward of action.
        """
        self.update_old_actor()

        adv = self.advantage(self.states, self.dr)

        # update actor
        # run ppo2 loss
        for _ in range(self.A_UPDATE_STEPS):
            self.atrain_op(self.states, self.action, self.adv)

        # update critic
        for _ in range(self.C_UPDATE_STEPS):
            self.ctrain_op(self.states, self.dr)

    def get_value(self, state):
        """get q value.

        Arguments:
            state: state.

        Returns:
           q_value.
        """
        if state.ndim < 2: state = state[np.newaxis, :]

        return self.v(state)
