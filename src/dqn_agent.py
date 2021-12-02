import numpy as np
import torch as T
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time
from datetime import datetime
import os
import resource
from src.lib.replay_memory import ReplayMemory, Experience
from src.lib.dqn_model import DqnModel, DuelingDqnModel
from src.lib.dqn_loss import vanilla_dqn_loss, double_dqn_loss

class DqnAgent():
    """

    """
    def __init__(self, env, env_name, config):
        self.env = env
        self.env_name = env_name
        self.config = config
        
        self.replay_memory = ReplayMemory(
            self.config['hp_replay_memory_capacity'])
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.dtype = \
                T.cuda.FloatTensor if T.cuda.is_available() else T.floatTensor
        self.training_start_timestamp = \
                datetime.now(tz=None).strftime("%Y-%m-%d_%H-%M-%S")
        self.timesteps_overall = -1
        self.timesteps_after_last_episode = 0
        self.obtained_returns = []
        self.avg_returns = []
       
        Model = DqnModel
        if self.config['hp_dueling'] == 1:
            Model = DuelingDqnModel 
        print("Used Model: ", Model)
        self.policy_net = Model(self.env.observation_space.shape,
                                   self.env.action_space.n).to(self.device)
        self.target_net = Model(self.env.observation_space.shape,
                                   self.env.action_space.n).to(self.device)
    
    def _update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def _reset(self):
        self.state = self.env.reset()
        self.last_obtained_return = 0

    def _calc_loss(self, batch):
        if bool(self.config['hp_double_dqn']):
            return double_dqn_loss(batch, self.policy_net, self.target_net,
                                    self.config['hp_gamma'], self.device,
                                    bool(self.config['hp_rescale_q_target']))
        return vanilla_dqn_loss(batch, 
                                self.policy_net, self.target_net,
                                self.config['hp_gamma'], self.device,
                                bool(self.config['hp_rescale_q_target']))

    def _get_epsilon(self):
        return max(self.config['hp_epsilon_end'], 
                   self.config['hp_epsilon_start'] - self.timesteps_overall /\
                   self.config['hp_epsilon_decay_last_frame'])

    def _select_action(self):
        epsilon = self._get_epsilon()
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            with T.no_grad():
                state = T.tensor(np.array([self.state], copy=False))\
                        .to(self.device)
                q_vals = self.policy_net(state)
                _, action = T.max(q_vals, dim=1)
                action = int(action.item())
                del state

        return action

    def _play_step(self):
        action = self._select_action()
        next_state, reward, done, _ = self.env.step(action)
        self.last_obtained_return+=reward
        exp = Experience(self.state, action, reward, done, next_state)
        self.replay_memory.append(exp)
        self.state = next_state
        del next_state

        return done


    def train_agent(self):
        """
            Train the Agent
        """
        writer = SummaryWriter(comment="_"+self.env_name+"_"+self.config['name'])
        optimizer = optim.Adam(self.policy_net.parameters(),
                               lr=self.config['hp_learning_rate'])
        print("Start Training on %s" % (self.device))

        episode = 0
        while True:
            episode += 1
            self._reset()
            done = False
            ts_episode_started = time.time()

            while(not done):
                self.timesteps_overall += 1
                done = self._play_step()

                if len(self.replay_memory) < \
                   self.config['hp_replay_memory_start_after']: 
                    continue

                if self.timesteps_overall % \
                   self.config['hp_update_frequency'] == 0:
                    optimizer.zero_grad()
                    batch=self.replay_memory.sample(self.config['hp_batch_size'])
                    loss = self._calc_loss(batch)
                    loss.backward()
                    #gradient clipping, like dueling DQN proposes
                    clipping_value = 10 #TODO Make this a config parameter
                    T.nn.utils.clip_grad_norm_(self.policy_net.parameters(),
                                               clipping_value)
                    optimizer.step()

                if self.timesteps_overall % \
                   self.config['hp_target_update_after'] == 0:
                    self._update_target_net()

            speed = (self.timesteps_overall - self.timesteps_after_last_episode)/\
                    (time.time() - ts_episode_started)
            ram_usage = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)/\
                    1024 / 1024

            self.timesteps_after_last_episode = self.timesteps_overall
            self.obtained_returns.append(self.last_obtained_return)
            self.avg_returns.append(np.mean(self.obtained_returns[-100:]))
            self._write_tensorboard(writer, speed, ram_usage)

            print("Episode %d completed, timesteps played: %d, return %d, \
                  speed %f, epsilon %f" \
                  % (episode, self.timesteps_overall, 
                     self.last_obtained_return, speed, self._get_epsilon()))
            print("Mean return of last 100 games: %f" \
                  % (self.avg_returns[-1]))
            print("Pytorch memory usage: %2f (gb)" \
                  % (ram_usage))
            print("Size of Replay Memory: %d" % (len(self.replay_memory)))

            if self.timesteps_overall >= self.config['nr_of_total_frames']:
                break

        writer.close()
        self._save_model_snapshot(self.avg_returns[-1])
        

    def _save_model_snapshot(self, score):
        if not os.path.isdir("models/vanilla/" + self.env_name):
            os.mkdir("models/vanilla/%s" % (self.env_name))
        T.save(self.policy_net.state_dict(), "models/vanilla/%s/snapshot_ts_%s_score_%d.dat" 
                                % (self.env_name, self.training_start_timestamp, score))

    def _write_tensorboard(self, writer, speed, ram_usage):
        writer.add_scalar('/Eval/AvgTotalReturn', self.avg_returns[-1], 
                          self.timesteps_overall)
        writer.add_scalar('Eval/ObtainedReturns', self.last_obtained_return,
                          self.timesteps_overall)
        writer.add_scalar('Parameter/Epsilon', self._get_epsilon(),
                          self.timesteps_overall)
        writer.add_scalar('Parameter/Speed', speed, 
                          self.timesteps_overall)
        writer.add_scalar('Parameter/MemoryUsage', ram_usage,
                          self.timesteps_overall)
        writer.add_scalar('Parameter/ReplayMemorySize', len(self.replay_memory),
                          self.timesteps_overall)
        
def _rescale_q_target(val, eps=0.01):
    return T.sign(val) * (T.sqrt(T.abs(val) + 1) - 1) + eps*val

if __name__=="__main__":
    print("Hello")
