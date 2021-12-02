from typing import Tuple, List, Union
from collections import namedtuple, deque
import numpy as np

Experience = namedtuple("Experience", field_names = \
                       ["state", "action", "reward", "done", "next_state"])

class ReplayMemory:
    """
    Original replay memory by Lin. Used for vanilla DQN, samples
    uniformely, no PER.
    """
    def __init__(self, capacity: int) -> None:
        """
        Args:
            capacity: size of buffer, int
        """
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, sample: Experience) -> None:
        """
        Append sample
        Args:
            sample A sample of an experience to store. Experience is 
            a tuple of (state, action, reward, done, next_state)
        """
        self.buffer.append(sample)


    def sample(self, batch_size: int = 1) -> Tuple:
        """
        Return batch of buffer, randomly (uniformly).
        Args:
            batch_size: size of batch
        """
        idxs = np.random.choice(len(self), batch_size, replace=False)

        states, actions, rewards, dones, next_states = \
                zip(*[self.buffer[idx] for idx in idxs])

        return np.array(states), np.array(actions), \
                np.array(rewards, dtype=np.float32), \
                np.array(dones, dtype=bool), \
                np.array(next_states)


class ReplayMemoryPER(ReplayMemory):
    def __init__(self, capacity: int, epsilon=0.01, alpha=0.6, beta=0.4) -> None:
        """
        """
        super(self, ReplayMemoryPER).__init__(capacity)
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta_start = beta
        self.beta = beta
        self.first_append = True

    def append(self, sample: Experience, tderror) -> None:
        pass

    def get_priority(self, tderror):
        pass

    def _recalculate_beta(self, frame, max_frame):
        pass

    def get_beta(self):
        return self.beta

    def sample(self, batch_size, frame:int, max_frame:int) -> Tuple:
        pass


class SumTree():
    pass


class WeightedLoss():
    pass
