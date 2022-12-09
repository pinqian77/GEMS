import random
from collections import namedtuple

# Adopt from https://github.com/pytorch/tutorials/blob/master/intermediate_source/reinforcement_q_learning.py
Transition = namedtuple('Transition',
                        ('state', 'action', 'done', 'next_state', 'ext_reward', 'int_reward')
                        )

TransitionL = namedtuple('TransitionL',
                        ('normal_state', 'oracle_state', 'next_normal_state', 'next_oracle_state', 'action', 'int_reward', 'cosine_reward', 'done', 'subgoal','next_subgoal')
                        )

TransitionH = namedtuple('TransitionH',
                        ('normal_state', 'oracle_state', 'next_normal_state', 'next_oracle_state', 'reward', 'done', 'subgoal')
                        )


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def pushL(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = TransitionL(*args)
        self.position = (self.position + 1) % self.capacity
    
    def pushH(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = TransitionH(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
