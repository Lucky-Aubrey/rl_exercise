from collections import defaultdict
import numpy as np
import random
import matplotlib.pyplot as plt

class BairdMarkovProcess():
    def __init__(self):
        self.states = 7
        

    def step(self):
        
        if random.random() < 6/7:
            next_state = random.randint(0, self.states - 1)
            action = 0
        else:
            next_state = self.states - 1
            action = 1

        return action, next_state
    
class OffPolicy:
    initial_W = np.array([1,1,1,1,1,1,10,1])
    def __init__(self, alpha=0.01, gamma=0.99):
        self.alpha = alpha
        self.gamma = gamma
        self.W = self.initial_W
        self.X = np.array([
            [2,0,0,0,0,0,0,1],
            [0,2,0,0,0,0,0,1],
            [0,0,2,0,0,0,0,1],
            [0,0,0,2,0,0,0,1],
            [0,0,0,0,2,0,0,1],
            [0,0,0,0,0,2,0,1],
            [0,0,0,0,0,0,1,2],
            ])

    def Q(self, state, action):
        return self.W.T @ self.X[state]

    def update(self, state, next_state, action):
        delta = self.gamma * np.max((self.Q(next_state, 0), self.Q(next_state, 1))) - self.Q(state, action)
        self.W = self.W + self.alpha * delta * self.X[state]
        print()

    def reset(self):
        self.W = self.initial_W

def game(agent):
    EXPERIMENT = 1
    HIST = []
    # set env the one in Figure 8.2 of the book
    env = BairdMarkovProcess()
    for experiment in range(EXPERIMENT):
        agent.reset()
        print(f'experiment {experiment}')
        hist = []
        state = random.randint(0, env.states - 1)
        for step in range(1000):
            action, next_state = env.step()
            # direct learn
            agent.update(state, next_state, action)
            state = next_state
            hist.append(agent.W)
        HIST.append(hist)
    HIST = np.mean(HIST, axis = 0)
    return HIST

hist = game(OffPolicy())
plt.plot(hist)
plt.legend(list(range(0,8)))
plt.show()