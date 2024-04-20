import random
import numpy as np
import time

FINISH_COLOR = "#FF0000"
HORIZONTAL_FIELDS = 6

START_COORDINATES = 0
FINISH_COORDINATES = 5


# START
# FINISH
SPEED = 50

# Return
RETURN_TIME = -1

# Policy iteration constants
GAMMA = 1
ALPHA = 0.01

# n in n-step TD
N = 10

# Actions
ACTION_ENCODER = {
    "right": 0,
    "left": 1, 
}

ACTION_DECODER = {
    0: "right",
    1: "left", 
}

# Epsilon
EPSILON = 0.1

class tdn_control:

    def __init__(self):

        self.V = np.random.rand(HORIZONTAL_FIELDS)
        self.R = []
        self.S = []


        # Trials, Time Steps, Velocity

        self.trials = 1
        self.success = 0
        self.time_steps = 0




    # create starting line
    def run(self):
        for episode in range(10000):
            self.coordinates = START_COORDINATES
            self.R = [0]
            self.S = [self.coordinates]
            T = 30000
            t = 0
            tau = 0
            while tau != (T - 1):
                
                if t < T:
                    self.move(random.choice(list(ACTION_ENCODER.keys())))
                    self.S.append(self.coordinates)
                    if self.coordinates == FINISH_COORDINATES:
                        self.R.append(0)
                        T = t + 1
                    else:
                        self.R.append(RETURN_TIME)
                
                tau = t - N + 1

                if tau >= 0:
                    G = 0
                    for i in range(tau + 1, np.minimum(tau + N, T) + 1):
                        G += GAMMA**(i - tau - 1) * self.R[i] 
                    if tau + N < T:
                        G = G + GAMMA**N * self.V[self.S[tau + N]]
                    self.V[self.S[tau]] = self.V[self.S[tau]] + ALPHA * (G - self.V[self.S[tau]])

                t += 1

            print(np.array(self.V).round(2))

    def run_2(self):
        for episode in range(10000):
            self.coordinates = START_COORDINATES
            self.R = [0]
            self.S = [self.coordinates]
            V_old = self.V.copy()
            T = 30000
            t = 0
            tau = 0
            while tau != (T - 1):
                
                if t < T:
                    self.move(random.choice(list(ACTION_ENCODER.keys())))
                    self.S.append(self.coordinates)
                    if self.coordinates == FINISH_COORDINATES:
                        self.R.append(0)
                        T = t + 1
                    else:
                        self.R.append(RETURN_TIME)
                
                tau = t - N + 1

                if tau >= 0:
                    G = 0
                    for i in range(tau + 1, np.minimum(tau + N, T) + 1):
                        G += GAMMA**(i - tau - 1) * self.R[i] 
                    if tau + N < T:
                        G = G + GAMMA**N * V_old[self.S[tau + N]]
                    self.V[self.S[tau]] = V_old[self.S[tau]] + ALPHA * (G - V_old[self.S[tau]])

                t += 1

            print(np.array(self.V).round(2))

    def random_walk(self):
        return random.choice(list(ACTION_ENCODER.keys()))

    def move(self, action):
        terminal_state = False
        self.time_steps += 1


        if action == "right":
            self.coordinates = int(np.minimum(self.coordinates + 1, HORIZONTAL_FIELDS - 1))
        # elif action == "left":
        #     self.coordinates = int(np.maximum(self.coordinates - 1, 0))
    

    def test(self):
        pass

tdn = tdn_control()
tdn.run()