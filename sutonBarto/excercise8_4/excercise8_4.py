import random
import numpy as np
import time
import matplotlib.pyplot as plt

VERTICAL_FIELDS = 6
HORIZONTAL_FIELDS = 9

START_COORDINATES = [3, 5]
FINISH_COORDINATES = [8, 0]



# START
# FINISH

# Return
RETURN_WIN = 1

# Policy iteration constants
GAMMA = 0.95
ALPHA = 0.1

# Planning loops
N = 5

# Exploration term
KAPPA = 0.01

# Epsilon
EPSILON = 0.2

# Task variation
BLOCKING_TASK = False
SHORTCUT_TASK = True
if BLOCKING_TASK and SHORTCUT_TASK:
    raise ValueError("both tasks cant be set to True")

#
MAX_EPISODES = 1000

# Actions
ACTION_ENCODER = {
    "right": 0,
    "left": 1, 
    "up": 2, 
    "down": 3,
}

ACTION_DECODER = {
    0: "right",
    1: "left", 
    2: "up", 
    3: "down",
}

class dynaQ:

    def __init__(self):

        self.Q = np.random.rand(HORIZONTAL_FIELDS, VERTICAL_FIELDS, len(list(ACTION_DECODER.keys())))
        self.model_dictionary = {}

        self.S = None
        self.S_next = None
        self.A = None
        self.R = None

        self.wall = [
            # [0,3],
            [1,3],
            [2,3],
            [3,3],
            [4,3],
            [5,3],
            [6,3],
            [7,3],
            [8,3],
        ]


        # statistics
        self.time_steps_taken_per_episode = []
        self.num_runs = 0
    
    def reset(self):
        self.Q = np.random.rand(HORIZONTAL_FIELDS, VERTICAL_FIELDS, len(list(ACTION_DECODER.keys())))
        self.model_dictionary = {}

        self.S = None
        self.S_next = None
        self.A = None
        self.R = None

        self.wall = [
            # [0,3],
            [1,3],
            [2,3],
            [3,3],
            [4,3],
            [5,3],
            [6,3],
            [7,3],
            [8,3],
        ]


    # create starting line
    def run(self):
        self.num_runs += 1
        self.S = START_COORDINATES.copy()
        episode = 1
        t = 0
        while episode < MAX_EPISODES:
            new_episode = False
            self.A = self.epsilon_greedy(self.S)
            self.move(self.A)
            if self.S == FINISH_COORDINATES:
                new_episode = True
                self.R = 1
            else:
                self.R = 0
            # Q update
            self.Q[*self.S, ACTION_ENCODER[self.A]] = self.Q[*self.S, ACTION_ENCODER[self.A]]\
                + ALPHA * (self.R + GAMMA * np.max(self.Q[*self.S_next]) - self.Q[*self.S, ACTION_ENCODER[self.A]])

            self.insert_model_information(self.S, self.A, self.R, self.S_next)
            for _ in range(N):
                model_S, model_A = self.random_observed_state_action()
                model_R, model_S_next = self.model_dictionary[tuple(model_S)][model_A]
                self.Q[*model_S, ACTION_ENCODER[model_A]] = self.Q[*model_S, ACTION_ENCODER[model_A]]\
                    + ALPHA * (model_R + GAMMA * np.max(self.Q[*model_S_next]) - self.Q[*model_S, ACTION_ENCODER[model_A]])
            
            self.S = self.S_next.copy()

            if new_episode:
                print(f'run {self.num_runs} - episode {episode} - time {t}')
                if self.num_runs > 1:
                    self.time_steps_taken_per_episode[episode-1] += ((t + 1) - self.time_steps_taken_per_episode[episode-1]) / self.num_runs
                else:
                    self.time_steps_taken_per_episode.append(t+1)
                t = 0
                self.S = START_COORDINATES.copy()
                episode += 1
                if (BLOCKING_TASK or SHORTCUT_TASK) and (episode == 100):
                    self.wall.pop(-1)
                    if BLOCKING_TASK:
                        self.wall.insert(0, [0, 3])

            t += 1

    
    def insert_model_information(self, s, a, r, s_next):
        state = tuple(s)
        if state not in list(self.model_dictionary.keys()):
            self.model_dictionary[state] = {}
        self.model_dictionary[state][a] = [r, s_next]

    def random_observed_state_action(self):
        s = random.choice(list(self.model_dictionary.keys()))
        a = random.choice(list(self.model_dictionary[s].keys()))
        return s, a


    def epsilon_greedy(self, state):
        if random.random() < EPSILON:
            return random.choice(list(ACTION_ENCODER.keys()))
        else:
            return ACTION_DECODER[np.argmax(self.Q[*state])]

    def move(self, action):
        self.S_next = list(self.S).copy()
        if action == "right":
            self.S_next[0] = int(np.minimum(self.S[0] + 1, HORIZONTAL_FIELDS - 1))
        elif action == "left":
            self.S_next[0] = int(np.maximum(self.S[0] - 1, 0))
        elif action == "up":
            self.S_next[1] = int(np.maximum(self.S[1] - 1, 0))
        elif action == "down":
            self.S_next[1] = int(np.minimum(self.S[1] + 1, VERTICAL_FIELDS - 1))

        if self.S_next in self.wall:
            self.S_next = self.S.copy()

    def plot(self):
        show_array = np.argmax(self.Q, axis=-1)
        for a in self.wall:
            show_array[*a] = 8
        for a in show_array.transpose():
            print(a)

        plt.plot(self.time_steps_taken_per_episode)
        plt.ylim(10,30)
        if BLOCKING_TASK:
            plt.savefig(f'excercise8_4/dynaQ_{N}_{ALPHA}_{EPSILON}_blocking.png')
        elif SHORTCUT_TASK:
            plt.savefig(f'excercise8_4/dynaQ_{N}_{ALPHA}_{EPSILON}_shortcut.png')
        else:
            plt.savefig(f'excercise8_4/dynaQ_{N}_{ALPHA}_{EPSILON}.png')
        plt.show()


    def test(self):
        pass

class dynaQplus:

    def __init__(self):

        self.Q = np.random.rand(HORIZONTAL_FIELDS, VERTICAL_FIELDS, len(list(ACTION_DECODER.keys())))
        self.model_dictionary = {}

        self.S = None
        self.S_next = None
        self.A = None
        self.R = None
        self.tau = np.zeros((HORIZONTAL_FIELDS, VERTICAL_FIELDS, len(list(ACTION_DECODER.keys()))))

        self.wall = [
            # [0,3],
            [1,3],
            [2,3],
            [3,3],
            [4,3],
            [5,3],
            [6,3],
            [7,3],
            [8,3],
        ]


        # statistics
        self.time_steps_taken_per_episode = []
        self.num_runs = 0
    
    def reset(self):
        self.Q = np.random.rand(HORIZONTAL_FIELDS, VERTICAL_FIELDS, len(list(ACTION_DECODER.keys())))
        self.model_dictionary = {}

        self.S = None
        self.S_next = None
        self.A = None
        self.R = None
        self.tau = np.zeros((HORIZONTAL_FIELDS, VERTICAL_FIELDS, len(list(ACTION_DECODER.keys()))))

        self.wall = [
            # [0,3],
            [1,3],
            [2,3],
            [3,3],
            [4,3],
            [5,3],
            [6,3],
            [7,3],
            [8,3],
        ]


    # create starting line
    def run(self):
        self.num_runs += 1
        self.S = START_COORDINATES.copy()
        episode = 1
        t = 1
        while episode < MAX_EPISODES:
            new_episode = False
            self.A = self.epsilon_greedy(self.S)
            self.move(self.A)
            # keep track of move
            self.tau += 1
            self.tau[*self.S, ACTION_ENCODER[self.A]] = 0
            if self.S == FINISH_COORDINATES:
                new_episode = True
                self.R = RETURN_WIN
            else:
                self.R = 0
            # Q update
            self.Q[*self.S, ACTION_ENCODER[self.A]] = self.Q[*self.S, ACTION_ENCODER[self.A]]\
                + ALPHA * (self.R + GAMMA * np.max(self.Q[*self.S_next]) - self.Q[*self.S, ACTION_ENCODER[self.A]])

            self.insert_model_information(self.S, self.A, self.R, self.S_next)
            for _ in range(N):
                model_S, model_A = self.random_observed_state_action()
                model_R, model_S_next = self.model_dictionary[tuple(model_S)][model_A]
                # add reward for state action not being tried
                model_R += KAPPA * np.sqrt(self.tau[*model_S, ACTION_ENCODER[model_A]])
                self.Q[*model_S, ACTION_ENCODER[model_A]] = self.Q[*model_S, ACTION_ENCODER[model_A]]\
                    + ALPHA * (model_R + GAMMA * np.max(self.Q[*model_S_next]) - self.Q[*model_S, ACTION_ENCODER[model_A]])
            
            self.S = self.S_next.copy()

            if new_episode:
                print(f'run {self.num_runs} - episode {episode} - time {t}')
                if self.num_runs > 1:
                    self.time_steps_taken_per_episode[episode-1] += (t - self.time_steps_taken_per_episode[episode-1]) / self.num_runs
                else:
                    self.time_steps_taken_per_episode.append(t)
                t = 0
                self.S = START_COORDINATES.copy()
                episode += 1
                if (BLOCKING_TASK or SHORTCUT_TASK) and (episode == 100):
                    self.wall.pop(-1)
                    if BLOCKING_TASK:
                        self.wall.insert(0, [0, 3])

            t += 1

    
    def insert_model_information(self, s, a, r, s_next):
        state = tuple(s)
        if state not in list(self.model_dictionary.keys()):
            self.model_dictionary[state] = {}
        self.model_dictionary[state][a] = [r, s_next]

    def random_observed_state_action(self):
        s = random.choice(list(self.model_dictionary.keys()))
        a = random.choice(list(self.model_dictionary[s].keys()))
        return s, a


    def epsilon_greedy(self, state):
        if random.random() < EPSILON:
            return random.choice(list(ACTION_ENCODER.keys()))
        else:
            return ACTION_DECODER[np.argmax(self.Q[*state])]

    def move(self, action):
        self.S_next = list(self.S).copy()
        if action == "right":
            self.S_next[0] = int(np.minimum(self.S[0] + 1, HORIZONTAL_FIELDS - 1))
        elif action == "left":
            self.S_next[0] = int(np.maximum(self.S[0] - 1, 0))
        elif action == "up":
            self.S_next[1] = int(np.maximum(self.S[1] - 1, 0))
        elif action == "down":
            self.S_next[1] = int(np.minimum(self.S[1] + 1, VERTICAL_FIELDS - 1))

        if self.S_next in self.wall:
            self.S_next = self.S.copy()

    def plot(self):
        show_array = np.argmax(self.Q, axis=-1)
        for a in self.wall:
            show_array[*a] = 8
        for a in show_array.transpose():
            print(a)

        plt.plot(self.time_steps_taken_per_episode)
        plt.ylim(10,30)
        if BLOCKING_TASK:
            plt.savefig(f'excercise8_4/dynaQplus_{N}_{ALPHA}_{EPSILON}_blocking.png')
        elif SHORTCUT_TASK:
            plt.savefig(f'excercise8_4/dynaQplus_{N}_{ALPHA}_{EPSILON}_shortcut.png')
        else:
            plt.savefig(f'excercise8_4/dynaQplus_{N}_{ALPHA}_{EPSILON}.png')
        plt.show()


    def test(self):
        pass

class dynaQplusExp:

    def __init__(self):

        self.Q = np.random.rand(HORIZONTAL_FIELDS, VERTICAL_FIELDS, len(list(ACTION_DECODER.keys())))
        self.model_dictionary = {}

        self.S = None
        self.S_next = None
        self.A = None
        self.R = None
        self.tau = np.zeros((HORIZONTAL_FIELDS, VERTICAL_FIELDS, len(list(ACTION_DECODER.keys()))))

        self.wall = [
            # [0,3],
            [1,3],
            [2,3],
            [3,3],
            [4,3],
            [5,3],
            [6,3],
            [7,3],
            [8,3],
        ]


        # statistics
        self.time_steps_taken_per_episode = []
        self.num_runs = 0
    
    def reset(self):
        self.Q = np.random.rand(HORIZONTAL_FIELDS, VERTICAL_FIELDS, len(list(ACTION_DECODER.keys())))
        self.model_dictionary = {}

        self.S = None
        self.S_next = None
        self.A = None
        self.R = None
        self.tau = np.zeros((HORIZONTAL_FIELDS, VERTICAL_FIELDS, len(list(ACTION_DECODER.keys()))))

        self.wall = [
            # [0,3],
            [1,3],
            [2,3],
            [3,3],
            [4,3],
            [5,3],
            [6,3],
            [7,3],
            [8,3],
        ]


    # create starting line
    def run(self):
        self.num_runs += 1
        self.S = START_COORDINATES.copy()
        episode = 1
        t = 1
        while episode < MAX_EPISODES:
            new_episode = False
            self.A = self.epsilon_greedy(self.S)
            self.move(self.A)
            # keep track of move
            self.tau += 1
            self.tau[*self.S, ACTION_ENCODER[self.A]] = 0
            if self.S == FINISH_COORDINATES:
                new_episode = True
                self.R = RETURN_WIN
            else:
                self.R = 0
            # Q update
            self.Q[*self.S, ACTION_ENCODER[self.A]] = self.Q[*self.S, ACTION_ENCODER[self.A]]\
                + ALPHA * (self.R + GAMMA * np.max(self.Q[*self.S_next]) - self.Q[*self.S, ACTION_ENCODER[self.A]])

            self.insert_model_information(self.S, self.A, self.R, self.S_next)
            for _ in range(N):
                model_S, model_A = self.random_observed_state_action()
                model_R, model_S_next = self.model_dictionary[tuple(model_S)][model_A]
                # add reward for state action not being tried
                self.Q[*model_S, ACTION_ENCODER[model_A]] = self.Q[*model_S, ACTION_ENCODER[model_A]]\
                    + ALPHA * (model_R + GAMMA * np.max(self.Q[*model_S_next]) - self.Q[*model_S, ACTION_ENCODER[model_A]])
            
            self.S = self.S_next.copy()

            if new_episode:
                print(f'run {self.num_runs} - episode {episode} - time {t}')
                if self.num_runs > 1:
                    self.time_steps_taken_per_episode[episode-1] += (t - self.time_steps_taken_per_episode[episode-1]) / self.num_runs
                else:
                    self.time_steps_taken_per_episode.append(t)
                t = 0
                self.S = START_COORDINATES.copy()
                episode += 1
                if (BLOCKING_TASK or SHORTCUT_TASK) and (episode == 100):
                    self.wall.pop(-1)
                    if BLOCKING_TASK:
                        self.wall.insert(0, [0, 3])

            t += 1

    
    def insert_model_information(self, s, a, r, s_next):
        state = tuple(s)
        if state not in list(self.model_dictionary.keys()):
            self.model_dictionary[state] = {}
        self.model_dictionary[state][a] = [r, s_next]

    def random_observed_state_action(self):
        s = random.choice(list(self.model_dictionary.keys()))
        a = random.choice(list(self.model_dictionary[s].keys()))
        return s, a


    def epsilon_greedy(self, state):
        if random.random() < EPSILON:
            return random.choice(list(ACTION_ENCODER.keys()))
        else:
            return ACTION_DECODER[np.argmax(self.Q[*state]+KAPPA * np.sqrt(self.tau[*state])) ]

    def move(self, action):
        self.S_next = list(self.S).copy()
        if action == "right":
            self.S_next[0] = int(np.minimum(self.S[0] + 1, HORIZONTAL_FIELDS - 1))
        elif action == "left":
            self.S_next[0] = int(np.maximum(self.S[0] - 1, 0))
        elif action == "up":
            self.S_next[1] = int(np.maximum(self.S[1] - 1, 0))
        elif action == "down":
            self.S_next[1] = int(np.minimum(self.S[1] + 1, VERTICAL_FIELDS - 1))

        if self.S_next in self.wall:
            self.S_next = self.S.copy()

    def plot(self):
        show_array = np.argmax(self.Q, axis=-1)
        for a in self.wall:
            show_array[*a] = 8
        for a in show_array.transpose():
            print(a)

        plt.plot(self.time_steps_taken_per_episode)
        plt.ylim(10,30)
        if BLOCKING_TASK:
            plt.savefig(f'excercise8_4/dynaQplusExp_{N}_{ALPHA}_{EPSILON}_blocking.png')
        elif SHORTCUT_TASK:
            plt.savefig(f'excercise8_4/dynaQplusExp_{N}_{ALPHA}_{EPSILON}_shortcut.png')
        else:
            plt.savefig(f'excercise8_4/dynaQplusExp_{N}_{ALPHA}_{EPSILON}.png')
        plt.show()


    def test(self):
        pass

# dynaQ = dynaQ()
# dynaQ = dynaQplus()
dynaQ = dynaQplusExp()
for i in range(5):
    dynaQ.run()
    dynaQ.reset()
dynaQ.plot()