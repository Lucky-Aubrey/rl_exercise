from tkinter import *
import random
import numpy as np
import time

BORDER_COLOR = "#444444"
AGENT_COLOR = "#4444FF"
BACKGROUND_COLOR = "#000000"
START_COLOR = "#00FF00"
FINISH_COLOR = "#FF0000"
VERTICAL_FIELDS = 7
HORIZONTAL_FIELDS = 10
SPACE_SIZE = 50
GAME_HEIGHT = VERTICAL_FIELDS * SPACE_SIZE
GAME_WIDTH = HORIZONTAL_FIELDS * SPACE_SIZE

START_COORDINATES = [0, 3]
FINISH_COORDINATES = [7, 3]

WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

if len(WIND) != HORIZONTAL_FIELDS:
    raise ValueError("Wind information must match vertical fields")

# START
# FINISH
SPEED = 50

# Return
RETURN_LOOSE = -1

# Policy iteration constants
GAMMA = 1
ALPHA = 0.5

# Actions
ACTION_ENCODER = {
    "right": 0,
    "left": 1, 
    "up": 2, 
    "down": 3,
    "right-up": 4,
    "right-down": 5,
    "left-up": 6,
    "left-down": 7,
    "stay": 8,
}

ACTION_DECODER = {
    0: "right",
    1: "left", 
    2: "up", 
    3: "down",
    4: "right-up",
    5: "right-down",
    6: "left-up",
    7: "left-down",
    8: "stay",
}

# Epsilon
EPSILON = 0.1

class sarsa_control:

    def __init__(self):

        self.q = np.random.rand(HORIZONTAL_FIELDS, VERTICAL_FIELDS, len(ACTION_ENCODER.keys()))
        self.q[*FINISH_COORDINATES,:] = 0

        self.target_policy = {} # [x,y,vx,vy] : "x+1"/"x-1"/"y+1"/"y-1"
        self.behaviour_policy = {} # [x,y,vx,vy] : "x+1"/"x-1"/"y+1"/"y-1"

        self.window = Tk()
        self.window.title("Right Turn")
        self.window.resizable(False, False)

        # Trials, Time Steps, Velocity

        self.trials = 1
        self.success = 0
        self.time_steps = 0

        # Track Steps
        self.label = Label(self.window, text="Try:{} ".format(self.trials)+"t:{}".format(self.time_steps), font=('consolas', 20))
        self.label.pack()

        # Background
        self.canvas = Canvas(self.window, bg=BACKGROUND_COLOR, height=GAME_HEIGHT, width=GAME_WIDTH)
        self.canvas.pack()

        # Set window to middle
        self.window.update()

        self.window_width = self.window.winfo_width()
        self.window_height = self.window.winfo_height()
        self.screen_width = self.window.winfo_screenwidth()
        self.screen_height = self.window.winfo_screenheight()

        self.x = int((self.screen_width/2) - (self.window_width/2))
        self.y = int((self.screen_height/2) - (self.window_height/2))

        self.window.geometry(f"{self.window_width}x{self.window_height}+{self.x}+{self.y}")

        self.start = START_COORDINATES.copy()
        self.canvas.create_rectangle(self.start[0] * SPACE_SIZE, 
                                     self.start[1] * SPACE_SIZE, 
                                     self.start[0] * SPACE_SIZE + SPACE_SIZE, 
                                     self.start[1] * SPACE_SIZE + SPACE_SIZE, 
                                     fill=START_COLOR, tag="start")
        self.finish = FINISH_COORDINATES
        self.canvas.create_rectangle(self.finish[0] * SPACE_SIZE, 
                                     self.finish[1] * SPACE_SIZE, 
                                     self.finish[0] * SPACE_SIZE + SPACE_SIZE, 
                                     self.finish[1] * SPACE_SIZE + SPACE_SIZE, 
                                     fill=FINISH_COLOR, tag="finish")
        
        self.coordinates = START_COORDINATES.copy()
        self.circle = self.canvas.create_oval(self.coordinates[0] * SPACE_SIZE, 
                                                self.coordinates[1] * SPACE_SIZE, 
                                                self.coordinates[0] * SPACE_SIZE + SPACE_SIZE, 
                                                self.coordinates[1] * SPACE_SIZE + SPACE_SIZE, 
                                                fill=AGENT_COLOR, tag="agent")


    # create starting line
    def run(self):
        self.loop()
        self.window.mainloop()

    def loop(self, A = None, update_Q = True): # Choose A from S using policy derived from Q
        
        # Choose A from S using policy derived from Q
        if A is None:
            A = self.epsilon_greedy_action()

        # store S in local variable
        S = self.coordinates.copy()

        # execute action and update state, check if a new episode started
        new_episode = self.move(ACTION_DECODER[A]) # decode: 0 -> "right"
        
        # observe R and S'
        if new_episode: # Win
            R = 0
            S_new = FINISH_COORDINATES.copy()
        else: # Each step has a return -1
            R = -1
            S_new = self.coordinates.copy()


        # Choose A' from S' using policy derived from Q
        A_new = self.epsilon_greedy_action()

        # update state action value table - Q
        if update_Q:
            self.q[*S, A] = self.q[*S, A] + ALPHA * (R + GAMMA * self.q[*S_new, A_new] - self.q[*S, A])
        

        # proceed with next iteration of the loop, use A_new and S_new as A and S (S is already updated through self.coordinates)
        if self.trials < 5000:
            self.window.after(0, self.loop, A_new)
        else:
            global EPSILON
            EPSILON = -1
            self.window.after(SPEED, self.loop, A_new, False)

    def epsilon_greedy_action(self):
        if random.random() > EPSILON:
            A = np.argmax(self.q[*self.coordinates])
        else:
            A = random.choice(list(range(len(list(ACTION_ENCODER.keys())))))
        return A

    def move(self, action):
        new_episode = False
        self.time_steps += 1

        self.label.config(text="Try:{} ".format(self.trials)+"t:{}".format(self.time_steps))

        self.canvas.delete(self.circle)

        horizontal_position = self.coordinates[0]

        if action == "right":
            self.coordinates[0] = int(np.minimum(self.coordinates[0] + 1, HORIZONTAL_FIELDS - 1))
        elif action == "left":
            self.coordinates[0] = int(np.maximum(self.coordinates[0] - 1, 0))
        elif action == "up":
            self.coordinates[1] = int(np.maximum(self.coordinates[1] - 1, 0))
        elif action == "down":
            self.coordinates[1] = int(np.minimum(self.coordinates[1] + 1, VERTICAL_FIELDS - 1))
        elif action == "right-up":
            self.coordinates[0] = int(np.minimum(self.coordinates[0] + 1, HORIZONTAL_FIELDS - 1))
            self.coordinates[1] = int(np.maximum(self.coordinates[1] - 1, 0))
        elif action == "right-down":
            self.coordinates[0] = int(np.minimum(self.coordinates[0] + 1, HORIZONTAL_FIELDS - 1))
            self.coordinates[1] = int(np.minimum(self.coordinates[1] + 1, VERTICAL_FIELDS - 1))
        elif action == "left-up":
            self.coordinates[0] = int(np.maximum(self.coordinates[0] - 1, 0))
            self.coordinates[1] = int(np.maximum(self.coordinates[1] - 1, 0))
        elif action == "left-down":
            self.coordinates[0] = int(np.maximum(self.coordinates[0] - 1, 0))
            self.coordinates[1] = int(np.minimum(self.coordinates[1] + 1, VERTICAL_FIELDS - 1))

        
        # wind
        self.coordinates[1] = int(np.maximum(self.coordinates[1] - WIND[horizontal_position], 0))
        
        if self.coordinates == FINISH_COORDINATES: # Win
            self.trials += 1
            self.time_steps = 0
            self.coordinates = START_COORDINATES.copy() # restart
            new_episode = True

        self.update_canvas(tag="agent")

        return new_episode
    
    def update_canvas(self, tag):
        del self.circle
        self.circle = self.canvas.create_oval(self.coordinates[0] * SPACE_SIZE, 
                                            self.coordinates[1] * SPACE_SIZE, 
                                            self.coordinates[0] * SPACE_SIZE + SPACE_SIZE, 
                                            self.coordinates[1] * SPACE_SIZE + SPACE_SIZE, 
                                            fill=AGENT_COLOR, tag=tag)
        

    def test(self, action=random.choice(list(ACTION_ENCODER.keys()))):
        if self.trials < 10:
            self.move(action)
            # self.test()
            # self.window.after(SPEED*10, self.test, "right")
            self.window.after(1, self.test, random.choice(list(ACTION_ENCODER.keys())))
        else:
            self.move("right")
            self.window.after(SPEED*10, self.test, "right")


sarsa = sarsa_control()
# sarsa.test()
sarsa.run()