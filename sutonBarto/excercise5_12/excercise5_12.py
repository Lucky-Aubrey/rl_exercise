from tkinter import *
import random
import numpy as np

BORDER_COLOR = "#444444"
CAR_COLOR = "#4444FF"
BACKGROUND_COLOR = "#000000"
START_COLOR = "#00FF00"
FINISH_COLOR = "#FF0000"
GAME_HEIGHT = 500
GAME_WIDTH = 500
SPACE_SIZE = 25
START_BEGIN = 1
START_END = 2
FINISH_BEGIN = 0
FINISH_END = 8

START = [[i * SPACE_SIZE, GAME_HEIGHT - SPACE_SIZE] for i in range(START_BEGIN, START_END)]
FINISH = [[GAME_WIDTH - SPACE_SIZE, i * SPACE_SIZE] for i in range(FINISH_BEGIN, FINISH_END)] + \
    [[GAME_WIDTH - 2 * SPACE_SIZE, i * SPACE_SIZE] for i in range(FINISH_BEGIN, FINISH_END)] + \
    [[GAME_WIDTH - 3 * SPACE_SIZE, i * SPACE_SIZE] for i in range(FINISH_BEGIN, FINISH_END)]
SPEED = 1
BORDER_BLOCKS = [[0]*20, # 1
          [0]*20,
          [0]*20,
          [0]*20,
          [0]*20,
          [0]*20, # 6
          [0]*20,
          [0]*20,
          [0]*8+[1]*12,
          [0]*8+[1]*12,
          [0]*8+[1]*12, # 11
          [0]*8+[1]*12,
          [0]*8+[1]*12,
          [0]*8+[1]*12,
          [0]*8+[1]*12,
          [0]*8+[1]*12, # 16
          [0]*8+[1]*12,
          [0]*8+[1]*12,
          [0]*8+[1]*12,
          [0]*8+[1]*12,]

# Return
RETURN_WIN = 1
RETURN_LOOSE = -1
RETURN_TIME = 0

# Policy iteration constants
GAMMA = 1

# Actions
ACTIONS = ["right", "left", "up", "down", None]

# Epsilon
EPSILON = 0.1

class Border:

    def __init__(self):
        self.border_cells = []
        for i, row in enumerate(BORDER_BLOCKS):
            for j, block in enumerate(row):
                if block:
                    x = i * SPACE_SIZE
                    y = j * SPACE_SIZE
                    self.border_cells.append([x, y])
                    canvas.create_rectangle(x, y, x + SPACE_SIZE, y + SPACE_SIZE, fill=BORDER_COLOR, tag="border")

class Start:

    def __init__(self):
        
        self.coordinates = []

        for i in range(START_BEGIN, START_END):
            x = i * SPACE_SIZE
            y = GAME_HEIGHT - SPACE_SIZE
            
            self.coordinates.append([x, y])
            
            canvas.create_rectangle(x, y, x + SPACE_SIZE, y + SPACE_SIZE, fill=START_COLOR, tag="start")

class Finish:

    def __init__(self):
        
        self.coordinates = []

        for x, y in FINISH:

            self.coordinates.append([x, y])
            
            canvas.create_rectangle(x, y, x + SPACE_SIZE, y + SPACE_SIZE, fill=FINISH_COLOR, tag="finish")


class Car:

    def __init__(self):
        
        global start

        x, y = random.choice(start.coordinates)

        self.coordinates = [x, y]
        self.velocity = [0,-1]
        self.state_action_return_list = []

        self.weight = 1
        self.G = 0 # return

        self.pick_action()
        self.circle = canvas.create_oval(x, y, x + SPACE_SIZE, y + SPACE_SIZE, fill=CAR_COLOR, tag="car")
    
    def reset(self):
        global start
        x, y = random.choice(start.coordinates)
        self.coordinates = [x, y]
        self.velocity = [0,-1]
        self.pick_action()
        self.circle = canvas.create_oval(x, y, x + SPACE_SIZE, y + SPACE_SIZE, fill=CAR_COLOR, tag="car")

    def pick_action(self):
        global trials, SPEED
        state = (*self.coordinates, *self.velocity)
        speed_switch = (trials < 100)
        if not speed_switch: SPEED = 50 
        if state in target_policy and random.random() > EPSILON:
            self.action = target_policy[state]
        else:
            self.action = random.choice(ACTIONS)


def next_time_step(car):

    global time_steps

    time_steps += 1

    label.config(text="Try:{} ".format(trials)+"Time Steps:{}".format(time_steps))

    x, y = [car.coordinates[0] + car.velocity[0] * SPACE_SIZE, car.coordinates[1] + car.velocity[1] * SPACE_SIZE]
    canvas.delete(car.circle)

    if collision([x, y], car.coordinates):
        car.state_action_return_list.append([(car.coordinates[0], car.coordinates[1], car.velocity[0], car.velocity[1]), car.action, RETURN_LOOSE])
        reset_car(car)
        
    elif win([x, y]):
        car.state_action_return_list.append([(car.coordinates[0], car.coordinates[1], car.velocity[0], car.velocity[1]), car.action, RETURN_WIN])
        reset_car(car, new_episode=TRUE)

    else:
        car.state_action_return_list.append([(car.coordinates[0], car.coordinates[1], car.velocity[0], car.velocity[1]), car.action, RETURN_LOOSE])
        car.coordinates = [x, y]

        del car.circle

        car.circle = canvas.create_oval(x, y, x + SPACE_SIZE, y + SPACE_SIZE, fill=CAR_COLOR)
        next_action(car)

        window.after(SPEED, next_time_step, car)

def collision(new_coordinates, old_coordinates):

    x, y = new_coordinates
    
    p1 = np.array(new_coordinates) // SPACE_SIZE
    p2 = np.array(old_coordinates) // SPACE_SIZE
    if np.sum(np.abs(p1-p2) <=1):
        points_between = [(p1*SPACE_SIZE).tolist(), (p2*SPACE_SIZE).tolist()]
    else:
        points_between = (connect(np.array([p1,p2]))*SPACE_SIZE).tolist()

    if x < 0 or y < 0 or x >= GAME_WIDTH or y >= GAME_HEIGHT:
        return True
    elif any(check in points_between for check in border.border_cells):
        return True
    else:
        return False
    
def connect(ends):
    d0, d1 = np.diff(ends, axis=0)[0]
    if np.abs(d0) > np.abs(d1): 
        return np.c_[np.arange(ends[0, 0], ends[1,0] + np.sign(d0), np.sign(d0), dtype=np.int32),
                     np.arange(ends[0, 1] * np.abs(d0) + np.abs(d0)//2,
                               ends[0, 1] * np.abs(d0) + np.abs(d0)//2 + (np.abs(d0)+1) * d1, d1, dtype=np.int32) // np.abs(d0)]
    else:
        return np.c_[np.arange(ends[0, 0] * np.abs(d1) + np.abs(d1)//2,
                               ends[0, 0] * np.abs(d1) + np.abs(d1)//2 + (np.abs(d1)+1) * d0, d0, dtype=np.int32) // np.abs(d1),
                     np.arange(ends[0, 1], ends[1,1] + np.sign(d1), np.sign(d1), dtype=np.int32)]


def win(coordinates):

    if coordinates in FINISH:
        return True
    else:
        return False

def reset_car(car, new_episode=False):

    global trials, time_steps, state_action_dic, target_policy
    # evaluate state action history
    for state, action, r in car.state_action_return_list:
        car.G = car.G * GAMMA + r
        if state in state_action_dic:
            state_action_dic[state][action]["C"] += car.weight
            state_action_dic[state][action]["Q"] += car.weight / state_action_dic[state][action]["C"] * (car.G - state_action_dic[state][action]["Q"])

        else:
            state_action_dic[state] = {}
            state_action_dic[state] = {a : {"C": 0, "Q": random.random() * 2 - 1} for a in ACTIONS}
            state_action_dic[state][action]["C"] = car.weight
            state_action_dic[state][action]["Q"] = car.weight / state_action_dic[state][action]["C"] * (car.G - state_action_dic[state][action]["Q"])
        
        target_policy[state] = max(state_action_dic[state], key=lambda d: state_action_dic[state][d]["Q"]) # get best action

        if action != target_policy[state]:
            new_episode = True
            trials -= 1
            break
        else:
            car.weight = car.weight * (1 / (EPSILON/5))


    canvas.delete('car')
    if new_episode:
        car = Car()
        time_steps = 0
        trials += 1
    else:
        car.reset()
    
    window.after(SPEED, next_time_step, car)
    
def next_action(car):
    if car.action == "right":
        car.velocity[0] = car.velocity[0] + 1
    elif car.action == "left":
        car.velocity[0] = int(np.minimum(car.velocity[0]-1,0))
        if (car.velocity[1] == 0): car.velocity[1] = -1 
    elif car.action == "up":
        car.velocity[1] = car.velocity[1] - 1
    elif car.action == "down":
        car.velocity[1] = int(np.maximum(car.velocity[1]+1,0))
        if (car.velocity[0] == 0): car.velocity[0] = 1
    
    # new action
    car.pick_action()

state_action_dic = {} # [[x,y,vx,vy], "x+1"/"x-1"/"y+1"/"y-1"]: {"Q": Q,
                      #                                          "C": C}
target_policy = {} # [x,y,vx,vy] : "x+1"/"x-1"/"y+1"/"y-1"
behaviour_policy = {} # [x,y,vx,vy] : "x+1"/"x-1"/"y+1"/"y-1"

window = Tk()
window.title("Right Turn")
window.resizable(False, False)

# Trials, Time Steps, Velocity

trials = 1
time_steps = 0

# Track Steps
label = Label(window, text="Try:{} ".format(trials)+"Time Steps:{}".format(time_steps), font=('consolas', 20))
label.pack()

# Background
canvas = Canvas(window, bg=BACKGROUND_COLOR, height=GAME_HEIGHT, width=GAME_WIDTH)
canvas.pack()

# Set window to middle
window.update()

window_width = window.winfo_width()
window_height = window.winfo_height()
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()

x = int((screen_width/2) - (window_width/2))
y = int((screen_height/2) - (window_height/2))

window.geometry(f"{window_width}x{window_height}+{x}+{y}")

# create starting line
border = Border()
start = Start()
finish = Finish()
car = Car()

next_time_step(car)

window.mainloop()