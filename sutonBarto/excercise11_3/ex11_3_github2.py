import numpy as np
W = np.ones((8,1))
W[-2] = 10
alpha = 0.1
gamma = 0.99
state_encode = {
    1:[2,0,0,0,0,0,0,1],
    2:[0,2,0,0,0,0,0,1],
    3:[0,0,2,0,0,0,0,1],
    4:[0,0,0,2,0,0,0,1],
    5:[0,0,0,0,2,0,0,1],
    6:[0,0,0,0,0,2,0,1],
    7:[0,0,0,0,0,0,1,2],
}
SA_dict = {}
def SA(state, action):
    global SA
    if (state,action) in SA_dict:
        # memoization
        return SA_dict[(state,action)]
    else:
        feature = np.array(state_encode[state])
        feature = np.reshape(feature, (-1,1))
        SA_dict[(state,action)] = feature
        return feature

def Q(state, action):
    # 0:dashed 1:solid
    return W.T@SA(state,action)

def b():
    dice = np.random.random()
    # 0 means dashed, 1 means solid
    if dice > 1/7:
        return 0, np.random.randint(low=1, high=6)
    else:
        return 1, 7

def update(state, next_state, action):
    global W
    # delta = reward(=0) + gamma * greedyQ(S+t+1, greedy action. w) -  Q(S,a,w)
    # Because Baird's special setting, we only need to care the destination of action
    # to determine the Q, which should be equal to the next state value(one step)
    delta = gamma*max(Q(next_state, 0),Q(next_state,1)) - Q(state,action)
    W = W + alpha*delta*SA(state, action)

def game():
    global W
    hist = []
    for episode in range(0,1):
        _,state = b()
        for t in range(0,1000):
            action, next_state = b()
            update(state, next_state, action)
            state = next_state
            hist.append(W)
    return hist

hist = game()

import matplotlib.pyplot as plt
hist = np.array(hist).squeeze()
plt.plot(hist)
plt.legend(list(range(9)))
plt.show()