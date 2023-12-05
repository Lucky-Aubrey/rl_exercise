import matplotlib.pyplot as plt
import numpy as np

# Define the number of time steps
num_steps = 1000

# Define the initial actions dictionary
actions = {
    0: {'avg': np.linspace(3, 3, num_steps), 'std': 1},
    1: {'avg': np.linspace(-2, 7, num_steps), 'std': 1},
    2: {'avg': np.linspace(4, 1, num_steps), 'std': 1},
}

def takeAction(a, t=0):
    avg = actions[a]['avg'][t]
    std = actions[a]['std']
    return np.random.normal(avg, std)

def testKBandit(epsilon = 0.1, stepsize=None):
    # Estimated value of each action
    Q = [0,0,0]
    # Q = [20,20,20] # Optimistic

    # Keeping track of number of samples for each action
    n = [0,0,0]

    action_history = []
    reward_history = []
    for t in range(1000):

        if np.random.uniform() > epsilon:
            action_chosen = np.argmax(Q)
        else:
            rng = np.random.default_rng()
            action_chosen = rng.choice(list(actions.keys()))

        n[action_chosen] += 1
        action_history.append(action_chosen)
        # reward = takeAction(action_chosen)
        reward = takeAction(action_chosen,t)

        # Update 
        if stepsize is None:
            Q[action_chosen] = Q[action_chosen] + 1/n [action_chosen] * (reward - Q[action_chosen])
        else:
            Q[action_chosen] = Q[action_chosen] + stepsize * (reward - Q[action_chosen])

        reward_history.append(reward)
    return reward_history, action_history

def multipleRuns(epsilon=0.1, stepsize=None):
    # Optimal action policy
    optimal_action = []
    for action, reward in actions.items():
        optimal_action.append(reward['avg'])

    optimal_action = np.array(optimal_action)
    optimal_action = np.argmax(optimal_action, axis=0)

    reward_histories, action_histories = ([], [])
    for i in range(100):
        reward_history, action_history = testKBandit(epsilon=epsilon, stepsize=stepsize)
        reward_histories.append(reward_history)
        action_histories.append(action_history)

    reward_histories = np.array(reward_histories)
    action_histories = np.array(action_histories)

    reward_histories = np.mean(reward_histories, 1)
    correct_actions = np.sum(action_histories == optimal_action[np.newaxis,:], axis=0)/100
    return correct_actions, reward_histories

sample_average_correct_actions,_ = multipleRuns(epsilon=0.1, stepsize=None)
constant_stepsize_correct_actions,_ = multipleRuns(epsilon=0.1, stepsize=1)

# print(f'average reward: {print(np.mean(reward_histories))}')
# plt.plot(reward_histories)
# plt.ylim(0, 5)
plt.plot(sample_average_correct_actions, label='Sample Average')
plt.plot(constant_stepsize_correct_actions, label='Constant Stepsize')
plt.legend(loc='upper right')
plt.ylim(0, 1)
plt.show()