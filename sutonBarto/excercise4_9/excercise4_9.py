import numpy as np
import matplotlib.pyplot as plt

money_for_win = 100

values_initial = np.zeros(money_for_win+1)
values_initial[-1] = 0

policy_initial = np.zeros(money_for_win-1) # 1...99

p_h = 0.1

def r(state, action, win):
    if win:
        return state + action >= money_for_win
    else:
        return -(state - action <= 0)

def value_iteration(value_table):
    delta = 100
    theta = 0.001
    gamma = 1
    values_updated = value_table.copy()
    policy = np.zeros(money_for_win+1) # 1...99
    
    counter = 0
    while delta > theta:
        delta = 0
        counter += 1
        for state in range(1,money_for_win):
            v = values_updated[state]
            values_updated[state]
            best_G = -10000
            for action in range(0, min(state,money_for_win-state)+1):
                if (state == 6) and counter == 5:
                    print("")
                state_after_win = np.minimum(state + action, money_for_win)
                state_after_loss = np.maximum(state - action, 0)
                # state_after_loss = 0

                G = p_h * (r(state=state,action=action,win=True) + gamma * values_updated[state_after_win]) \
                + (1 - p_h) * (r(state=state,action=action,win=False) + gamma * values_updated[state_after_loss]) 
                # G = p_h * (r(state=state,action=action,win=True) + gamma * values_updated[state_after_win]) \
                # + (1 - p_h) * (-1 + gamma * values_updated[state_after_loss]) 

                if G > best_G:
                    best_G = G
            
            values_updated[state] = best_G

            delta = np.maximum(delta, np.abs(v-values_updated[state]))
        print(f'value iteration -- counter: {counter}',f'delta: {delta}')
        # print(np.round(values_updated.copy(),2))

    for state in range(1,money_for_win):
        best_G = -10000
        for action in range(0, min(state,money_for_win-state)+1):
            state_after_win = np.minimum(state + action, money_for_win)
            state_after_loss = np.maximum(state - action, 0)

            G = p_h * (r(state=state,action=action,win=True) + gamma * values_updated[state_after_win]) \
            + (1 - p_h) * (r(state=state,action=action,win=False + gamma * values_updated[state_after_loss])) 

            if G > best_G:
                best_G = G
                policy[state] = action

    return policy, values_updated

def plot_policy(policy):
    
    plt.plot(policy)
    plt.show()

if __name__ == "__main__":

    policy, values = value_iteration(value_table=values_initial)

    plot_policy(policy)
    plot_policy(values)