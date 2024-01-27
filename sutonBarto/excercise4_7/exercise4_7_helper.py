import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.stats import poisson
import pickle
import os


maximum_cars_per_parking_lot = 20
rent = 10
moving_cost = 2
parking_cost = 4
gamma = 0.9
theta = 1
lambda_returned_1 = 3
lambda_returned_2 = 2
lambda_requested_1 = 3
lambda_requested_2 = 4
# x = np.transpose(np.indices((21,21,21,21)), axes=(1,2,3,4,0))
# x = np.transpose(np.indices((10,10,10,10)), axes=(1,2,3,4,0))
x = np.transpose(np.indices((8,8,8,8)), axes=(1,2,3,4,0))

# all to first parking lot
def policy_1(cars, maximum_cars_movable):
    move_cars = min(cars[1],maximum_cars_movable)
    return np.array((cars[0]+move_cars,cars[1]-move_cars)), move_cars
    

def plot_colormap(data, fig_title='2D Array Plot', show_fig=True):
    fig, ax = plt.subplots()
    # Use imshow on the Axes object (ax)
    img = ax.imshow(data, cmap='viridis', origin='lower', interpolation='none')
    fig.colorbar(img)  # Add a colorbar to the plot
    ax.set_title(fig_title)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    if show_fig:
        fig.show()
    return fig

def surface_plot_3d(data):

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create a grid of X and Y values
    x, y = np.meshgrid(np.arange(data.shape[0]), np.arange(data.shape[1]))

    # Plot the 3D surface
    ax.plot_surface(x, y, data, cmap='viridis')

    # Add labels
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    plt.title('3D Surface Plot')
    plt.show()

def state_transition(action, state, maximum_cars_per_parking_lot):
    if action >= 0:
        action = min(maximum_cars_per_parking_lot - state[0], state[1], action)
    else:
        action = -min(state[0], maximum_cars_per_parking_lot - state[1], -action)

    return state + np.array((action, -action)), action

def expected_return(state, action, value_table):
        # state transition of moving cars
    if action >= 0:
        action = min(maximum_cars_per_parking_lot - state[0], state[1], action)
    else:
        action = -min(state[0], maximum_cars_per_parking_lot - state[1], -action)

    state_after_move = state + np.array((action, -action))
    
    # State after cars are rented out
    state_after_rent = np.maximum(state_after_move - x[:,:,:,:,:2], np.zeros(2))
    
    # Reward as in exercise
    r = np.sum(state_after_move - state_after_rent, axis=-1) * rent \
        - np.abs(np.max(action-1,0)) * moving_cost \
        - np.abs(np.min(action,0)) * moving_cost \
        - np.sum(np.maximum(np.minimum(np.ones(2),state_after_rent-np.ones(2)*10), np.zeros(2)), axis=-1) * parking_cost
    
    # # Reward for moving and renting cars, with first 1 car moved for free to site A
    # r = np.sum(state_after_move - state_after_rent, axis=-1) * rent \
    #     - np.abs(np.max(action-1,0)) * moving_cost \
    #     - np.abs(np.min(action,0)) * moving_cost
    
    # # Standard problem
    # r = np.sum(state_after_move - state_after_rent, axis=-1) * rent \
        # - np.abs(action) * moving_cost 

    # State after cars are returned
    state_after_return = np.minimum(state_after_rent + x[:,:,:,:,2:],np.array((maximum_cars_per_parking_lot,maximum_cars_per_parking_lot)))
    
    return np.sum(poisson.pmf(x[...,0], lambda_returned_1)\
        *poisson.pmf(x[...,1], lambda_returned_2)\
        *poisson.pmf(x[...,2], lambda_requested_1)\
        *poisson.pmf(x[...,3], lambda_requested_2)\
        *(r+gamma*value_table[state_after_return[:,:,:,:,0].astype(int),state_after_return[:,:,:,:,1].astype(int)]))

def policy_evaluation(policy, value_table):


    delta = 100000  
    new_value_table = value_table.copy()
    while delta > theta:
        delta = 0
        counter = 0
        for i in range(0,maximum_cars_per_parking_lot+1):
            for j in range(0,maximum_cars_per_parking_lot+1):
                v = new_value_table[i,j]
                state = np.array((i,j))
                action = policy[i,j]

                new_value_table[i,j] = expected_return(state, action, new_value_table)
                
                delta = np.maximum(delta, np.abs(v-new_value_table[i,j]))

                counter +=1
                print(f'policy evaluation -- counter: {counter}',f'delta: {delta}', f'v: {new_value_table[i,j]}')

    return new_value_table
    
def policy_update(policy, value_table):
    new_policy = policy.copy()
    policy_stable = True
    counter = 0
    for i in range(0,maximum_cars_per_parking_lot+1):
        for j in range(0,maximum_cars_per_parking_lot+1):

            state = np.array((i,j))

            old_action = new_policy[i,j]

            G_best = 0 # return
            for action in range(max(max(-5,-i),-(maximum_cars_per_parking_lot-j)),
                                min(min(6,j+1),maximum_cars_per_parking_lot-i+1)):
                
                G = expected_return(state, action, value_table)
                
                if G > G_best:
                    G_best = G
                    new_policy[i,j] = action

            if new_policy[i,j] != old_action:
                policy_stable = False

            counter += 1
            print(f'policy update -- counter: {counter}',f'policy_stable: {policy_stable}')

    return new_policy, policy_stable

def policy_iteration(value_table, policy):
    values_next = value_table.copy()
    policy_next = policy.copy()
    policy_stable = False
    counter = 0
    while not policy_stable and counter < 100:
        values_next = policy_evaluation(policy_next, values_next)
        policy_next, policy_stable = policy_update(policy_next, values_next)
        counter += 1
        print(f'------ {counter}. iteration ------')
    
    return policy_next, values_next

# Simulate (outdated)
def simulate(action=None, state=(0,0)):

    # generate poisson numbers
    # Create a single random number generator
    rng = np.random.default_rng()

    # Generate random numbers using the same generator
    cars_requested = np.array([rng.poisson(3, 100), rng.poisson(4, 100)])
    cars_returned = np.array([rng.poisson(3, 100), rng.poisson(2, 100)])

    all_parking_lot_cars = np.array(state)

    days = list(range(max_days))
    profit = [0]
    parking_hist = []
    parking_hist.append(all_parking_lot_cars.copy())
    for d in days:
        # Actions
        # all to first parking lot
        if action:
            all_parking_lot_cars, cars_moved = action(all_parking_lot_cars, maximum_cars_movable)
            profit.append(profit[-1] - cars_moved * moving_cost)

        # Rentout to customer
        profit_per_day = 0
        for p in range(all_parking_lot_cars.shape[0]):
            requested = cars_requested[p,d]
            available = all_parking_lot_cars[p]
            returned = cars_returned[p,d]

            if requested <= available:
                all_parking_lot_cars[p] -= requested
                profit_per_day += requested * rent
            else:
                all_parking_lot_cars[p] = 0
                profit_per_day += available * rent
            
            all_parking_lot_cars[p] = min(all_parking_lot_cars[p]+returned,20)
        profit.append(profit_per_day + profit[-1])
        parking_hist.append(all_parking_lot_cars.copy())
    return profit, np.array(parking_hist)

# def test_arr():
#     arr = np.zeros((maximum_cars_per_parking_lot+1,maximum_cars_per_parking_lot+1))
#     for i in range(0,maximum_cars_per_parking_lot+1):
#         for j in range(0,maximum_cars_per_parking_lot+1):
#             action_list = []
#             for action in range(max(max(-5,-i),-(maximum_cars_per_parking_lot-j)),
#                                 min(min(6,j+1),maximum_cars_per_parking_lot-i+1)):
#                 arr[i,j] = action if (np.abs(action) > np.abs(arr[i,j])) else arr[i,j]
#                 # action_list.append(action)
#             # print(f'{(i,j)}',action_list)
#     plt.imshow(arr, origin='lower')
#     plt.show()

# test_arr()
