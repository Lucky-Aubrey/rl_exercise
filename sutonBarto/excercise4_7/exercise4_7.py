import numpy as np
from exercise4_7_helper import surface_plot_3d, plot_colormap, policy_evaluation, policy_update, policy_iteration
import pickle
import os

max_days = 100
# rent = 10
# moving_cost = 0
maximum_cars_per_parking_lot = 20
maximum_cars_movable = 5

# Discount
gamma = 0.9

# Initiate random policy
# policy_initial = rng.integers(-5,5,(21,21))
policy_initial = np.zeros((maximum_cars_per_parking_lot+1,maximum_cars_per_parking_lot+1))
# policy_initial = np.ones((21,21)) * 5

# Initiate random values
values_initial = np.zeros((maximum_cars_per_parking_lot+1, maximum_cars_per_parking_lot+1))
# values_initial = np.sum(np.indices((maximum_cars_per_parking_lot+1, maximum_cars_per_parking_lot+1)),axis=0)



if __name__ == "__main__":

    run_name = "excercise_solution"
    file_path = f'{run_name}_policy.pkl'

    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            policy_next = pickle.load(file)
            values_next = policy_evaluation(policy_next, values_initial)

    else:
        policy_next, values_next = policy_iteration(policy_initial, values_initial)

        with open(file_path, 'wb') as file:
            pickle.dump(policy_next, file)

    # create plots and save them
    fig1 = plot_colormap(values_next, fig_title=f'{run_name} - optimal value', show_fig=False)
    fig2 = plot_colormap(policy_next, fig_title=f'{run_name} - optimal policy', show_fig=False)

    fig1.savefig(f'{run_name}_optimal_value.png')
    fig2.savefig(f'{run_name}_optimal_policy.png')

    import matplotlib.pyplot as plt
    plt.show()



    # policy_next, values_next = policy_iteration(values_initial, policy_initial)

    # profit1, hist1 = simulate()
    # profit2, hist2 = simulate(state=(20,20))

    # # Plot
    # plt.figure()
    # plt.plot(profit1)
    # plt.plot(profit2)
    # plt.show()
