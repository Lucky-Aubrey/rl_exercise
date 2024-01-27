import numpy as np
from exercise4_7_helper import surface_plot_3d, plot_colormap, policy_evaluation, policy_update, policy_iteration
import pickle
import os

max_days = 100
rent = 10
moving_cost = 2
maximum_cars_per_parking_lot = 20
maximum_cars_movable = 5

# Discount
gamma = 0.9

# Initiate random policy
# policy_initial = rng.integers(-5,5,(21,21))
policy_initial = np.zeros((maximum_cars_per_parking_lot+1,maximum_cars_per_parking_lot+1))
# policy_initial = np.ones((21,21)) * 5

# Initiate random values
values_initial = np.sum(np.indices((maximum_cars_per_parking_lot+1, maximum_cars_per_parking_lot+1)),axis=0)



if __name__ == "__main__":

    # values_next = policy_evaluation(values_initial, policy_initial)
    # policy_next, _ = policy_update(policy_initial, values_initial)

    policy_next, values_next = policy_iteration(policy_initial, values_initial)

    run_name = "standard"
    file_path = f'{run_name}_policy.pkl'
    with open(file_path, 'wb') as file:
        pickle.dump(policy_next, file)

    fig1 = plot_colormap(values_next, fig_title=f'{run_name} - optimal value', show_fig=False)
    fig2 = plot_colormap(policy_next, fig_title=f'{run_name} - optimal policy', show_fig=False)

    fig1.savefig(f'{run_name}_optimal_value.png')
    fig2.savefig(f'{run_name}_optimal_policy.png')

    # values_next = policy_evaluation(values_next, policy_next)
    # policy_next, _ = policy_update(policy_next, values_next)



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
