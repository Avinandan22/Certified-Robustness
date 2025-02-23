from utils import *
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import multiprocessing as mp

# np.random.seed(0)

epsilon = 0.15
delta = 0.0
eta = 1e-2
eta_list = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3]
# kappa_list = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
kappa_list = [1.0]
epsilon_list = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

# Example parameters for the NIW prior
d = 2  # Dimensionality of the multivariate Gaussian
mean_prior = np.zeros(d)  # Prior mean vector
cov_prior = 0.1 * np.eye(d)  # Prior covariance matrix
df_prior = d + 2  # Prior degrees of freedom
scale_prior = np.eye(d)  # Prior scale matrix

# Combine parameters into a tuple
niw_params = (mean_prior, cov_prior, df_prior, scale_prior)

# Number of samples to generate
K = 10

# Generate samples from the NIW prior
means, covariances = generate_niw_prior_samples(d, niw_params, K)
# means = [np.zeros(d) for i in range(K)]
# Generate test samples from the NIW prior
K_test = 2
means_test, covariances_test = generate_niw_prior_samples(d, niw_params, K_test)
# means_test = [np.zeros(d) for i in range(K_test)]

runs = 5

for epsilon in epsilon_list:
    for eta in eta_list:
        print("epsilon : ", epsilon)
        # print("eta : ", eta)
        B_list = []
        for kappa in kappa_list:
            S = alt_min_mean_estimation_bound_meta(
                means, covariances, eta, epsilon, kappa=kappa, delta=0.0
            )
            print("Is the matrix positive definite?", is_positive_definite(S))
            B, _ = BBT_decomposition(S)
            B_list.append(B)

        print("Train Performance")

        defense_loss_list_train = {kappa: [] for kappa in kappa_list}
        no_defense_loss_list_train = []

        for i in range(K):
            mu = means[i]
            Sigma = covariances[i]

            defense_losses = []
            for i in range(len(kappa_list)):
                B = B_list[i]
                kappa = kappa_list[i]

                def calculate_loss_for_run(run_index):
                    theta = gradient_descent(
                        mu,
                        Sigma,
                        epsilon,
                        eta,
                        B,
                        max_iterations=2000,
                        convergence_threshold=1e-5,
                    )
                    # print(theta)
                    return np.linalg.norm(mu - theta) ** 2

                # Initialize a pool of processes
                pool = mp.Pool(
                    mp.cpu_count()
                )  # Use as many processes as there are CPU cores

                # Use a list comprehension to map the function over the range of runs
                loss = pool.map(calculate_loss_for_run, range(runs))

                # Close the pool to release resources
                pool.close()
                defense_loss_list_train[kappa].append(np.mean(loss))

            def calculate_loss_for_run(run_index):
                theta = gradient_descent(
                    mu,
                    Sigma,
                    epsilon,
                    eta,
                    np.zeros(d),
                    max_iterations=2000,
                    convergence_threshold=1e-5,
                )
                return np.linalg.norm(mu - theta)

            # Initialize a pool of processes
            pool = mp.Pool(
                mp.cpu_count()
            )  # Use as many processes as there are CPU cores

            # Use a list comprehension to map the function over the range of runs
            no_defense_loss = pool.map(calculate_loss_for_run, range(runs))

            # Close the pool to release resources
            pool.close()
            no_defense_loss_list_train.append(np.mean(no_defense_loss))

        for kappa in kappa_list:
            # print(defense_loss_list_train[kappa])
            print(
                f"Train - Kappa: {kappa}, Avg Loss: {np.mean(defense_loss_list_train[kappa])}"
            )
            print(
                f"Train - Kappa: {kappa}, Max Loss: {np.max(defense_loss_list_train[kappa])}"
            )

        print("Train - Avg Loss no defense: ", np.mean(no_defense_loss_list_train))
        print("Train - Max Loss no defense: ", np.max(no_defense_loss_list_train))

        # Save train losses as numpy files
        for kappa in kappa_list:
            np.save(
                f"defense_loss_train_epsilon_{epsilon}_eta_{eta}_kappa_{kappa}_d_{d}.npy",
                np.array(defense_loss_list_train[kappa]),
            )
        np.save(
            f"no_defense_loss_train_epsilon_{epsilon}_eta_{eta}_d_{d}.npy",
            np.array(no_defense_loss_list_train),
        )

        print("Test Performance")

        defense_loss_list_test = {kappa: [] for kappa in kappa_list}
        no_defense_loss_list_test = []

        for i in range(K_test):
            # def evaluate(i):
            print(i)
            mu = means_test[i]
            Sigma = covariances_test[i]
            defense_losses = []
            for i in range(len(kappa_list)):
                B = B_list[i]
                kappa = kappa_list[i]

                def calculate_loss_for_run(run_index):
                    theta_list = gradient_descent_stationary(
                        mu,
                        Sigma,
                        epsilon,
                        eta,
                        B,
                        max_iterations=500,
                        convergence_threshold=1e-5,
                    )
                    errors = np.mean(
                        [np.linalg.norm(mu - theta) ** 2 for theta in theta_list]
                    )
                    # return np.linalg.norm(mu - theta)**2
                    return errors

                # Initialize a pool of processes
                pool = mp.Pool(
                    mp.cpu_count()
                )  # Use as many processes as there are CPU cores

                # Use a list comprehension to map the function over the range of runs
                loss = pool.map(calculate_loss_for_run, range(runs))

                # Close the pool to release resources
                pool.close()
                defense_loss_list_test[kappa].append(np.mean(loss))
                # defense_losses.append(loss)

            def calculate_loss_for_run(run_index):
                theta_list = gradient_descent_stationary(
                    mu,
                    Sigma,
                    epsilon,
                    eta,
                    B,
                    max_iterations=500,
                    convergence_threshold=1e-5,
                )
                errors = np.mean(
                    [np.linalg.norm(mu - theta) ** 2 for theta in theta_list]
                )
                # return np.linalg.norm(mu - theta)**2
                return errors

            # Initialize a pool of processes
            pool = mp.Pool(
                mp.cpu_count()
            )  # Use as many processes as there are CPU cores

            # Use a list comprehension to map the function over the range of runs
            no_defense_loss = pool.map(calculate_loss_for_run, range(runs))

            # Close the pool to release resources
            pool.close()
            no_defense_loss_list_test.append(np.mean(no_defense_loss))

        for kappa in kappa_list:
            print(
                f"Test - Kappa: {kappa}, Avg Loss: {np.mean(defense_loss_list_test[kappa])}"
            )
            print(
                f"Test - Kappa: {kappa}, Max Loss: {np.max(defense_loss_list_test[kappa])}"
            )

        print("Test - Avg Loss no defense: ", np.mean(no_defense_loss_list_test))
        print("Test - Max Loss no defense: ", np.max(no_defense_loss_list_test))

        # Save test losses as numpy files
        for kappa in kappa_list:
            np.save(
                f"logs/defense_loss_test_epsilon_{epsilon}_eta_{eta}_kappa_{kappa}_d_{d}.npy",
                np.array(defense_loss_list_test[kappa]),
            )
        np.save(
            f"logs/no_defense_loss_test_epsilon_{epsilon}_eta_{eta}_d_{d}.npy",
            np.array(no_defense_loss_list_test),
        )
