# Ensure compatibility with Python 2
from __future__ import absolute_import, division, print_function, unicode_literals
from scipy.stats import invwishart, multivariate_normal

from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

import numpy as np

# import trustregion
# import torch
# import torch.optim as optim
import cvxpy as cp

from functools import partial


def block_diagonal(matrices):
    # Create block diagonal matrix
    block_diag_matrix = np.block(matrices)
    return block_diag_matrix


def is_negative_semidefinite(matrix):
    # Perform eigenvalue decomposition
    eigvals, _ = np.linalg.eigh(matrix)

    # Check if all eigenvalues are non-positive
    return np.all(eigvals <= 0)


import cvxpy as cp


def inf_1(mu, Sigma, S, eta, epsilon, kappa=0.1, delta=0.0):
    d = mu.shape[0]
    Sigma = Sigma.reshape((d, d))
    I = np.eye(d)
    c = (1 + delta) ** 2 * d

    # Define variables
    # Min over A, b, nu
    # S = cp.Variable((d, d), PSD=True)
    A = cp.Variable((d, d), symmetric=True)
    b = cp.Variable(d)
    nu = cp.Variable(1)

    D_1 = ((1 - eta) ** 2 - 1) * A + I
    D_2 = 2 * eta * epsilon * (1 - eta) * A
    d_3 = 2 * (1 - epsilon) * eta * (1 - eta) * A @ mu - 2 * mu - eta * b
    d_4 = (
        (1 - epsilon)
        * (eta**2 * cp.trace(Sigma @ A) + eta**2 * mu.T @ A @ mu + eta * b.T @ mu)
        + mu.T @ mu
        + eta**2 * cp.trace(A @ S)
    )
    # + eta**2 * (cp.norm(A) ** 2 +  cp.norm(S) ** 2) * d * 0.5
    # + eta**2 * cp.trace(A @ S) # Using an upper bound on this term instead since this is not DCP compatible

    # Define objective function

    E = -1.0 * cp.bmat([[D_1, D_2 / 2], [D_2 / 2, epsilon * eta**2 * A + nu * I]])
    e = cp.hstack([d_3, epsilon * eta * b - 2 * nu * mu])
    # print(E.shape)
    # print(e.shape)
    obj = cp.Minimize(
        eta**2 * cp.trace(S)
        + kappa * (0.25 * cp.matrix_frac(e, E) + d_4 + nu * (mu.T @ mu - c))
    )
    # Define constraints
    constraints = [E >> 0, nu <= 0]
    # Define and solve the problem
    problem = cp.Problem(obj, constraints)
    problem.solve(max_iters=1000000)

    # Return optimal values
    return {
        "optimal_value": problem.value,
        "optimal_A": A.value,
        "optimal_b": b.value,
        "optimal_nu": nu.value,
    }


def inf_2(mu, Sigma, A, b, nu, eta, epsilon, kappa=0.1, delta=0.0):
    d = mu.shape[0]
    Sigma = Sigma.reshape((d, d))
    I = np.eye(d)
    c = (1 + delta) ** 2 * d

    # Define variables
    # Min over S
    S = cp.Variable((d, d), PSD=True)

    D_1 = ((1 - eta) ** 2 - 1) * A + I
    D_2 = 2 * eta * epsilon * (1 - eta) * A
    d_3 = 2 * (1 - epsilon) * eta * (1 - eta) * A @ mu - 2 * mu - eta * b
    d_4 = (
        (1 - epsilon)
        * (eta**2 * cp.trace(Sigma @ A) + eta**2 * mu.T @ A @ mu + eta * b.T @ mu)
        + mu.T @ mu
        + eta**2 * cp.trace(A @ S)
    )

    # Define objective function

    E = -1.0 * cp.bmat([[D_1, D_2 / 2], [D_2 / 2, epsilon * eta**2 * A + nu * I]])
    e = cp.hstack([d_3, epsilon * eta * b - 2 * nu * mu])
    obj = cp.Minimize(
        eta**2 * cp.trace(S)
        + kappa * (0.25 * cp.matrix_frac(e, E) + d_4 + nu * (mu.T @ mu - c))
    )
    # Define constraints
    # constraints = [E >> 0, nu <= 0]
    constraints = []
    # Define and solve the problem
    problem = cp.Problem(obj, constraints)
    problem.solve(max_iters=1000000)

    # Return optimal values
    return {"optimal_value": problem.value, "optimal_S": S.value}


def inf_2_meta(
    mu_list, Sigma_list, A_list, b_list, nu_list, eta, epsilon, kappa=0.1, delta=0.0
):

    d = mu_list[0].shape[0]
    # Sigma = Sigma.reshape((d, d))
    I = np.eye(d)
    c = (1 + delta) ** 2 * d

    # Define variables
    # Min over S
    S = cp.Variable((d, d), PSD=True)

    constraints = []
    n = len(A_list)
    avg_obj = 0
    for i in range(n):

        A = A_list[i]
        b = b_list[i]
        nu = nu_list[i]
        mu = mu_list[i]
        Sigma = Sigma_list[i]

        D_1 = ((1 - eta) ** 2 - 1) * A + I
        D_2 = 2 * eta * epsilon * (1 - eta) * A
        d_3 = 2 * (1 - epsilon) * eta * (1 - eta) * A @ mu - 2 * mu - eta * b
        d_4 = (
            (1 - epsilon)
            * (eta**2 * cp.trace(Sigma @ A) + eta**2 * mu.T @ A @ mu + eta * b.T @ mu)
            + mu.T @ mu
            + eta**2 * cp.trace(A @ S)
        )

        # Define objective function

        E = -1.0 * cp.bmat([[D_1, D_2 / 2], [D_2 / 2, epsilon * eta**2 * A + nu * I]])
        e = cp.hstack([d_3, epsilon * eta * b - 2 * nu * mu])
        obj = cp.Minimize(
            eta**2 * cp.trace(S)
            + kappa * (0.25 * cp.matrix_frac(e, E) + d_4 + nu * (mu.T @ mu - c))
        )
        avg_obj += obj

    avg_obj = obj / n
    # Define and solve the problem
    problem = cp.Problem(obj, constraints)
    problem.solve(max_iters=1000000)

    # Return optimal values
    return {"optimal_value": problem.value, "optimal_S": S.value}


def alt_min_mean_estimation_bound(mu, Sigma, eta, epsilon, kappa=0.1, delta=0.0):

    d = mu.shape[0]
    # Sigma = Sigma.reshape((d, d))
    I = np.eye(d)
    c = (1 + delta) ** 2 * d
    S = np.eye(d)

    for t in range(5):
        # Define variables
        # Min over A, b, nu
        X = inf_1(mu, Sigma, S, eta, epsilon, kappa=0.1, delta=0.0)
        A = X["optimal_A"]
        b = X["optimal_b"]
        nu = X["optimal_nu"]
        X = inf_2(mu, Sigma, A, b, nu, eta, epsilon, kappa=0.1, delta=0.0)
        S = X["optimal_S"]
        opt = X["optimal_value"]
        print("T : ", t + 1, "obj : ", opt)

    # Return optimal values
    return S


import concurrent.futures


def execute_inf_1(args):
    # Helper function to unpack arguments and execute f_1
    return inf_1(*args)


def alt_min_mean_estimation_bound_meta(
    mu_list, Sigma_list, eta, epsilon, kappa=0.1, delta=0.0
):

    d = mu_list[0].shape[0]
    # Sigma = Sigma.reshape((d, d))
    I = np.eye(d)
    c = (1 + delta) ** 2 * d
    S = np.eye(d)
    T = 10
    for t in range(T):
        # Define variables
        # Min over A, b, nu
        A_list = []
        b_list = []
        nu_list = []
        n = len(mu_list)

        X_list = []
        for i in range(n):
            # print(i)
            mu = mu_list[i]
            Sigma = Sigma_list[i]
            X_list.append(inf_1(mu, Sigma, S, eta, epsilon, kappa, delta))

        for i in range(n):
            A_list.append(X_list[i]["optimal_A"])
            b_list.append(X_list[i]["optimal_b"])
            nu_list.append(X_list[i]["optimal_nu"])

        X = inf_2_meta(
            mu_list, Sigma_list, A_list, b_list, nu_list, eta, epsilon, kappa, delta
        )
        S = X["optimal_S"]
        opt = X["optimal_value"]
        print("T : ", t + 1, "obj : ", opt)

    # Return optimal values
    return S


def mean_estimation_meta_defense(
    means, covariances, K, d, eta, epsilon, kappa=0.1, delta=0.0
):

    # print(cp.installed_solvers())
    # Define variables
    S = cp.Variable((d, d), symmetric=True)
    # S = cp.Variable((d, d), PSD=True)
    A_list = [cp.Variable((d, d), symmetric=True) for i in range(K)]
    b_list = [cp.Variable(d) for i in range(K)]
    # nu_list = [cp.Variable(1) for i in range(K)]
    nu = cp.Variable(K)
    I = np.eye(d)
    c = (1 + delta) ** 2 * d
    avg_obj = 0
    constraints = [nu <= 0]
    for i in range(K):
        # print(i)
        mu = means[i]
        # mu = mu.reshape((d, 1))
        Sigma = covariances[i]
        Sigma = Sigma.reshape((d, d))

        A = A_list[i]
        b = b_list[i]
        # nu = nu_list[i]

        D_1 = ((1 - eta) ** 2 - 1) * A + I
        D_2 = 2 * eta * epsilon * (1 - eta) * A
        d_3 = 2 * (1 - epsilon) * eta * (1 - eta) * A @ mu - 2 * mu - eta * b
        d_4 = (
            (1 - epsilon)
            * (eta**2 * cp.trace(Sigma @ A) + eta**2 * mu.T @ A @ mu + eta * b.T @ mu)
            + mu.T @ mu
            + eta**2 * (cp.norm(A, "fro") ** 2 + cp.norm(S, "fro") ** 2) / 2
        )
        # + eta**2 * (cp.norm(A) ** 2 +  cp.norm(S) ** 2) * d * 0.5
        # + eta**2 * cp.trace(A @ S) # Using an upper bound on this term instead since this is not DCP compatible

        # Define objective function

        E = -1.0 * cp.bmat(
            [[D_1, D_2 / 2], [D_2 / 2, epsilon * eta**2 * A + nu[i] * I]]
        )
        e = cp.hstack([d_3, epsilon * eta * b])
        constraints.append(E >> 0.000001 * np.eye(2 * d))
        # constraints.append(nu <= 0)

        avg_obj = avg_obj + (0.25 * cp.matrix_frac(e, E) + d_4 - nu[i] * c)
    avg_obj = avg_obj / K
    # obj = cp.Minimize(eta**2 * cp.trace(S)+ kappa * (0.25 * cp.matrix_frac(e, E) + d_4 - nu * c))
    obj = cp.Minimize(eta**2 * cp.trace(S) + kappa * avg_obj)
    constraints.append(S >> 0.000001 * np.eye(d))
    # Define constraints
    # constraints = [E >> 0]
    # Define and solve the problem
    problem = cp.Problem(obj, constraints)
    problem.solve(solver=cp.SCS, max_iters=100000)
    # problem.solve(verbose=True)

    # Return optimal values
    return {
        "optimal_value": problem.value,
        "optimal_S": S.value,
        "optimal_nu": nu.value,
    }


def is_positive_definite(matrix):
    """
    Check if a matrix is positive definite.

    Parameters:
    matrix (ndarray): Input matrix

    Returns:
    bool: True if the matrix is positive definite, False otherwise.
    """
    # Check if the matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        return False

    # Check if all eigenvalues are positive
    eigenvalues = np.linalg.eigvals(matrix)
    return np.all(eigenvalues > 0)


from scipy.optimize import minimize


def gradient(z, eta, epsilon, theta, mu):
    grad = 2 * eta**2 * epsilon**2 * z + 2 * eta * epsilon * (
        (1 - eta) * theta + eta * (1 - epsilon) * mu
    )
    return grad


def project(z, c):
    norm_z = np.linalg.norm(z)
    # print(norm_z)
    if norm_z <= np.sqrt(c):
        return z
    else:
        return z * np.sqrt(c) / norm_z


def optimize_z_gradient(
    eta, epsilon, theta, mu, c, learning_rate=1.0, max_iter=500, tolerance=1e-5
):
    # Initialize z randomly
    z = np.random.rand(len(theta))

    # Project z onto the feasible set
    z = project(z, c)
    # print("z init : ", z)

    # Perform gradient descent
    for i in range(max_iter):
        # Compute the gradient of the objective function
        grad = gradient(z, eta, epsilon, theta, mu)
        # print("grad : ", grad)

        # Update z using the gradient descent step
        z_new = z + learning_rate * grad

        # Project z_new onto the feasible set
        z_new = project(z_new, c)

        # Check for convergence
        if np.linalg.norm(z_new - z) < tolerance:
            break

        z = z_new

    return z


def BBT_decomposition(S):
    # Eigenvalue decomposition of the PSD matrix S
    eig_vals, eig_vecs = np.linalg.eigh(S)

    # Square root of the eigenvalues
    sqrt_eig_vals = np.sqrt(np.maximum(eig_vals, 0))  # Ensure non-negative eigenvalues

    # Construct the lower triangular matrix L
    L = np.dot(eig_vecs, np.diag(sqrt_eig_vals))

    # Compute the conjugate transpose of L
    L_conj_transpose = np.conj(L.T)

    return L, L_conj_transpose


# Gradient descent function
def gradient_descent(
    mu,
    Sigma,
    epsilon,
    eta,
    B,
    delta=0.0,
    max_iterations=1000,
    convergence_threshold=1e-5,
):
    # Initialize theta randomly
    np.random.seed()
    theta = np.random.rand(mu.shape[0])
    d = mu.shape[0]
    # d = theta.shape[0]
    c = (1 + delta) ** 2 * d
    # theta = np.random.multivariate_normal(np.zeros(d), 0.1 * np.eye(d))
    # theta = np.zeros(d)

    # Initialize a variable for the previous theta values
    prev_theta = theta.copy()
    # print(theta)
    # Perform gradient descent
    for i in range(max_iterations):
        # Optimize z_adv
        # z_adv = optimize_z(epsilon, eta, theta, mu)
        # z_adv = optimize_z(eta, epsilon, theta, mu, c)
        # z_adv = optimize_z_cvxpy(eta, epsilon, theta, mu, c)
        z_adv = optimize_z_gradient(eta, epsilon, theta, mu, c)
        d = mu.shape[0]
        Sigma = Sigma.reshape((d, d))
        # Sample z and w from their respective distributions
        z = np.random.multivariate_normal(mu, Sigma)
        # print("Iter : ", i)
        # print("z_adv : ", z_adv)
        # print("theta : ", theta)
        # w = np.random.normal(0, 1, len(B))
        w = np.random.multivariate_normal(np.zeros(d), np.eye(d))

        # Update theta using the given formula
        theta = (
            (1 - eta) * theta
            + eta * epsilon * z_adv
            + eta * (1 - epsilon) * z
            - eta * np.dot(B, w)
        )
        # theta = (1 - eta) * theta + eta * epsilon * z_adv + eta * (1 - epsilon) * mu - eta * np.dot(B, w)

        # # Check for convergence
        # if np.linalg.norm(theta - prev_theta) < convergence_threshold:
        #     print("Converged after", i, "iterations.")
        #     break

        # Update prev_theta for the next iteration
        prev_theta = theta.copy()

    return theta


# Gradient descent function
def gradient_descent_stationary(
    mu,
    Sigma,
    epsilon,
    eta,
    B,
    delta=0.0,
    max_iterations=1000,
    convergence_threshold=1e-5,
):
    # Initialize theta randomly
    np.random.seed()
    theta = np.random.rand(mu.shape[0])
    d = mu.shape[0]
    c = (1 + delta) ** 2 * d

    # Initialize a variable for the previous theta values
    prev_theta = theta.copy()
    theta_list = []
    # Perform gradient descent
    for i in range(max_iterations):
        # Optimize z_adv
        z_adv = optimize_z_gradient(eta, epsilon, theta, mu, c)
        d = mu.shape[0]
        Sigma = Sigma.reshape((d, d))
        # Sample z and w from their respective distributions
        z = np.random.multivariate_normal(mu, Sigma)
        w = np.random.multivariate_normal(np.zeros(d), np.eye(d))

        # Update theta using the given formula
        theta = (
            (1 - eta) * theta
            + eta * epsilon * z_adv
            + eta * (1 - epsilon) * z
            - eta * np.dot(B, w)
        )

        if i > 0.9 * max_iterations:
            theta_list.append(theta)

        # Update prev_theta for the next iteration
        prev_theta = theta.copy()

    return theta_list


def generate_random_covariance_matrix(d):
    # Generate a random matrix with dimensions d x d
    A = np.random.randn(d, d)

    # Make the matrix symmetric
    A = np.triu(A) + np.triu(A, 1).T

    # Ensure the matrix is positive definite
    eig_vals, eig_vecs = np.linalg.eigh(A)
    A = np.dot(np.dot(eig_vecs, np.diag(np.maximum(eig_vals, 0))), eig_vecs.T)

    return A


def generate_niw_prior_samples(d, niw_params, K):
    """
    Generate samples from Normal-Inverse-Wishart (NIW) prior distribution.

    Parameters:
        d (int): Dimensionality of the multivariate Gaussian.
        niw_params (tuple): Parameters of the NIW distribution in the form (mean_prior, cov_prior, df_prior, scale_prior).
        K (int): Number of samples to generate.

    Returns:
        means (list of ndarrays): List of mean vectors.
        covariances (list of ndarrays): List of covariance matrices.
    """
    mean_prior, cov_prior, df_prior, scale_prior = niw_params

    means = []
    covariances = []
    for _ in range(K):
        # Sample mean vector from the Gaussian prior
        mean_sample = np.random.multivariate_normal(mean_prior, cov_prior)

        # Sample covariance matrix from the inverse-Wishart prior
        cov_sample = invwishart.rvs(df_prior, scale_prior)

        means.append(mean_sample)
        covariances.append(cov_sample)

    return means, covariances
