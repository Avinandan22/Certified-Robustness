import numpy as np
import cvxpy as cp
import torch
import torch.optim as optim

from sklearn.preprocessing import normalize


def hinge_certificate(Z, sigma, eta, epsilon):
    """
    This function solves the optimization problem described with the given parameters.

    Parameters:
    Z (np.ndarray): The dataset with shape (N, d).
    sigma (float): The value of sigma.
    epsilon (float): The value of epsilon.
    mu (np.ndarray): A vector of dimension (d,).
    delta_1 (float): A scalar value for delta_1.

    Returns:
    dict: A dictionary containing the optimal value and the optimal variables.
    """
    N, d = Z.shape  # Get the dimensions from the provided dataset Z

    # Define variables
    nu_1 = cp.Variable(N, nonneg=True)
    nu_2 = cp.Variable(N, nonneg=True)
    nu_3 = cp.Variable((N, d), nonneg=True)
    nu_4 = cp.Variable((N, d), nonneg=True)
    nu_5 = cp.Variable((N, d), nonneg=True)
    nu_6 = cp.Variable((N, d), nonneg=True)
    nu_7 = cp.Variable(N, nonneg=True)
    nu_8 = cp.Variable(nonneg=True)
    nu_9 = cp.Variable(nonneg=True)
    nu_10 = cp.Variable(nonneg=True)

    # Define bfA and bfb as d × d matrix and d-dimensional vector
    bfA = cp.Variable((d, d), symmetric=True)
    bfb = cp.Variable(d)
    # bfA = np.eye(d)
    # bfb = np.zeros(d)

    # Define functions p, p', D, D', q, q', r_i, s
    p = 0.5 * cp.hstack(
        [
            -sigma * eta * bfb
            + cp.sum(
                [(nu_1[i] - nu_2[i]) * Z[i] - nu_4[i] + nu_6[i] for i in range(N)],
                axis=0,
            ),
            epsilon * eta * bfb,
        ]
    )

    p_prime = 0.5 * (
        -sigma * eta * epsilon * bfb
        + cp.sum(
            [(nu_1[i] - nu_2[i]) * Z[i] - nu_4[i] + nu_6[i] for i in range(N)], axis=0
        )
    )

    D = cp.bmat(
        [
            [
                (1 - (1 - sigma * eta) ** 2) * bfA + nu_8 * np.eye(d),
                -epsilon * (1 - sigma * eta) * eta * bfA + nu_9 * np.eye(d),
            ],
            [
                -epsilon * (1 - sigma * eta) * eta * bfA + nu_9 * np.eye(d),
                -epsilon * eta**2 * bfA + nu_10 * np.eye(d),
            ],
        ]
    )

    D_prime = (1 - (1 - sigma * eta) ** 2) * bfA + nu_8 * np.eye(d)

    q = (
        -cp.sum(nu_1)
        + (2 + 1 / sigma) * cp.sum(nu_2)
        + (1 / sigma) * cp.sum(nu_4 + nu_6)
        + cp.sum(nu_7)
        + nu_8 / (sigma**2)
        + 2 * nu_9
        + nu_10
    )
    q_prime = (
        -cp.sum(nu_1)
        + (2 + 1 / sigma) * cp.sum(nu_2)
        + (1 / sigma) * cp.sum(nu_4 + nu_6)
        + cp.sum(nu_7)
        + nu_8 / (sigma**2)
    )
    # q = -cp.sum(nu_1) + (2 + 1/sigma) * cp.sum(nu_2) + (1/sigma) * cp.sum(nu_4 + nu_6) + cp.sum(nu_7) +  nu_8 / sigma + 2 * nu_9 + nu_10
    # q_prime = -cp.sum(nu_1) + (2 + 1/sigma) * cp.sum(nu_2) + (1/sigma) * cp.sum(nu_4 + nu_6) + cp.sum(nu_7) + nu_8 / sigma

    # Define constraints
    constraints = []
    for i in range(N):
        a_i = (
            (1 + 1 / sigma) * nu_1[i]
            - (1 + 1 / sigma) * nu_2[i]
            + (1 / sigma) * cp.sum((nu_3[i] - nu_4[i] + nu_5[i] - nu_6[i]))
            - nu_7[i]
        )
        b_i = nu_3[i, :] + nu_4[i, :] - nu_5[i, :] - nu_6[i, :]
        c_i = (1 / N) * (
            (1 - epsilon) * eta**2 * Z[i].T @ bfA @ Z[i]
            + (1 - epsilon) * eta * bfb.T @ Z[i]
            + 1
        )
        d_i = (1 / N) * (
            2 * (1 - epsilon) * eta * (1 - sigma * eta) * bfA @ Z[i] - Z[i]
        )
        constraints.append(a_i + c_i == 0)
        constraints.append(b_i + d_i == 0)

    constraints.append(D >> 0)
    # constraints.append(D_prime >> 0)

    # Define the first minimization problem (minimizing \|p\|^2_{D^{-1}} + q)
    obj1 = cp.Minimize(cp.matrix_frac(p, D) + q)
    prob1 = cp.Problem(obj1, constraints)
    min_val1 = prob1.solve()

    objective = min_val1

    # Define the second minimization problem (minimizing \|p'\|^2_{D'^{-1}} + q')
    obj2 = cp.Minimize(cp.matrix_frac(p_prime, D_prime) + q_prime)
    prob2 = cp.Problem(obj2, constraints)
    min_val2 = prob2.solve()

    # objective = min_val2

    # # Now take the maximum of the two minimization results
    objective = max(min_val1, min_val2)

    # Return the results
    return {
        "optimal_value": objective,
        "nu_1": nu_1.value,
        "nu_2": nu_2.value,
        "nu_3": nu_3.value,
        "nu_4": nu_4.value,
        "nu_5": nu_5.value,
        "nu_6": nu_6.value,
        "nu_7": nu_7.value,
        "nu_8": nu_8.value,
        "nu_9": nu_9.value,
        "nu_10": nu_10.value,
        "bfA": bfA.value,
        "bfb": bfb.value,
    }


def hinge_certificate_meta(Z_list, benign_loss_list, sigma_eta_pairs, epsilon, kappa):
    min_obj = np.inf
    best_params = None
    for eta, sigma in sigma_eta_pairs:
        i = 0
        meta_loss = 0
        for Z in Z_list:
            cert = hinge_certificate(Z, sigma, eta, epsilon)
            meta_loss += (
                kappa * cert["optimal_value"] + benign_loss_list[i][(eta, sigma)]
            )
            i += 1
        if meta_loss <= min_obj:
            min_obj = meta_loss
            best_params = (eta, sigma)
    return best_params


def generate_linearly_separable_dataset(num_samples, num_features, theta, scale=5.0):
    # Generate random feature vectors
    features = np.random.uniform(-scale, scale, size=(num_samples, num_features))

    # Assign labels based on the side of the hyperplane
    labels = np.sign(np.dot(features, theta))

    # Generate product of label and feature
    labeled_features = labels[:, np.newaxis] * features

    return labeled_features


def hinge_loss(theta, Z):
    predictions = np.dot(Z, theta)
    loss = np.maximum(0, 1 - predictions)
    return np.mean(loss)


def adversarial_examples(
    Z,
    theta,
    epsilon,
    cuz,
    num_random_selections,
    learning_rate=0.01,
    num_iterations=1000,
):
    max_loss = -float("inf")
    max_loss_perturbation = None

    for _ in range(num_random_selections):
        num_points_to_perturb = int(len(Z) * epsilon)
        indices_to_perturb = np.random.choice(
            len(Z), num_points_to_perturb, replace=False
        )

        theta_adv = theta.copy()
        adversarial_products = []

        # print(len(Z))
        for idx in range(len(Z)):
            # print(z)
            z = Z[idx]
            if idx in indices_to_perturb:
                perturbed_z = Z[idx].copy()
                for _ in range(num_iterations):
                    if np.dot(perturbed_z, theta_adv) <= 1:
                        gradient = -perturbed_z
                        theta_adv -= learning_rate * gradient
                        perturbed_z = np.clip(z - perturbed_z, -cuz, cuz) + z
                        perturbed_z = np.clip(
                            perturbed_z, -1, 1
                        )  # Ensure features remain within [-1, 1]^d
                adversarial_products.append(perturbed_z)
            else:
                adversarial_products.append(Z[idx])

        adversarial_products = np.array(adversarial_products)
        adversarial_loss = hinge_loss(theta_adv, adversarial_products)

        if adversarial_loss > max_loss:
            max_loss = adversarial_loss
            max_loss_perturbation = adversarial_products

    return max_loss_perturbation


def update_theta(
    theta, theta_star, cuz, eta, epsilon, T, zeta, B, num_random_selections, batch_size
):
    """
    Update the parameter vector theta over T epochs given the learning rate eta and vectors z.
    After each update, scale theta if its norm exceeds zeta.

    Parameters:
        theta (np.ndarray): Initial parameter vector.
        eta (float): Learning rate.
        z (list of np.ndarray): List of vectors, one for each epoch.
        T (int): Number of epochs (iterations).
        zeta (float): Norm threshold for scaling theta.

    Returns:
        np.ndarray: Updated parameter vector after T epochs.
    """
    d = theta.shape[0]
    theta_list = []
    benign_count = 0
    adv_count = 0
    for t in range(T):
        # print(t)
        Z = generate_linearly_separable_dataset(batch_size, d, theta_star)
        Z_perturbed = adversarial_examples(
            Z,
            theta,
            epsilon,
            cuz,
            num_random_selections,
            learning_rate=0.01,
            num_iterations=1000,
        )
        nonzero_grad_elements = Z_perturbed[Z_perturbed <= 1]
        gradient = -np.sum(nonzero_grad_elements) / batch_size
        theta = theta + eta * gradient
        norm_theta = np.linalg.norm(theta)
        if norm_theta > zeta:
            theta = (theta / norm_theta) * zeta
        w = np.random.randn(d)
        noise = np.dot(B, w)
        theta = theta + eta * noise
        if t >= 0.9 * T:
            theta_list.append(theta)
    return theta_list


def SAA_outer(theta, theta_star, eta, T, zeta, B, batch_size):
    d = theta.shape[0]
    proj_list = []
    noise_list = []
    benign_count = 0
    adv_count = 0
    for t in range(T):
        Z = generate_linearly_separable_dataset(batch_size, d, theta_star)
        nonzero_grad_elements = Z[Z <= 1]
        gradient = -np.sum(nonzero_grad_elements) / batch_size
        theta = theta + eta * gradient
        norm_theta = np.linalg.norm(theta)
        if norm_theta > zeta:
            theta = (theta / norm_theta) * zeta
        w = np.random.randn(d)
        noise = np.dot(B, w)
        theta = theta + eta * noise
        if t >= 0.9 * T:
            proj_list.append(theta)
            noise_list.append(w)
    return proj_list, noise_list


def optimize_S(proj_list, noise_list, Z, B, A, kappa):
    n = len(proj_list)
    learning_rate = 0.01  # Define your learning rate
    num_iterations = 100  # Define the number of iterations for gradient descent

    B = torch.tensor(B, requires_grad=True, dtype=torch.float32)
    Z = torch.tensor(Z, dtype=torch.float32)
    A = torch.tensor(A, dtype=torch.float32)

    optimizer = optim.Adam([B], lr=learning_rate)

    for j in range(num_iterations):
        optimizer.zero_grad()
        loss = 0
        for i in range(n):
            theta = torch.tensor(proj_list[i], dtype=torch.float32) + torch.matmul(
                B, torch.tensor(noise_list[i], dtype=torch.float32)
            )
            predictions = torch.matmul(Z, theta)
            hinge_loss = torch.mean(torch.clamp(1 - predictions, min=0))
            loss += hinge_loss
        loss /= n
        print("Iter", j, "Loss : ", loss)
        trace_term = kappa * torch.trace(torch.matmul(torch.matmul(A, B), B.t()))
        loss += trace_term
        loss.backward()
        optimizer.step()

    return B.detach().numpy(), loss.detach().item()


def optimal_S(theta, theta_star, eta, T, zeta, B, batch_size, A, kappa):
    proj_list, noise_list = SAA_outer(theta, theta_star, eta, T, zeta, B, batch_size)
    print(proj_list)
    d = theta.shape[0]
    Z = generate_linearly_separable_dataset(batch_size, d, theta_star)
    return optimize_S(proj_list, noise_list, Z, B, A, kappa)


def generate_theta_star(d):
    """Generates a d-dimensional random vector"""
    return np.random.randn(d)


def generate_data(theta_star, n):
    """
    Generates n d-dimensional vectors from a standard multivariate Gaussian,
    and returns an array of these vectors modified by the sign of their dot product with theta_star.
    """
    d = len(theta_star)
    theta_star /= np.linalg.norm(theta_star)
    n = int(n / 2)
    X1 = theta_star + np.random.randn(n, d)  # Generate n d-dimensional Gaussian vectors
    X2 = -theta_star + np.random.randn(
        n, d
    )  # Generate n d-dimensional Gaussian vectors
    X = np.concatenate([X1, X2])
    # print(X.shape)
    # Compute dot products for all vectors at once
    dot_products = np.dot(X, theta_star)
    # Apply the sign function to each dot product and modify X accordingly
    signs = np.sign(dot_products)[
        :, np.newaxis
    ]  # make it (n,1) to broadcast along rows
    Z = (
        X * signs
    )  # element-wise multiplication that broadcasts the sign across each row

    return Z


def evaluate(theta, theta_star):
    """
    Generates n d-dimensional vectors from a standard multivariate Gaussian,
    and returns an array of these vectors modified by the sign of their dot product with theta_star.
    """
    d = len(theta)
    Z = generate_linearly_separable_dataset(100000, d, theta_star)
    return hinge_loss(theta, Z)


def evaluate_classification_rate(theta, theta_star):
    """
    Generates n d-dimensional vectors from a standard multivariate Gaussian,
    and returns an array of these vectors modified by the sign of their dot product with theta_star.
    """
    d = len(theta)
    Z = generate_linearly_separable_dataset(100000, d, theta_star)

    dot_products = np.dot(Z, theta)
    positive_mask = dot_products > 0

    # Count positive numbers
    positive_count = np.sum(positive_mask)

    # Total number of elements in the array
    total_elements = dot_products.size

    # Fraction of elements that are positive
    fraction_positive = positive_count / total_elements

    return fraction_positive


def gram_schmidt(V):
    def projection(u, v):
        return (np.dot(v, u) / np.dot(u, u)) * u

    U = np.zeros(V.shape)
    U[0] = V[0]

    for i in range(1, V.shape[0]):
        U[i] = V[i] - sum(projection(U[j], V[i]) for j in range(i))
        if np.linalg.norm(U[i]) < 1e-10:  # Avoid zero division errors
            U[i] = np.random.randn(*U[i].shape)  # Replace linearly dependent vector
            U[i] -= sum(projection(U[j], U[i]) for j in range(i))
        U[i] /= np.linalg.norm(U[i])  # Normalize

    return U


def extend_to_orthonormal_basis(theta_star, d):
    # Normalize theta_star
    theta_star = theta_star / np.linalg.norm(theta_star)

    # Create an initial set of basis vectors (e.g., standard basis)
    V = np.eye(d)

    # Replace the first vector with theta_star (or any other, strategically)
    V[0] = theta_star

    # Apply Gram-Schmidt to get an orthonormal basis including theta_star
    return gram_schmidt(V)


def gradient_descent_step(A, gradient, alpha):
    """
    Perform a gradient descent step.
    """
    return A - alpha * gradient


def project_psd_trace_one(A):
    """
    Project matrix A to be PSD with trace 1.
    """
    # Eigen-decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    # Enforce PSD
    eigenvalues = np.maximum(eigenvalues, 0)
    # Normalize trace to 1
    if eigenvalues.sum() > 0:
        eigenvalues /= eigenvalues.sum()
    # Reconstruct the matrix
    A_projected = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    return A_projected


def update_and_project_A(zadv, theta, q, Z, A, n, alpha):
    """
    Compute the gradient, update matrix A using gradient descent, and project it onto PSD with trace = 1.

    Args:
    - X (np.array): Matrix of shape (num_terms, n) representing x_i vectors.
    - Y (np.array): Matrix of shape (num_terms, n) representing y_i vectors.
    - A (np.array): The matrix to be updated, of shape (n, n).
    - alpha (float): The learning rate for the gradient descent step.

    Returns:
    - A (np.array): Updated and projected matrix A.
    """
    # Compute the gradient
    l1 = np.array([q[i] * np.outer(Z[i], Z[i]) for i in range(n)])
    l1 = np.mean(l1, axis=0)
    l2 = np.array([q[i] * np.outer(Z[i], theta) for i in range(n)])
    l2 = np.mean(l2, axis=0)
    gradient = (
        epsilon * eta**2 * np.outer(zadv, zadv)
        + 2 * epsilon * eta * np.outer(zadv, theta)
        + (1 - epsilon) * eta**2 * l1
        + 2 * (1 - epsilon) * eta * l2
        + eta**2 * np.eye(d)
    )
    A_updated = gradient_descent_step(A, gradient, alpha)

    # # Project onto PSD with trace 1
    # A_projected = project_psd_trace_one(A_updated)
    A_projected = A_updated

    return A_projected


from scipy.stats import norm
from scipy import optimize


def minimize_objective(Z, A, theta_star, eta, n, kappa):
    """
    Minimize the given objective function over matrix S.

    Args:
    Z (np.ndarray): An n x d matrix of data points.
    theta_star (np.ndarray): A d-dimensional vector.
    eta (float): A scalar parameter.
    n (int): Number of data points (rows of Z).

    Returns:
    np.ndarray: The optimized d x d positive semi-definite matrix S.
    """
    d = Z.shape[1]  # Dimension of each data point

    # Define the objective function
    def objective(L_flat, A, Z, theta_star, eta, n, kappa):
        # Reshape L_flat back to matrix form and ensure it's lower triangular
        L = np.tril(L_flat.reshape((d, d)))

        # Compute S as L * L.T to ensure it's PSD
        S = np.dot(L, L.T)
        total = 0
        for i in range(n):
            z_i = Z[i]
            v = 1 - np.dot(theta_star.T, z_i)
            sqrt_S_z = np.sqrt(np.dot(z_i.T, np.dot(S, z_i)))

            # Compute terms with Phi and phi
            term1 = v * norm.cdf(v / (eta * sqrt_S_z))
            term2 = eta * sqrt_S_z * norm.pdf(v / (eta * sqrt_S_z))

            total += term1 + term2
        total /= n
        total += eta**2 * np.trace(np.matmul(A, S)) * kappa
        return total / n

    # Initial guess for L (identity matrix flattened)
    L_initial = np.linalg.cholesky(np.eye(d)).flatten()

    # Minimize the objective function
    result = optimize.minimize(
        fun=objective,
        x0=L_initial,
        args=(A, Z, theta_star, eta, n, kappa),
        method="L-BFGS-B",
        options={"disp": False},
    )

    # Reconstruct S from the optimized L
    L_opt = np.tril(result.x.reshape((d, d)))
    S_opt = np.dot(L_opt, L_opt.T)

    objective_value = result.fun
    return S_opt, objective_value


def draw_samples(mu, N):
    mu = np.array(mu)
    d = mu.shape[0]  # Dimensionality of mu
    samples = []

    while len(samples) < N:
        sample = np.random.multivariate_normal(mu, 0.1 * np.eye(d))
        l2_norm = np.linalg.norm(sample)
        if l2_norm >= 1:
            sample /= l2_norm
        samples.append(sample)
        # if np.all(np.abs(sample) <= 1):
        #     samples.append(sample)

    return np.array(samples)


# Example usage
d = 5  # Dimension of the vector
n = 100  # Number of vectors to generate

mu_z = 0.5 * np.ones(d)  # Mean vector of the normal distribution
Z = draw_samples(mu_z, n)
# print(Z[0:5])
n_test = 2000
Z_test = draw_samples(mu_z, n_test)
n_train = 10000
Z_train = draw_samples(mu_z, n_train)
epochs = n_train


def F(theta, z, sigma, eta):
    theta_dot_z = np.dot(theta, z)
    if theta_dot_z <= 1:
        q = 1
    else:
        q = 0
    return (1 - sigma * eta) * theta + eta * q * z


def gradient_descent(Z, epochs, sigma, eta):
    d = Z.shape[1]
    N = Z.shape[0]
    theta = np.random.randn(d)
    l2_norm = np.linalg.norm(theta)
    if l2_norm >= (1 / sigma):
        theta /= l2_norm
    # theta /= np.max(np.abs(theta))
    # theta /= sigma
    theta_list = []
    for i in range(epochs):
        j = np.random.randint(N)
        theta = F(theta, Z[j], sigma, eta)
        if i >= 0.9 * epochs:
            theta_list.append(theta)
    return theta_list


def gradient_descent_poisoned(Z, epochs, sigma, eta, epsilon, typ):
    d = Z.shape[1]
    N = Z.shape[0]
    theta = np.random.randn(d)
    l2_norm = np.linalg.norm(theta)
    if l2_norm >= (1 / sigma):
        theta /= l2_norm
    # theta /= np.max(np.abs(theta))
    # theta /= sigma
    theta_list = []
    for i in range(epochs):
        adv = np.random.binomial(1, epsilon)
        j = np.random.randint(N)
        if adv == 1:
            theta = F(theta, -Z[j], sigma, eta)
        else:
            theta = F(theta, Z[j], sigma, eta)
        if i >= 0.9 * epochs:
            theta_list.append(theta)
    return theta_list


import numpy as np


def F(theta, z, sigma, eta):
    """
    Compute F(theta, z) based on the given formula.
    """
    indicator = 1 if np.dot(theta, z) > 0 else 0
    return (1 - sigma * eta) * theta + eta * indicator * z


def hinge_loss(F_theta_z, z_i):
    """
    Compute hinge loss for max(0, 1 - F(theta, z)^T z_i)
    """
    return max(0, 1 - np.dot(F_theta_z, z_i))


def hinge_loss_tensor(F_theta_z, z_i):
    """
    Compute hinge loss for max(0, 1 - F(theta, z)^T z_i)
    """
    return max(0, 1 - torch.dot(F_theta_z, z_i))


def objective(z, theta, Z, sigma, eta):
    """
    Compute the objective function: sum of hinge losses.
    """
    return sum(hinge_loss(F(theta, z, sigma, eta), z_i) for z_i in Z)


def project_onto_l2_ball(z, radius=1.0):
    """
    Project a vector z onto the L2 ball with the given radius.
    """
    norm = np.linalg.norm(z)
    if norm > radius:
        return z * (radius / norm)
    return z


def F_sigmoid(theta, z, sigma, eta):
    indicator = torch.sigmoid(torch.dot(theta, z))
    return (1 - sigma * eta) * theta + eta * indicator * z


def forward(theta_0, z_1_to_N_minus_K, zadv_1_to_K, sigma, eta):
    theta = theta_0
    # num_updates_per_adv = (N - K) // K
    K = len(zadv_1_to_K)
    num_updates_per_adv = len(z_1_to_N_minus_K) // K

    for i, z in enumerate(z_1_to_N_minus_K):
        # theta = (1 - sigma * eta) * theta + eta * (1 if torch.dot(theta, z) > 0 else 0) * z
        theta = (1 - sigma * eta) * theta + eta * torch.sigmoid(torch.dot(theta, z)) * z

        # Apply adversarial update at fixed interval
        if i % num_updates_per_adv == 0 and i != 0:
            adv_idx = i // num_updates_per_adv - 1
            # theta = (1 - sigma * eta) * theta + eta * (1 if torch.dot(theta, zadv_1_to_K[adv_idx]) > 0 else 0) * zadv_1_to_K[adv_idx]
            theta = (1 - sigma * eta) * theta + eta * torch.sigmoid(
                torch.dot(theta, zadv_1_to_K[adv_idx])
            ) * zadv_1_to_K[adv_idx]

    return theta


def optimize_adversarial_sequence(
    theta_0,
    z_1_to_N_minus_K,
    zprime_1_to_M,
    sigma,
    eta,
    K,
    d,
    num_iterations=100,
    learning_rate=0.01,
):
    zadv_1_to_K = [torch.randn(d, requires_grad=True) for _ in range(K)]
    # zadv_1_to_K = torch.stack(zadv_1_to_K, dim=0)
    optimizer = optim.Adam(zadv_1_to_K, lr=learning_rate)

    for i in range(num_iterations):
        total_hinge_loss = 0
        for zprime in zprime_1_to_M:
            theta = forward(theta_0, z_1_to_N_minus_K, zadv_1_to_K, sigma, eta)
            hinge_loss_value = hinge_loss_tensor(theta, zprime)
            total_hinge_loss += hinge_loss_value

        optimizer.zero_grad()
        total_hinge_loss.backward()

        # # Project each zadv onto the unit ball
        # for j, zadv in enumerate(zadv_1_to_K):
        #     zadv.grad /= torch.norm(zadv.grad)
        #     zadv.data -= learning_rate * zadv.grad
        #     zadv.data /= torch.max(torch.ones_like(zadv.data), torch.norm(zadv.data))

        optimizer.step()
    return zadv_1_to_K


def projected_gradient_ascent(theta, Z, sigma, eta, lr=0.1, max_iters=10):
    """
    Perform projected gradient ascent to find the optimal z, but ensure initial z satisfies np.dot(theta, z) > 0.
    """
    # Initialize z such that np.dot(theta, z) > 0 (indicator = 1)
    while True:
        z = np.random.randn(theta.shape[0])  # Random initialization
        z = project_onto_l2_ball(z)  # Project onto the L2 ball
        if np.dot(theta, z) > 0:  # Check if indicator is 1
            break  # If indicator is 1, break the loop and use this z

    for _ in range(max_iters):
        grad = np.zeros_like(z)
        for z_i in Z:
            F_theta_z = F(theta, z, sigma, eta)
            if 1 - np.dot(F_theta_z, z_i) > 0:  # if hinge loss is positive
                grad -= z_i  # gradient of the hinge loss w.r.t. z

        # Update z using gradient ascent
        z += lr * grad

        # Project z back onto the L2 ball
        z = project_onto_l2_ball(z)

    return z


from pyomo.environ import *
import numpy as np


def optimize_with_epsilon(T, N, d, sigma, eta, epsilon, W, Z_const=None):
    """
    Optimizes the given sequence of decision variables where only 1/epsilon fraction of T
    points are decision variables and the rest are constants. Decision variables are bounded by L2 norm of 1.

    Parameters:
        T (int): Total number of time steps.
        N (int): Number of constant vectors w.
        d (int): Dimension of z and theta.
        sigma (float): Sigma parameter for the evolution equation.
        eta (float): Eta parameter for the evolution equation.
        epsilon (float): Fraction of T that are decision variables.
        W (numpy array): Constant vectors w (shape: N x d).
        Z_const (numpy array): Predefined constant values for non-decision z_t's (shape: T x d).

    Returns:
        z_1 (list): Optimal value of the first decision variable z_1.
    """

    # If Z_const is not provided, generate random constants for the fixed z_t's
    if Z_const is None:
        Z_const = np.random.randn(T, d)

    # Initialize the Pyomo model
    model = ConcreteModel()

    # Identify the decision steps: every 1/epsilon step
    decision_indices = np.arange(0, T, int(1 / epsilon))

    # Define decision variables for z_t only at the decision steps
    model.z = Var(decision_indices, range(d), domain=Reals)

    # Define theta variables for each time step
    model.theta = Var(range(T + 1), range(d), domain=Reals)

    # Define the indicator variable I_t for each time step (0 or 1 depending on the condition)
    model.I = Var(range(T), domain=Binary)

    # Auxiliary variables to represent the hinge loss terms (non-negative)
    model.hinge_loss = Var(range(N), domain=NonNegativeReals)

    # Objective function: maximize the hinge loss sum for the final theta_T
    def hinge_loss_rule(model):
        return sum(model.hinge_loss[i] for i in range(N))

    model.obj = Objective(rule=hinge_loss_rule, sense=maximize)

    # Initial condition for theta_0
    initial_theta = np.random.randn(d)
    for j in range(d):
        model.theta[0, j] = initial_theta[j]

    # Constraint for the hinge loss: hinge_loss_i = max(0, 1 - theta_T^T w_i)
    def hinge_loss_constraint_rule(model, i):
        return model.hinge_loss[i] >= 1 - sum(
            model.theta[T, j] * W[i, j] for j in range(d)
        )

    model.hinge_loss_constraint = Constraint(range(N), rule=hinge_loss_constraint_rule)

    # Evolution of theta_t based on the update rule
    def theta_evolution_rule(model, t, j):
        if t in decision_indices:
            z_t = model.z[t, j]  # Free decision variables at every 1/epsilon step
        else:
            z_t = Z_const[t, j]  # Constant values for all other steps

        return (
            model.theta[t + 1, j]
            == (1 - sigma * eta) * model.theta[t, j] + eta * model.I[t] * z_t
        )

    model.theta_evolution = Constraint(range(T), range(d), rule=theta_evolution_rule)

    # Define the indicator constraint: I_t = 1 if theta_t^T z_t <= 1, otherwise 0
    # Set Big M = 1 / sigma
    Big_M = 1 / sigma

    def indicator_rule(model, t):
        if t in decision_indices:
            z_t = [
                model.z[t, j] for j in range(d)
            ]  # Free decision variables at decision steps
        else:
            z_t = Z_const[t, :]  # Constant values for the rest

        return sum(model.theta[t, j] * z_t[j] for j in range(d)) <= 1 + Big_M * (
            1 - model.I[t]
        )

    model.indicator_constraint = Constraint(range(T), rule=indicator_rule)

    # L2 norm constraint: ||z_t||_2 <= 1 for decision variables
    def l2_norm_constraint(model, t):
        return sum(model.z[t, j] ** 2 for j in range(d)) <= 1

    model.l2_norm_constraints = Constraint(decision_indices, rule=l2_norm_constraint)

    # Solve the optimization problem
    solver = SolverFactory("glpk")
    results = solver.solve(model, tee=True)

    # Collect the optimal value for z_1 (the first decision variable)
    z_1 = [value(model.z[decision_indices[0], j]) for j in range(d)]

    # Return only z_1
    return z_1
