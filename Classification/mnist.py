import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torchvision.models import resnet18
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim
import numpy as np
from hinge_loss import *
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pickle
import json

certificate = 1

# 1. Load and prepare the MNIST dataset (1 vs 7)
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # Resize to 224x224 for ResNet
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229]),  # Normalize for grayscale
    ]
)

train_dataset = MNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = MNIST(root="./data", train=False, transform=transform, download=True)

label_pairs = [(3, 5), (4, 9), (5, 8), (3, 8), (0, 6)]
label1 = 0
label2 = 6

# Filter out only the classes 1 and 7
train_mask = (train_dataset.targets == label1) | (train_dataset.targets == label2)
test_mask = (test_dataset.targets == label1) | (test_dataset.targets == label2)
train_dataset.data = train_dataset.data[train_mask]
train_dataset.targets = train_dataset.targets[train_mask]
test_dataset.data = test_dataset.data[test_mask]
test_dataset.targets = test_dataset.targets[test_mask]

# Convert targets to +1 and -1
train_dataset.targets[train_dataset.targets == label1] = -1
train_dataset.targets[train_dataset.targets == label2] = 1
test_dataset.targets[test_dataset.targets == label1] = -1
test_dataset.targets[test_dataset.targets == label2] = 1

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Step 2: Pass the images through pre-trained ResNet18 to get features
resnet = resnet18(pretrained=True)
resnet.fc = (
    nn.Identity()
)  # Remove the final fully connected layer to use as a feature extractor


def extract_features(loader, model):
    features, labels = [], []
    with torch.no_grad():
        for images, target in loader:
            images = images.repeat(1, 3, 1, 1)  # Convert grayscale to 3 channels
            output = model(images)
            features.append(output)
            labels.append(target)
    return torch.cat(features), torch.cat(labels)


train_features, train_labels = extract_features(train_loader, resnet)
test_features, test_labels = extract_features(test_loader, resnet)


# Apply PCA for dimensionality reduction
def apply_pca(features, n_components):
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(features)
    return reduced_features


# Parameters
n_components = 10  # Desired number of components

# Convert features to numpy arrays for PCA
train_features_np = train_features.numpy()
test_features_np = test_features.numpy()

# Apply PCA to reduce dimensionality
train_features_reduced = apply_pca(train_features_np, n_components)
test_features_reduced = apply_pca(test_features_np, n_components)

# Convert back to PyTorch tensors
train_features = torch.tensor(train_features_reduced, dtype=torch.float32)
test_features = torch.tensor(test_features_reduced, dtype=torch.float32)

# Step 3: Normalize the dataset by zero mean and max L2 norm
mean = train_features.mean(dim=0)
train_features -= mean
test_features -= mean

train_norms = torch.norm(train_features, p=2, dim=1)
max_norm = train_norms.max()
train_features /= max_norm

test_norms = torch.norm(test_features, p=2, dim=1)
test_features /= max_norm

# Step 4: Concatenate 1/sqrt(d) to each feature vector and normalize
d = train_features.shape[1]
bias_feature = 1 / np.sqrt(d)
# bias_feature = 0

train_features = torch.cat(
    [train_features, bias_feature * torch.ones(train_features.size(0), 1)], dim=1
)
test_features = torch.cat(
    [test_features, bias_feature * torch.ones(test_features.size(0), 1)], dim=1
)

new_train_norms = torch.norm(train_features, p=2, dim=1)
new_max_norm = new_train_norms.max()
train_features /= new_max_norm

new_test_norms = torch.norm(test_features, p=2, dim=1)
test_features /= new_max_norm

if certificate:

    train_features *= train_labels.view(-1, 1).float()
    test_features *= test_labels.view(-1, 1).float()

    # Step 6: Randomly select 500 data points
    num_samples = 500
    indices = torch.randperm(len(train_features))[:num_samples]

    # Extract features and labels for these 100 samples
    Z = train_features[indices]
    Z = Z.numpy()
    print("Computing Certificate")

    # # Example epsilon, eta, and sigma lists
    epsilon_list = [0.01, 0.02, 0.03, 0.04, 0.05]
    eta_list = [5e-5, 1e-4]
    sigma_list = [3e-3, 6e-3, 1e-2, 3e-2, 6e-2]

    from concurrent.futures import ProcessPoolExecutor

    def compute_certificate(Z, epsilon, eta, sigma):
        # Simulate the hinge_certificate function
        certificate_value = hinge_certificate(Z, sigma, eta, epsilon)["optimal_value"]
        return epsilon, eta, sigma, certificate_value

    # Prepare parameters for parallel execution
    params = [
        (Z, epsilon, eta, sigma)
        for epsilon in epsilon_list
        for eta in eta_list
        for sigma in sigma_list
    ]
    # eta_sigma_pairs = [(5e-5, 6e-3), (5e-5, 1e-2), (1e-4, 1e-2), (5e-5, 6e-2)]
    # params = [(Z, epsilon, eta, sigma) for epsilon in epsilon_list for (eta, sigma) in eta_sigma_pairs]

    # Function to unpack parameters and call compute_certificate
    def worker(params):
        return compute_certificate(*params)

    results = {}
    with ProcessPoolExecutor() as executor:
        for epsilon, eta, sigma, certificate_value in executor.map(worker, params):
            key = (eta, sigma)
            if key not in results:
                results[key] = {"epsilon": [], "certificate": []}
            results[key]["epsilon"].append(epsilon)
            results[key]["certificate"].append(certificate_value)

    print(results)
    # Save the results dictionary to a file
    with open("results_mnist_{}_vs_{}_hinge.pkl".format(label1, label2), "wb") as f:
        pickle.dump(results, f)

else:
    # Prepare data loaders for extracted features
    batch_size = 1

    num_samples = 500
    indices = torch.randperm(len(train_features))[:num_samples]
    train_data = TensorDataset(train_features[indices], train_labels[indices])
    # train_data = TensorDataset(train_features, train_labels)
    Z_target = train_features[indices] * train_labels[indices].view(-1, 1).float()
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # Step 5: Train a linear model with no bias using regularized hinge loss
    class LinearModel(nn.Module):
        def __init__(self, input_dim):
            super(LinearModel, self).__init__()
            self.linear = nn.Linear(input_dim, 1, bias=False)

        def forward(self, x):
            return self.linear(x)

    # Hinge loss function
    def hinge_loss(output, target):
        return torch.mean(torch.clamp(1 - output * target, min=0))

    input_dim = train_features.size(1)

    import json
    import torch
    import concurrent.futures
    from torch import optim

    # Define your LinearModel and hinge_loss functions here if not defined in the snippet

    # Initialize dictionaries to store results
    accuracy_results = {}
    hinge_loss_results = {}

    epsilon_list = [0.01, 0.02, 0.03, 0.04, 0.05]
    # epsilon_list = [0.0, ]
    eta_list = [5e-3, 1e-3, 5e-4, 1e-4, 5e-5]
    sigma_list = [3e-3, 6e-3, 1e-2, 3e-2, 6e-2]
    # eta_sigma_pairs = [(eta, sigma) for eta in eta_list for sigma in sigma_list]
    eta_sigma_pairs = [(5e-5, 6e-3), (5e-5, 1e-2), (1e-4, 1e-2), (5e-5, 6e-2)]
    attack_types = ["fgsm", "pgd", "label_flip"]

    for eta, sigma in eta_sigma_pairs:
        for epsilon in epsilon_list:
            print(epsilon)
            for attack in attack_types:
                print(attack)
                input_dim = train_features.size(1)
                model = LinearModel(input_dim)
                # model.print_weights()

                # Use SGD with an initial learning rate
                optimizer = optim.SGD(model.parameters(), lr=eta, weight_decay=sigma)

                # Initialize a list to store parameters from the last epoch
                last_epoch_weights = []

                # Train the model using batch SGD
                epochs = 5000

                for epoch in range(epochs):
                    model.train()
                    for batch_features, batch_labels in train_loader:
                        optimizer.zero_grad()
                        adv = np.random.binomial(1, epsilon)
                        if adv == 1:
                            # outputs = model(-batch_features).squeeze()
                            if attack == "fgsm":
                                z_adv = projected_gradient_ascent(
                                    model.linear.weight.data.squeeze().numpy(),
                                    Z_target.numpy(),
                                    sigma,
                                    eta,
                                    100,
                                    1,
                                )
                            elif attack == "pgd":
                                z_adv = projected_gradient_ascent(
                                    model.linear.weight.data.squeeze().numpy(),
                                    Z_target.numpy(),
                                    sigma,
                                    eta,
                                    1,
                                    100,
                                )
                            else:
                                z_adv = -batch_features
                            z_adv = torch.tensor(z_adv, dtype=torch.float32)
                            outputs = model(z_adv).squeeze()
                        else:
                            outputs = model(batch_features).squeeze()
                        loss = hinge_loss(outputs, batch_labels.float())
                        loss.backward()
                        optimizer.step()
                        # print(loss)
                        # model.print_weights()

                        # Store model parameters if we are in the last epoch
                        if epoch == epochs - 1:
                            last_epoch_weights.append(
                                {
                                    name: param.clone()
                                    for name, param in model.named_parameters()
                                }
                            )

                # Evaluate the model on the test set for the last epoch weights
                model.eval()
                total_hinge_loss = 0.0
                total_accuracy = 0.0

                with torch.no_grad():
                    for weights in last_epoch_weights:
                        # Load the stored parameters into the model
                        for name, param in model.named_parameters():
                            param.copy_(weights[name])

                        # Compute hinge loss on test set
                        test_outputs = model(test_features).squeeze()
                        test_hinge_loss = hinge_loss(
                            test_outputs, test_labels.float()
                        ).item()
                        # test_outputs = model(train_features).squeeze()
                        # test_hinge_loss = hinge_loss(test_outputs, train_labels.float()).item()
                        total_hinge_loss += test_hinge_loss

                        # Compute accuracy on test set
                        predictions = torch.sign(test_outputs)
                        accuracy = (predictions == test_labels).float().mean().item()
                        # accuracy = (predictions == train_labels).float().mean().item()
                        total_accuracy += accuracy

                    # Compute the average hinge loss and accuracy
                    average_hinge_loss = total_hinge_loss / len(last_epoch_weights)
                    average_accuracy = total_accuracy / len(last_epoch_weights)

                # Print the results
                print(f"Eta: {eta}, Sigma: {sigma}")
                print(f"Average Test Hinge Loss (last epoch): {average_hinge_loss:.4f}")
                print(
                    f"Average Test Accuracy (last epoch): {average_accuracy * 100:.2f}%"
                )

                # Convert tuple to string for keys
                # hinge_loss_results[f"{eta}_{sigma}"] = average_hinge_loss
                # accuracy_results[f"{eta}_{sigma}"] = average_accuracy

                key = f"{eta}_{sigma}_{attack}"
                if key not in hinge_loss_results:
                    hinge_loss_results[key] = {"epsilon": [], "certificate": []}
                    accuracy_results[key] = {"epsilon": [], "certificate": []}
                hinge_loss_results[key]["epsilon"].append(epsilon)
                hinge_loss_results[key]["certificate"].append(average_hinge_loss)
                accuracy_results[key]["epsilon"].append(epsilon)
                accuracy_results[key]["certificate"].append(average_accuracy)

    # Save the results to JSON files
    with open("accuracy_results_flip_attack_new.json", "w") as f:
        json.dump(accuracy_results, f)

    with open("hinge_loss_results_flip_attack_new.json", "w") as f:
        json.dump(hinge_loss_results, f)
