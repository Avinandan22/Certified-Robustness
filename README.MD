# Keeping up with dynamic attackers: Certifying robustness to adaptive online data poisoning
=====================================================
## Description
---------------
This project aims to certify the robustness of machine learning models against adaptive online data poisoning attacks.
## Installation
--------------
To install the dependencies, run the following command:
pip install -r requirements.txt

## General Certificate Abstract Class
The generalized_certificate.py module provides an abstract class for computing certificates and using them to meta-learn a robust learning algorithm. This class serves as a foundation for implementing specific certification methods.

## Example Implementations
MeanEstimation/mean_estimation_toy.py: Provides a concrete implementation of the certificate abstract class for the mean estimation task.
Classification/classification_toy.py: Provides a concrete implementation of the certificate abstract class for the classification task.

## Mean Estimation
To perform mean estimation, follow these steps:
1. Navigate to the `MeanEstimation` directory: cd MeanEstimation
2. Run the following script: python3 mean_estimation.py

## Classification
To perform classification, follow these steps:
1. Download the MNIST dataset (or any corresponding dataset) and store the dataset in the `data` folder.
2. Navigate to the `Classification` directory: cd Classification
3. Run the following script: python3 mnist.py
