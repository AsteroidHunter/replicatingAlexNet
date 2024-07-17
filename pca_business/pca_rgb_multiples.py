"""
__author__: akash

This script implements incremental PCA on all the AlexNet training images. The image
paths are used as the input, and they are resized to a (256 * 256, 3)-sized tensor and
scaled before being passed to the model. The model iterates through all the provided
images in O(n) time. The eigenvalues and eigenvectors are computed and stored as tensors.

These eigenvalues and eigenvectors will later be used to augment the images when they
are fed into the convolutional neural network model.

"""

import torchvision.transforms.functional as F
from sklearn import decomposition
from numpy.linalg import eig
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
import os


def standard_scaling(
    data,
    quantiles
):
    mean, standard_deviation = quantiles
    return (data - mean) / standard_deviation


def prep_data_for_pca(
        image_path,
        mean_activity,
        quantiles,
        scaling_function=standard_scaling,
        scaling_bool=True,
):
    opened_image = Image.open(image_path)
    prepped_tensor = F.pil_to_tensor(opened_image)

    # subtracting the mean activity of the training set
    # since this is a pre-processing step, it should be done
    # before the augmentation
    # I saved the mean activity with dimension (256, 256, 3)
    if mean_activity.shape != (3, 256, 256):
        mean_activity = mean_activity.permute(2, 0, 1)
    prepped_tensor = prepped_tensor - mean_activity

    # reshaping from (channel, height, width) to (height, width, channel)
    prepped_tensor = prepped_tensor.permute(1, 2, 0)

    # and then to (height * width, channel)
    prepped_tensor = prepped_tensor.reshape(-1, 3)

    # converting to dtype int16 to reduce the amount of space and
    # support storage of -ve integers which may show up during the PCA
    prepped_tensor = prepped_tensor.to(torch.int16)

    # scaling the prepped tensor based on the quantiles found
    # using quantiles_for_data_scaling.py
    if scaling_bool:
        prepped_tensor = scaling_function(prepped_tensor, quantiles)

    return prepped_tensor

if __name__ == "__main__":
    # loading and compiling all the image paths in a list
    training_images_output_path = "../images/training_images_processed/"
    training_images_processed_path = [
        training_images_output_path + f for f in os.listdir(training_images_output_path)
        if f.endswith('.JPEG')
    ]

    mean_activity_all_training = torch.load(
        "../images/mean_activity_of_all_training_images.pt"
    )

    # loading the previously computed mean and standard deviation
    best_estimated_mean = torch.load("true_means.pt")
    best_estimated_std = torch.load("best_estimated_stds.pt")

    quantiles_standard = (
        best_estimated_mean,
        best_estimated_std,
    )

    # setting up and running the incremental PCA model
    ipca = decomposition.IncrementalPCA(n_components=3, batch_size=1)
    for tensor in tqdm(training_images_processed_path):
        # at each iteration, one image as a tensor is scaled and passed to the model
        tensor_scaled = prep_data_for_pca(
            tensor,
            mean_activity=mean_activity_all_training,
            quantiles=quantiles_standard
        )

        ipca.partial_fit(tensor_scaled)

    # retrieving the results and computing the covariance matrix to find the eigenv*s
    components = ipca.components_
    explained_variance = ipca.explained_variance_
    covariance_matrix = components.T @ np.diag(explained_variance) @ components

    eigenvalues, eigenvectors = eig(covariance_matrix)

    # saving the eigenv*s as tensors
    torch.save(
        torch.from_numpy(eigenvalues).to(torch.float16),
        "eigenvalues_from_ipca.pt"
    )
    torch.save(
        torch.from_numpy(eigenvectors).to(torch.float16),
        "eigenvectors_from_ipca.pt"
    )


