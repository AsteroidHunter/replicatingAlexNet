"""
This script employs a running mean technique to compute the mean of R, G, and B values
of all images in the training set.

The computed mean will later be used for scaling the data when PCA is performed on the
images for data augmentation.
"""

import torchvision.transforms.functional as F
from tqdm import tqdm
from PIL import Image
import torch
import os


def prep_data_for_pca(
        image_path,
        mean_activity
):
    """
    Function that subtracts the mean activity of the entire training set from a three
    channel 256 x 256 image and returns a (256 * 256, 3) dimension tensor with int16
    datatype.

    The mean activity should be computed separately and passed as a tensor.

    For the PCA performed in the AlexNet paper, this function should be run on
    all training images, stacked vertically, and then passed as the input for the PCA.

    :param image_path: Path to the image
    :param mean_activity: Mean activity of all images as a tensor
    :return: Image as a torch tensor with dimension (256 * 256, 3) and int16 datatype
    """
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

    return prepped_tensor


def compute_running_mean(
        path_to_all_images,
        mean_activity_all_images,
):
    """
    Simple function that computes the running mean of RGB values of all images.

    :param path_to_all_images: The path of the directory containing all training images
    :param mean_activity_all_images: The mean activity of all training images
    :return: The mean of R, G, and B values (respectively) as a torch tensor
    """
    running_mean = torch.zeros(3)
    count = 0

    for i in tqdm(path_to_all_images):
        prepped_tensor = prep_data_for_pca(i, mean_activity_all_images)
        prepped_tensor_mean = torch.mean(prepped_tensor.to(torch.float32), dim=0)

        count += 1
        running_mean += (prepped_tensor_mean - running_mean) / count

    return running_mean


if __name__ == "__main__":
    # compiling the paths to all the training images in a list
    training_images_output_path = "../images/training_images_processed/"
    training_images_processed_path = [
        training_images_output_path + f for f in os.listdir(training_images_output_path)
        if f.endswith('.JPEG')
    ]

    # loading the mean activity of the image computed
    # under bullet point 1 of augment_ingest.ipynb
    mean_activity_all_training = torch.load(
        "../images/mean_activity_of_all_training_images.pt"
    )

    # running the function
    mean_for_scaling = compute_running_mean(
        training_images_processed_path,
        mean_activity_all_training
    )

    # reshaping the mean to match the shape of quantiles saved from
    # quantiles_for_data_scaling
    mean_for_scaling = mean_for_scaling.reshape(1, 3)

    # saving the mean as a tensor
    torch.save(mean_for_scaling, "true_means.pt")
