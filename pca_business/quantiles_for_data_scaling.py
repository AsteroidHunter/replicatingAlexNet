"""
__author__: akash

This script is used to estimate the mean, standard deviation, and three quantile values
(0.25, 0.5, and 0.75) which will be used for scaling the images (as tensors) before
they are used as an input for the incremental PCA analysis.

The estimation is done by randomly sampling n rows for m runs; the results from the m
runs are saved as a CSV, and median values of the m runs are stored as individual
tensors. The script was run using n = 10,000,000 and m = 10.

During testing, it was found that standard deviation and quantile estimation is highly
reliable even when 0.01% of all rows are sampled; however, the estimated mean isn't
accurate. A few lines which saved the mean values in the CSV and as a tensor have been
hashed out.
"""

import torchvision.transforms.functional as F
from multiprocessing import Pool
from tqdm import tqdm
from PIL import Image
import polars as pl
import numpy as np
import torch
import os


def chunkify(
        list_of_things,
        n_split
):
    """
    This function divides a list of things to list of lists.

    The last sub-list may have more items than the other sub-lists because some items
    don't neatly fall into n_split portions and are appended to the last sub-list.

    :param list_of_things: A list with items
    :param n_split: Number of sub-lists to split the original list into
    :return: List with n_split sub-lists
    """
    quotient, remainder = divmod(len(list_of_things), n_split)

    chunks = [list_of_things[quotient * n:quotient * (n + 1)] for n in range(n_split)]
    chunks[-1].extend(list_of_things[-remainder:])

    return chunks


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


def estimate_quantiles_for_scaling(
        batched_image_paths,
        random_subset_index,
        mean_activity_all_images,
        rows_to_sample,
):
    """
    This function returns the estimated median, Q1, and Q3 values for an N x 3 shaped
    tensor, where N >>> 0. The estimated quantiles are returned for each column,
    so the output is a tuple with three numpy arrays with one row and three columns.

    The mean activity of all the images from the training set is first subtracted from
    the input tensor to center the values around zero, then rows_to_sample amount of
    rows are randomly sampled and stacked as a single N x 3 tensor. Three quantile values
    for each column of this randomly sampled and stacked tensor is computed and returned.

    :param batched_image_paths: paths of all images as a list of lists (or similar)
    :param random_subset_index: a randomly selected index (integer value)
    :param mean_activity_all_images: the mean activity of the entire training set
    :param rows_to_sample: Number of random rows to sample (ideally >0.1% of the total
    number of rows)
    :return: the estimated Q1, median, and Q3 values of three rows as a tuple of
    three numpy arrays
    """

    # the index above is used below to select one of n_chunks
    tensors_for_pca = [
        prep_data_for_pca(i, mean_activity_all_images)
        for i in batched_image_paths[random_subset_index]
    ]
    all_stacked_for_pca = torch.vstack(tensors_for_pca)

    # selecting n_random_samples amount of random indices
    total_range_of_rows = range(int(all_stacked_for_pca.shape[0]))
    randomly_chosen_row_nums = np.random.choice(
        total_range_of_rows,
        size=rows_to_sample,
        replace=False
    )

    # extracting rows at those random indices and stacking them as a tensor
    randomly_stacked_subset = torch.vstack(
        [all_stacked_for_pca[i] for i in randomly_chosen_row_nums]
    )

    # torch median also returns indices, which isn't necessary for this test, hence [0]
    median_of_randomly_stacked_subset = np.median(randomly_stacked_subset.numpy(), axis=0)

    # finding the mean; torch's mean function requires dtype float32 (or better)
    mean_of_randomly_stacked_subset = torch.mean(
        randomly_stacked_subset.to(torch.float32),
        dim=0
    ).numpy()

    # finding the standard deviation; torch's SD function requires float32 (or better)
    std_of_randomly_stacked_subset = torch.std(
        randomly_stacked_subset.to(torch.float32),
        dim=0
    ).numpy()

    # torch quantile fails for even 1/1000th of the dataset
    # so using the numpy equivalent here
    q1_of_randomly_stacked_subset = np.percentile(
        randomly_stacked_subset.numpy(),
        q=25,
        axis=0
    )

    q3_of_randomly_stacked_subset = np.percentile(
        randomly_stacked_subset.numpy(),
        q=75,
        axis=0
    )

    return (
        mean_of_randomly_stacked_subset,
        std_of_randomly_stacked_subset,

        q1_of_randomly_stacked_subset,
        median_of_randomly_stacked_subset,
        q3_of_randomly_stacked_subset
    )


if __name__ == "__main__":
    # setting a random seed
    np.random.seed(234213)

    # defining the total number of cores
    number_of_cores = 10

    # compiling the paths to all the training images in a list
    training_images_output_path = "../images/training_images_processed/"
    training_images_processed_path = [
        training_images_output_path + f for f in os.listdir(training_images_output_path)
        if f.endswith('.JPEG')
    ]

    # splitting the path of images into n_chunks
    n_chunks = 100
    image_paths_as_many_chunks = chunkify(
        training_images_processed_path,
        n_chunks
    )

    # randomly selecting number_of_cores amount of indices
    randomly_select_subset_indices = np.random.choice(
        range(len(image_paths_as_many_chunks)),
        size=number_of_cores,
        replace=False
    )

    # loading the mean activity of the image computed
    # under bullet point 1 of augment_ingest.ipynb
    mean_activity_all_training = torch.load(
        "../images/mean_activity_of_all_training_images.pt"
    )

    # fixing the number of random samples
    n_random_samples = 10000000

    # compiling arguments for the function
    # the only argument that varies for each sub-tuple is the index
    arguments_for_quantiles = [
        (
            image_paths_as_many_chunks,
            random_index.item(),
            mean_activity_all_training,
            n_random_samples
        )
        for random_index in randomly_select_subset_indices
    ]

    ### initially planned to run the code in parallel but faced memory constraints
    # # running the function n times on n cores
    # with Pool(number_of_cores) as pool:
    #     results = pool.starmap(
    #         estimate_quantiles_for_robust_scaling,
    #         arguments_for_quantiles
    #     )

    results = []

    for n in tqdm(range(number_of_cores)):
        results.append(
            estimate_quantiles_for_scaling(
                batched_image_paths=arguments_for_quantiles[n][0],
                random_subset_index=arguments_for_quantiles[n][1],
                mean_activity_all_images=arguments_for_quantiles[n][2],
                rows_to_sample=arguments_for_quantiles[n][3],
            )
        )

    # the estimated mean from random sampling seems less reliable
    # see section 3.x in augment_ingest.ipynb for more information

    # unpacking the results
    _, std_list, q1_list, median_list, q3_list = zip(*results)

    # converting to numpy arrays and reshape
    # mean_list = np.array(_).reshape(-1, 3)
    std_list = np.array(std_list).reshape(-1, 3)
    q1_array = np.array(q1_list).reshape(-1, 3)
    median_array = np.array(median_list).reshape(-1, 3)
    q3_array = np.array(q3_list).reshape(-1, 3)

    # creating a polars dataframe
    results_as_df = pl.DataFrame({
        # 'mean_R': mean_list[:, 0],
        # 'mean_G': mean_list[:, 1],
        # 'mean_B': mean_list[:, 2],

        'std_R': std_list[:, 0],
        'std_G': std_list[:, 1],
        'std_B': std_list[:, 2],

        'q1_R': q1_array[:, 0],
        'q1_G': q1_array[:, 1],
        'q1_B': q1_array[:, 2],
        'median_R': median_array[:, 0],
        'median_G': median_array[:, 1],
        'median_B': median_array[:, 2],
        'q3_R': q3_array[:, 0],
        'q3_G': q3_array[:, 1],
        'q3_B': q3_array[:, 2]
    })

    results_as_df.write_csv("estimated_quantile_values.csv")

    # finding the median value from n runs for all quantiles
    # best_estimated_means = (
    #     results_as_df.select(pl.all().median())[["mean_R", "mean_G", "mean_B"]]
    #     .to_torch()
    #     .to(torch.float32)
    # )

    best_estimated_stds = (
        results_as_df.select(pl.all().median())[["std_R", "std_G", "std_B"]]
        .to_torch()
        .to(torch.float32)
    )

    best_estimated_q1s = (
        results_as_df.select(pl.all().median())[["q1_R", "q1_G", "q1_B"]]
        .to_torch()
        .to(torch.int16)
    )

    best_estimated_medians = (
        results_as_df.select(pl.all().median())[["median_R", "median_G", "median_B"]]
        .to_torch()
        .to(torch.int16)
    )

    best_estimated_q3s = (
        results_as_df.select(pl.all().median())[["q3_R", "q3_G", "q3_B"]]
        .to_torch()
        .to(torch.int16)
    )

    # saving the estimated quantiles as tensors
    # torch.save(best_estimated_means, "best_estimated_means.pt")
    torch.save(best_estimated_stds, "best_estimated_stds.pt")
    torch.save(best_estimated_q1s, "best_estimated_q1s.pt")
    torch.save(best_estimated_medians, "best_estimated_medians.pt")
    torch.save(best_estimated_q3s, "best_estimated_q3s.pt")
