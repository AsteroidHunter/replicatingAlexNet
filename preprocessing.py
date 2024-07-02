"""
This is the final script for rescaling and cropping the training images for AlexNet.

The script creates a list of paths to all the training images, splits them into n chunks
(where n is the number of cores available), and then rescales/crops them to 256 x 256 sized
images. The runtime is ~12 minutes for 1.2 million images being preprocessed on ten
cores.

For more information and output checks, see preprocessing_playground.ipynb.
"""

import os
from multiprocessing import Pool
from PIL import Image


def process_images_for_alexnet(
        image_list,
        image_name_list,
        location_to_save,
        new_width=256,
        new_height=256,

):
    """
    This function preprocesses images as defined in section 2 of Krizhevsky et al. (2012)
    It rescales and converts the images provided into the size 256 x 256. If the image
    is rectangular, the short side is first scaled to 256 pixels and then the center
    portion of the image is cropped such that it is a square image. Square images are
    simply upscaled or downscaled. The lanzcos resampling routine is used because during
    testing, it subjectively appeared to preserve quality better than other available
    resampling methods.

    :param image_list: List of paths to the images
    :param image_name_list: List of names of the images
    :param location_to_save: Directory where preprocessed images will be saved
    :param new_width: Transformed width of the image, defaults to 256
    :param new_height: Transformed width of the image, defaults to 256
    :return: The function saves the images after transforming them
    """
    for image_single, image_name_single in zip(image_list, image_name_list):

        one_image = Image.open(image_single)
        width, height = one_image.size

        # the three situations, which ended up being kind of redundant
        large_square_image = width >= 256 and height >= 256 and width == height
        small_square_image = width < 256 and height < 256 and width == height
        rectangular_image = width != height

        if rectangular_image:

            # rescaling such that the smaller side is 256 pixels in length
            if width < height:
                width_ratio = new_width / width
                scaled_height = int(height * width_ratio)
                transformed_image = one_image.resize((new_width, scaled_height),
                                                     Image.LANCZOS)
            else:
                height_ratio = new_height / height
                scaled_width = int(width * height_ratio)
                transformed_image = one_image.resize((scaled_width, new_height),
                                                     Image.LANCZOS)

            width_transformed, height_transformed = transformed_image.size

            # cropping the 256 x 256 portion in the middle
            left = (width_transformed - new_width) // 2
            top = (height_transformed - new_height) // 2
            right = left + new_width
            bottom = top + new_height

            transformed_image = transformed_image.crop((left, top, right, bottom))

        elif large_square_image or small_square_image:
            # here, the first argument is size=(W, H) and the second one is resample
            transformed_image = one_image.resize((256, 256), Image.LANCZOS)

        transformed_image.save(f"./{location_to_save}/{image_name_single}")


# defining the path to all the training images
training_images_path = "./images/training_images/"

# creating list of paths to all the images and all images names
training_images_path_all = [
    training_images_path + f for f in os.listdir(training_images_path)
    if f.endswith(".JPEG")
]

training_images_just_names = [
    f for f in os.listdir(training_images_path)
    if f.endswith(".JPEG")
]


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


# creating a directory to store the outputs
try:
    os.mkdir("./images/training_images_processed/")
except FileExistsError:
    pass

training_images_output_path = "./images/training_images_processed/"

if __name__ == "__main__":
    number_of_cores = 10

    # splitting the paths and names into n portions
    path_chunks = chunkify(training_images_path_all, number_of_cores)
    name_chunks = chunkify(training_images_just_names, number_of_cores)

    # list of arguments for the all the tasks
    arguments = [
        (path_chunks[i], name_chunks[i], training_images_output_path, 256, 256)
        for i in range(number_of_cores)
    ]

    with Pool(number_of_cores) as pool:
        pool.starmap(process_images_for_alexnet, arguments)
