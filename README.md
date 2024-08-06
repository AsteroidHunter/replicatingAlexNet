# Replicating AlexNet

<img src="https://img.shields.io/badge/Status-In_Progress-orange" alt="Status" height="40">
<img src="https://img.shields.io/badge/Currently_Working_On-Prepping_model_for_the_HPC-8A2BE2" alt="Currently Working On" height="30">

This repository contains my attempt at replicating [Krizhevsky et al. (2012)](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf). I chose AlexNet because it is a foundational paper in deep learning and replicating it would allow me to get a better grasp of basic research engineering in DL.

### Brief description of the main files/folders
1. `./images/`: This folder contains images that were used for testing some of the preprocessing steps. Locally, this folder also contains the training, testing, validation, and patch images files. The latter set of files weren't pushed as they are collectively over 140 GB in size.
2. `preprocessing_playground.ipynb`: Contains code that was used to unzip the training images, replace damaged images with the provided patch images, removing corrupted images (turns out, there was only one such image), and tests related to rescaling/cropping images as described in the paper and mean image substraction.
3. `preprocessing.py`: This script contains cleaned up code from the above mentioned python notebook which was used to rescales and crops 1.2 million training images, as well as the validation and testing images.
4. `augment_ingest.ipynb`: This notebook is partly a continuation of `preprocessing_playground.ipynb` as the mean activity of all the training images were computed here and then saved. Tests related to augmentations — random sampling of image portions to expand the training dataset by a factor of 2048 and PCA on all RGB pixels — were performed in here as well. Lastly, sysnsets and lemmas related to the training labels were explored, and a CSV with image names, training labels, and other information were created and saved. *This is the bulkiest notebook in the repository.*
5. `./pca_business/`: This folder contains cleaned up scripts that were used to compute the mean and quantiles for scaling the data and then performing Incremental PCA on all RGB pixels of the training images + the saved eigenvectors and eigenvalues from the PCA run.
6. `alexnet_torch.ipynb`: So far, this contains a PyTorch-ified version of the 2-GPU neural network architechture as defined in the paper, functions that augment the images on the fly, and trial training runs on a subset of the entire training set.

### To-do's
- [x] Modifying the *2048x augmentation* such that each batch contains a diverse set of images
- [x] Trialing a small run on the university's High Performance Computer (HPC)
- [ ] Running the model on the entire training set on the HPC 

### A succinct delineation of steps I followed
(Will be added after the model starts training on the HPC.) 