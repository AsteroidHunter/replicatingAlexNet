# Replicating AlexNet

<img src="https://img.shields.io/badge/Status-In_Progress-orange" alt="Status" height="40">
<img src="https://img.shields.io/badge/Currently_Working_On-PCA_of_RGB_values-blue" alt="Currently Working On" height="30">

This repository contains my attempt at replicating [Krizhevsky et al. (2012)](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf). I chose AlexNet because it is a foundational paper in deep learning and replicating it would allow me to get a better grasp of the basics and some research engineering in DL.

### Description of files
1. `preprocessing_playground.ipynb`: Contains code that was used to unzip the training images, replacing damaged images with the provided patch images, removing corrupted images (turns out, there was only one such image), a function for rescaling and cropping images as described in the paper, mean image substraction, and tests related to the previous two preprocessing steps 
2. `preprocessing.py`: This script contains code from the above mentioned python notebook, and it rescales and crops 1.2 million training images using the multiprocessing library
3. `augment_ingest.ipynb`: In this notebook, I computed the mean activity of all the training images and then saved them. I also tested a few ways to sample 1024 random crops of an input image (to increase the training data size as done in the paper) and perform principal component analysis of the RGB pixel values.
4. `alexnet_torch.ipynb`: So far, this notebook contains portions of the neural network architechture employed by the authors.
5. `images`: In the repository, this folder contains images that were used for testing some of the preprocessing steps. Locally, this folder also contains the training, testing, validation, and patch images files. The latter set of files weren't pushed as they are collectively over 140 GB in size.

### A succinct delineation of steps I followed
(Will be added after the model starts training) 

### Time spent:
- Preprocessing: 10 pomodoro sessions (~250 minutes)
