# Replicating AlexNet

<img src="https://img.shields.io/badge/Status-In_Progress-orange" alt="Status" height="40">
<img src="https://img.shields.io/badge/Currently_Working_On-Defining_the_Model-blue" alt="Currently Working On" height="30">

This repository contains my attempt at replicating [Krizhevsky et al. (2012)](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf). I chose AlexNet because it is a foundational paper in deep learning and replicating it would allow me to get a better grasp of the basics and some research engineering in DL.

### Description of files
1. `preprocessing_playground.ipynb`: Contains code that was used to unzip the training images, replacing damaged images with the provided patch images, removing corrupted images (turns out, there was only one such image), a function for rescaling and cropping images as described in the paper, mean image substraction, and tests related to the previous two preprocessing steps 
2. `preprocessing.py`: This script contains code from the above mentioned python notebook, and it rescales and crops 1.2 million training images using the multiprocessing library
3. `alexnet_torch.ipynb`: So far, this notebook contains portions of the neural network architechture employed by the authors.

### Time spent:
- Preprocessing: 10 pomodoro sessions (~250 minutes)
