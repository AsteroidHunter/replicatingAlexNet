# Replicating AlexNet

<img src="https://img.shields.io/badge/Status-Complete-green" alt="Status" height="40">
<img src="https://img.shields.io/badge/Currently_Working_On-Nil-8A2BE2" alt="Currently Working On" height="30">

This repository contains my attempt at replicating [Krizhevsky et al. (2012)](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf). I chose AlexNet because it is a foundational paper in deep learning and replicating it would allow me to get a better grasp of basic research engineering in DL.

### Results

<center>

|Type of Error|Krizhevsky et al. (2012)| This Replication|Difference|
|----|----|----|----|
|Top-1|37.5%|40.25%|**2.75%**|
|Top-5|17.5%|19.68%|**2.18%**|

</center>

<div style="text-align: center; width: 100%; display: inline-block;">
<p style="max-width: 300px; margin: 0 auto; text-align: justify;">

**Table 1.** Table comparing the top-1 and top-5 percentage errors between the original paper and this replication. These values were computed using the *ten-patch prediction averaging* method described in section 4.1 of the paper. The code can be found in the script titled `val_acc_multi.py`.

</p>
</div>

![](./plots/metrics_second_hpc_run.png)

<p style="font-size: 0.85em; text-align: justify;">
<b><i>Note:</b></i> The validation accuracy shown in the graph above was calculated by cropping a 224x224-sized, center portion of the validation image. Extracting ten patches from each validation image and averaging the model's predictions on all these patches was comparatively computationally expensive, and thus, was avoided during the training run.
</p>

### Brief description of the main files/folders
1. `./images/`: This folder contains images that were used for testing some of the preprocessing steps. Locally, this folder also contains the training, testing, validation, and patch images files. The latter set of files weren't pushed as they are collectively over 140 GB in size.
2. `./devkit-1.0/data/`: This folder contains the ground truth labels for the validation images. The data was downloaded from the [ImageNet competition website](https://image-net.org/challenges/LSVRC/2010/index.php).
3. `preprocessing_playground.ipynb`: Contains code that was used to unzip the training images, replace damaged images with the provided patch images, removing corrupted images (turns out, there was only one such image), and tests related to rescaling/cropping images as described in the paper and mean image substraction.
4. `preprocessing.py`: This script contains cleaned up code from the above mentioned python notebook which was used to rescales and crops 1.2 million training images, as well as the validation and testing images.
5. `augment_ingest.ipynb`: This notebook is partly a continuation of `preprocessing_playground.ipynb` as the mean activity of all the training images were computed here and then saved. Tests related to augmentations — random sampling of image portions to expand the training dataset by a factor of 2048 and PCA on all RGB pixels — were performed in here as well. Lastly, sysnsets and lemmas related to the training labels were explored, and a CSV with image names, training labels, and other information were created and saved. *This is the bulkiest notebook in the repository.*
6. `./pca_business/`: This folder contains cleaned up scripts that were used to compute the mean and quantiles for scaling the data and then performing Incremental PCA on all RGB pixels of the training images + the saved eigenvectors and eigenvalues from the PCA run.
7. `alexnet_torch.ipynb`: Contains a PyTorch-ified version of the 2-GPU neural network architechture as defined in the paper, functions that augment the images on the fly, trial training runs on a subset of the entire training set, and some experimentation with the hash-based augmentation routine.
8. `alexnet_torch_hpc_run.py`: The final, cleaned training script with the augmentation steps, model, model training, et cetera.
9. `training_script.slurm`: SLURM script that was used for executing the job on the HPC.
10. `val_acc_multi.py`: Script for parallel-y computing the top-1 and top-5 accuracy rates using the *ten patch prediction averaging* method described in section 4.1 of the paper.
11. `./logs/`: Folder that contains data from various training runs as well as the model weights and the state of the optimizer.
12. `./plots`: self-explanatory

### To-do's
- [x] Modifying the *2048x augmentation* such that each batch contains a diverse set of images
- [x] Trialing a small run on the university's High Performance Computer (HPC)
- [x] Running the model on the entire training set on the HPC
- [x] Fixing the >12% difference in top-1 accuracy between the original model and this one

### A succinct delineation of steps I followed
(Will be added later.)
