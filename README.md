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

### Reflection and Notes
I worked on this project primarily from the ~25th of June to the ~8th of August, 2024. I spent a couple of hours the week of 19th of August ironing out last doubts/issues with the code, and trained the model twice — once on the 22nd of August and another on 12th of September (with some minor fixes). The last training run, which took around two days on my university’s high performance computer, yielded adequate results.

For the duration of those ~6 weeks, 25 hours a week (on average) was spent working on this project. Based on my tracked pomodoro's and logs, the majority of time was spent on debugging — 
- Programming (30%): Trying out different approaches to code a particular section of the paper
- Learning (30%): Reading tutorials on how to use a particular library and reading about and refreshing my memory of foundational ML concepts before implementing them (for instance, before implementing the PCA portion of the paper, I spent ~5-7 hours watching lectures on PCA and related linear algebra concepts).
- Debugging and clarification (40%): Issues I ran into are outlined in detail below.

#### Steps involved
1. **Data preparation and preprocessing**: After downloading the entire ImageNet 2010 dataset, I resized the images such that at least one side was 256 pixels long, and then cropped the center portion of the image. There were also ~2000 non-RGB images, which were converted to RGB for consistency. For faster execution, I used multiprocessing to modify these images. The cleaned-up code can be found in [preprocessing.py](https://github.com/AsteroidHunter/replicatingAlexNet/blob/aabcf980a2d38f8837b154f00419900749f2f65d/preprocessing.py) and the notebook where I initially explored all this is [preprocessing_playground.ipynb](https://github.com/AsteroidHunter/replicatingAlexNet/blob/aabcf980a2d38f8837b154f00419900749f2f65d/preprocessing_playground.ipynb). After preprocessing, I computed the mean activity of all images using a running mean estimation, which took some time but was also quite simple (Part 1 of [augment_ingest.ipynb](https://github.com/AsteroidHunter/replicatingAlexNet/blob/aabcf980a2d38f8837b154f00419900749f2f65d/augment_ingest.ipynb).

2. **Principal Component Analysis of R, G, & B channels**: This portion was a bit more challenging primarily because of the sheer size of the data. After resizing all images as tensors with three columns for each channel, I sampled ~0.01%-0.1% of the data to estimate the quantiles for each channel. I used a running mean estimation to compute the mean of all three columns. Then, I used the obtained value to scale the data before feeding it to SciKit’s incremental PCA algorithm. The cleaned-up code for the same can be found in the folder [./pca_business/](https://github.com/AsteroidHunter/replicatingAlexNet/tree/aabcf980a2d38f8837b154f00419900749f2f65d/pca_business), and initial failed attempts, exploration, and validation of the scaling done can be found in section 3 of [augment_ingest.ipynb](https://github.com/AsteroidHunter/replicatingAlexNet/blob/aabcf980a2d38f8837b154f00419900749f2f65d/augment_ingest.ipynb). The obtained eigenv✶s were multiplied with random values sampled from a Gaussian and then added to the R, G, and B channels of the input images during augmentation.

3. **Setting up the model**: I wanted to implement model parallelism as originally done in the paper. I used the `torch.nn` module to define identical layers on both GPUs; at certain layers, I concatenate the output from the two GPUs and pass this concatenated output to the next layer. While PyTorch makes implementation of such split models quite easy, this part took me quite a while because of issues highlighted in the section below. The final model used can be found in [alexnet_torch_hpc_run.py](https://github.com/AsteroidHunter/replicatingAlexNet/blob/aabcf980a2d38f8837b154f00419900749f2f65d/alexnet_torch_hpc_run.py).

4. **On-the-fly augmentation**: One of the most time-consuming part was designing a data augmentation algorithm that could theoretically produce up to 2048 unique variations of the same image and pass them to the DataLoader object without any repetition. Ultimately, I settled for a hash-based technique, where I generate a random seed using the hash value obtained from the image filename, and then the generated random seed is used to create N unique image corner coordinates (where N is the number of training epochs). The image corner coordinates are passed to an augmentation function which crops a 224 x 224 portion from the 256 x 256-sized image. The final implementation can be found in [alexnet_torch_hpc_run.py](https://github.com/AsteroidHunter/replicatingAlexNet/blob/aabcf980a2d38f8837b154f00419900749f2f65d/alexnet_torch_hpc_run.py)., and some other attempts and comments can be found in [alexnet_torch.ipynb](https://github.com/AsteroidHunter/replicatingAlexNet/blob/aabcf980a2d38f8837b154f00419900749f2f65d/alexnet_torch.ipynb).

For the final training run, `alexnet_torch_hpc_run.py` was used, and the final validation code is in [val_acc_multi.py.](https://github.com/AsteroidHunter/replicatingAlexNet/blob/aabcf980a2d38f8837b154f00419900749f2f65d/val_acc_multi.py). The top-1 validation error of my replication was 40.25% (37.5% in the original paper) and the top-5 validation error was 19.68% (17.5% in the original paper). 

Training logs and model weights are present in the folder titled ./logs/. In hindsight, the final training script is too long and could have been truncated by having a separate script where the model is defined.

#### Issues faced
Here is a list of a few minor/major hurdles I faced while re-implementing the paper:
1. Image issues: dealing with the one corrupted image that won't open but that trashed the mean activity computation, finding and converting all non-RGB images to RGB images, the actual input image size, et cetera.
    - Re input image size: there is quite a bit of confusion here, as the original paper states the size should be 224 x 224 while some others, citing Karpathy, think it should be 227 x 227. I don't know if the latter is plausible because in the data augmentation section, the authors mention that there are 1024 ways of sampling a 224 x 224 image from a 256 x 256 preprocessed image. If the actual image input image size were 227 x 227, then there would be only ~900 possible ways of sampling such images from a 256 x 256 one. Maybe, the "1024" figure in the paper is also erroneous.
2. PCA on all images without stacking 1.2 million tensors
    - Scaling tensors for PCA by estimating the quantiles and the mean (as `scipy`'s `StandardScaler` caused memory overload issues); also, turns out robust scaling produces wildly different results.
3. Cross-GPU interaction issues on the high-performance computer: I implemented a backward hook and noticed that the gradients wouldn't backpropagate for the layers residing on the second GPU. quite annoyingly, this consumed a week of time as I tried to correct how my model was being split across the two GPUs and investigate CUDA packages and such on the HPC. Confusingly, the same model worked fine when split across the CPU and the GPU on my local machine. Ultimately, I switched to a different compute cluster within the HPC which had a recently added support for multi-GPU runs, and the model worked perfectly fine on that machine.
4. Retrieving and investigating training labels (lemmas from synsets); ended up being not necessary for training but useful for validating if my training and validation dataloaders were set up properly.
5. Model shape error: I was splitting the input image for the top and bottom layers due to an alternate interpretation of the statement “One GPU runs the layer-parts at the top of the figure while the other runs the layer-parts at the bottom.” in Figure 2 of the paper.
7. Zero loss issue during test runs
    - Masking value? No
    - Not enough data? No
    - Something fundamental in the model? No
		- Tested using a slighltly modified Torch version of AlexNet
	- Labels? Yes
		- They need to be relabeled to 0-{total number of classes}
11. Static loss issue
	- Things that I checked and fixed but weren't causing the issue:
		- Data imbalance
		- Learning rate or hyperparameters
		- The overall model architecture
		- Mis-indexing, DataLoader, etc.
	- *Thing that was causing the issue*: returning the softmax outputs instead of the logits...
12. 2048x augmentation
    - I figured out two ways to return 2048 augmented versions of the training images in the dataset without any repetition, but running the modified Dataset class with this augmentation crashed my machine. After doing a back-of-the-envelope, it seemed pretty clear that the authors did not use 2048 variations of all training images during every run (if they actually did so, it would have taken them ~50 days to train the model!). Instead, they used one variation of every image during each training run. Hypothetically, if they ran the model for 2048 epochs, then the model would see a dataset which was enlarged by a factor of 2048; but this isn't stated clearly in the paper.
    - I don't think I am the only person who took the authors' statement this literally. For instance, [this review](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-019-0197-0) cites the 2048 enlargement verbatim, so do several bloggers trying to summarize the paper. [This redditor](https://www.reddit.com/r/MLQuestions/comments/ahu204/how_long_does_it_take_to_run_an_epoch_of_alexnet/) had the same doubt as I did; thankfully, a commenter was able to help them out.
    - The model I ran, and the ones the authors ran, saw 90 unique variations of all the training images.

I am unsure if this paper was the optimal choice for replication. No one seems to do model parallelism anymore (or at least, [not in the way done by the authors](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/)), so clarifying doubts was a bit challenging. The same goes for other key doubts, such as the image input size and the dataset enlargement routine. The sheer size of the data made what would have been otherwise easy steps more challenging (but this was a choice). I predict that a deep learning paper dealing with textual data would have taken me half as much time to finish, and I would have likely learned as much as I did from this endeavor. 