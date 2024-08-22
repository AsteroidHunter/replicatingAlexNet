"""
__author__: akash

A 2-GPU implementation of AlexNet (Krizhevsky et al., 2012). 

This script relies on files that were created in ~3 different python notebooks.
Check the repository's README for more details.

Link to the original publication: 
https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
"""


import os
import csv
import torch
import random
import hashlib
import numpy as np
import polars as pl
from PIL import Image
import torch.nn as nn
import torch.nn.functional as nnF
from datetime import date, datetime
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader, Dataset


random.seed(234213)
np.random.seed(234213)

device1 = torch.device("cuda:0")
device2 = torch.device("cuda:1")
device3 = torch.device("cpu")


#### DATA LOADING BLOCK

# collecting training image paths and names into lists
training_images_output_path = "./images/training_images_processed/"

# collecting validation image paths and names into lists
validation_images_output_path = "./images/validation_images_processed/"

validation_images_processed_path = [
    validation_images_output_path + f for f in os.listdir(validation_images_output_path)
    if f.endswith('.JPEG')
]

validation_images_processed_names = [
    f for f in os.listdir(validation_images_output_path)
    if f.endswith('.JPEG')
]

# sorting names in ascending order; this is crucial to ensure that the
# proper validation labels are attached to the file names
validation_images_processed_names = sorted(validation_images_processed_names)

# loading the validation IDs provided in the Image Net devkit
validation_ILSVRC2010_IDs = []

with open("./devkit-1.0/data/ILSVRC2010_validation_ground_truth.txt") as file:
    while line := file.readline():
        validation_ILSVRC2010_IDs.append(int(line.rstrip()))

# converting the IDs to labels for the validation data
validation_true_labels = [i - 1 for i in validation_ILSVRC2010_IDs]


# loading the saved mean activity from section 1.1 of augment_ingest.ipynb
mean_activity_all_training = torch.load(
    "./images/mean_activity_of_all_training_images.pt",
    weights_only=False
)

# loading the computed PCA eigenv*s values
# For more information, see ./pca_business/pca_rgb_multiples.py 
# and section 3 of augment_ingest.ipynb
training_eigenvectors = torch.load("./pca_business/eigenvectors_from_ipca.pt",
                                  weights_only=False).to(torch.float32)
training_eigenvalues = torch.load("./pca_business/eigenvalues_from_ipca.pt",
                                  weights_only=False).to(torch.float32)

# the wordnet IDs were mapped to values between 0-999 
# the latter will be used as the training labels
# loading a file created in section 4 of augment_ingest.ipynb
labels_w_information = pl.read_csv("./images/labels_w_information.csv")

# getting training labels
training_labels_for_CEloss = labels_w_information["made_up_label"].to_list()


# dictionary with names and labels for the training and valiadation sets
label_from_name_dictionary = {
    name: label for name, label 
    in zip(labels_w_information["image_name"], labels_w_information["made_up_label"])
}

validation_label_from_name_dictionary = {
    image_name: label for image_name, label 
    in zip(validation_images_processed_names, validation_true_labels)
}


#### DEFINING THE MODEL

class AlexNet(nn.Module):

    """
    A 2-GPU AlexNet model. 
    
    The top layers of the model reside on the first GPU, and the bottom layers
    reside on the second GPU. The input image is fed to both the layers. 
    However, for the top layer, the bottom half is masked (pixel values are
    multiplied by `mask_value=0`) and vice-versa. Masking was done because 
    in the paper, the authors' stated that "One GPU runs the layer-parts at
    the top of the figure while the other runs the layer-parts at the bottom." 
    The masking approach was the best solution I could devise to replicate what
    the author's did in the original publication.
    
    There are a few redundant function calls & 
    repeated output concatenations in this model; I have left them in place as
    they are good for visually tracking how the layers are set up and avoiding
    GPU-layer mismatch errors.
    """
    
    def __init__(self, number_of_classes=1000, mask_value=0):
        super().__init__()

        self.number_of_classes = number_of_classes
        self.mask_value = mask_value
        
        self.conv1_top = nn.Conv2d(
            in_channels=3,
            out_channels=48,
            kernel_size=11,
            stride=4,
            padding=2,
        ).to(device1)

        self.conv1_bottom = nn.Conv2d(
            in_channels=3,
            out_channels=48,
            kernel_size=11,
            stride=4,
            padding=2,
        ).to(device2)

        self.conv2_top = nn.Conv2d(
            in_channels=48,
            out_channels=128,
            kernel_size=5,
            padding=2,
        ).to(device1)

        self.conv2_bottom = nn.Conv2d(
            in_channels=48,
            out_channels=128,
            kernel_size=5,
            padding=2,
        ).to(device2)

        self.conv3_top = nn.Conv2d(
            in_channels=256,
            out_channels=192,
            kernel_size=3,
            padding=1,
        ).to(device1)

        self.conv3_bottom = nn.Conv2d(
            in_channels=256,
            out_channels=192,
            kernel_size=3,
            padding=1,
        ).to(device2)

        self.conv4_top = nn.Conv2d(
            in_channels=192,
            out_channels=192,
            kernel_size=3,
            padding=1,
        ).to(device1)

        self.conv4_bottom = nn.Conv2d(
            in_channels=192,
            out_channels=192,
            kernel_size=3,
            padding=1,
        ).to(device2)

        self.conv5_top = nn.Conv2d(
            in_channels=192,
            out_channels=128,
            kernel_size=3,
            padding=1,
        ).to(device1)

        self.conv5_bottom = nn.Conv2d(
            in_channels=192,
            out_channels=128,
            kernel_size=3,
            padding=1,
        ).to(device2)

        self.dense6_top = nn.Linear(
            in_features=256 * 6 * 6, 
            out_features=2048,
        ).to(device1)

        self.dense6_bottom = nn.Linear(
            in_features=256 * 6 * 6,
            out_features=2048,
        ).to(device2)

        self.dense7_top = nn.Linear(
            in_features=4096, 
            out_features=2048,
        ).to(device1)

        self.dense7_bottom = nn.Linear(
            in_features=4096, 
            out_features=2048,
        ).to(device2)

        self.dense_last = nn.Linear(
            in_features=4096, 
            out_features=self.number_of_classes,
        ).to(device2)

        # local response normalization layer with 
        # hyperparameters as described in the paper
        self.lrn_top = nn.LocalResponseNorm(
            size=5,
            alpha=0.0001, 
            beta=0.75, 
            k=2
        ).to(device1)
        
        self.lrn_bottom = nn.LocalResponseNorm(
            size=5,
            alpha=0.0001, 
            beta=0.75, 
            k=2
        ).to(device2)

        # not all layer details are mentioned in the paper
        # For the kernel size, I downloaded and referenced 
        # the model wiki provided by the authors
        # see file "LayerParams.wiki.ini"
        self.maxpool_top = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
        ).to(device1)
        
        self.maxpool_bottom = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
        ).to(device2)

        self.dropout_top = nn.Dropout(
            p=0.5,
        ).to(device1)
        
        self.dropout_bottom = nn.Dropout(
            p=0.5,
        ).to(device2)

    def mask_input(self, x, mask_value=None):
        if mask_value is None:
            mask_value = self.mask_value
            
        x_device = x.device
        *_, height, width = x.shape
        
        upper_mask = torch.ones((height, width))
        upper_mask[int(height / 2):, :] = mask_value
        
        lower_mask = torch.zeros((height, width))
        lower_mask[:int(height / 2), :] = mask_value

        upper_mask = upper_mask.to(x_device)
        lower_mask = lower_mask.to(x_device)
        
        return x * upper_mask, x * lower_mask
    
    def forward(self, x):
        
        top_image, bottom_image = self.mask_input(x)
        top_image = top_image.to(device1)
        bottom_image = bottom_image.to(device2)


        # first layer
        top_image = nnF.relu(self.conv1_top(top_image.to(device1))).to(device1)
        bottom_image = nnF.relu(self.conv1_bottom(bottom_image.to(device2))).to(device2)

        ## response norm after first layer
        top_image = self.lrn_top(top_image.to(device1))
        bottom_image = self.lrn_bottom(bottom_image.to(device2))

        ## max pooling after first response norm
        top_image = self.maxpool_top(top_image.to(device1))
        bottom_image = self.maxpool_bottom(bottom_image.to(device2))

        
        # second layer
        top_image = nnF.relu(self.conv2_top(top_image.to(device1)))
        bottom_image = nnF.relu(self.conv2_bottom(bottom_image.to(device2)))

        ## response norm after second layer
        top_image = self.lrn_top(top_image.to(device1))
        bottom_image = self.lrn_bottom(bottom_image.to(device2))

        ## max pooling after second response norm
        top_image = self.maxpool_top(top_image.to(device1))
        bottom_image = self.maxpool_bottom(bottom_image.to(device2))

        # my best interpretation of how the output is passed to the third layer
        top_image_full = torch.cat(
            (top_image.to(device1), bottom_image.to(device1)), 
            dim=1
        )
        
        bottom_image_full = torch.cat(
            (top_image.to(device2), bottom_image.to(device2)), 
            dim=1
        )

        # third layer
        top_image = nnF.relu(self.conv3_top(top_image_full))
        bottom_image = nnF.relu(self.conv3_bottom(bottom_image_full))

        
        # fourth layer
        top_image = nnF.relu(self.conv4_top(top_image).to(device1))
        bottom_image = nnF.relu(self.conv4_bottom(bottom_image).to(device2))

        
        # fifth layer
        top_image = nnF.relu(self.conv5_top(top_image).to(device1))
        bottom_image = nnF.relu(self.conv5_bottom(bottom_image).to(device2))

        
        # my best interpretation of how the output is passed to the sixth layer
        top_image_full = torch.cat(
            (top_image.to(device1), bottom_image.to(device1)), 
            dim=1
        )
        
        bottom_image_full = torch.cat(
            (top_image.to(device2), bottom_image.to(device2)), 
            dim=1
        )

        # applying max pooling with masked image inputs 
        # if the input is split into two halves vertically, then
        # average pooling should be used to avoid dimension errors
        top_image_full = self.maxpool_top(top_image_full) 
        bottom_image_full = self.maxpool_bottom(bottom_image_full)
        
        top_image_full = top_image_full.reshape(top_image_full.size(0), -1)
        bottom_image_full = bottom_image_full.reshape(bottom_image_full.size(0), -1)

        # sixth layer
        top_image_full = self.dropout_top(top_image_full)
        top_image = nnF.relu(self.dense6_top(top_image_full))
        
        bottom_image_full = self.dropout_bottom(bottom_image_full)
        bottom_image = nnF.relu(self.dense6_bottom(bottom_image_full))


        top_image_full = torch.cat(
            (top_image.to(device1), bottom_image.to(device1)), 
            dim=1
        )
        
        bottom_image_full = torch.cat(
            (top_image.to(device2), bottom_image.to(device2)), 
            dim=1
        )

        # seventh layer
        top_image_full = self.dropout_top(top_image_full)
        top_image = nnF.relu(self.dense7_top(top_image_full))
        
        bottom_image_full = self.dropout_bottom(bottom_image_full)
        bottom_image = nnF.relu(self.dense7_bottom(bottom_image_full))
        
        
        # final layer
        combined_output = torch.cat(
            (top_image.to(device2), bottom_image.to(device2)), 
            dim=1
        )
        
        x = self.dense_last(combined_output)
        
        return x.to(device2)

def initialize_weights_biases(model):
    """
    This function initializes the weights and biases 
    of the AlexNet model as described in the paper.
    """
    for name, module in model.named_modules():
        
        # initializing weight ~ N(0, 0.01 in all layers)
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.normal_(module.weight, mean=0, std=0.01)
        
        # intializing biases in the 2nd, 4th, and 5th layer
        if name == "conv2_top" or name == "conv2_bottom" \
        or name == "conv4_top" or name == "conv4_bottom" \
        or name == "conv5_top" or name == "conv5_bottom" \
        or name == "dense6_top" or name == "dense6_bottom" \
        or name == "dense7_top" or name == "dense7_bottom":
            nn.init.constant_(module.bias, 1)

        # setting bias = 0 in the remaining layers
        elif name == "conv1_top" or name == "conv1_bottom" \
        or name == "conv3_top" or name == "conv3_bottom" \
        or name == "conv4_top" or name == "conv4_bottom" \
        or name == "dense_last":
            nn.init.constant_(module.bias, 0)

#### DEFINING OTHER ESSENTIAL FUNCTIONS

def subtract_mean_activity(
    tensor,
    mean_activity=mean_activity_all_training,
):
    """
    This function subtracts the mean activity of all 
    training images from the input tensor.

    :param tensor: Input image as a tensor
    :param mean_activity: The mean activity of all training images
    :return: Tensor with the mean image subtracted
    """
    if mean_activity.shape != (3, 256, 256):
        mean_activity = mean_activity.permute(2, 0, 1)
    
    mean_activity_subtracted_tensor = tensor - mean_activity
    return mean_activity_subtracted_tensor


def add_pca_multiples(
    tensor,
    random_alphas,
    eigenvectors=training_eigenvectors,
    eigenvalues=training_eigenvalues,
):
    """
    This function adds the following PCA multiples to the input tensor:
    [P1 P2 P3] [α1 λ1, α2 λ2, α3 λ3]

    Where Pi are the eigenvectors, αi are values from a random distribution,
    and λi are the eigenvalues. The eigenv*s are computed by performing PCA
    on RGB pixel values of all training images.
    
    :param tensor: Input image as a tensor
    :param random_alphas: A 1 x 3 array with values drawn from N(0, 0.1)
    :param eigenvectors: The eigenvectors from performing PCA on all RGB pixels
    :param eigenvalues: The eigenvalues from performing PCA on all RGB pixels
    :return: Tensor with PCA multiples added
    """
    multiple_to_add = eigenvectors @ (eigenvalues * random_alphas)
    multiple_to_add_reshaped = multiple_to_add.reshape(3, 1, 1)
    
    tensor_w_multples_added = tensor + multiple_to_add_reshaped
    return tensor_w_multples_added
    

def extract_augment_image_co(
    image_path,
    corners,
    orientation_bool,
    subtract_mean_activity=subtract_mean_activity,
    add_pca_multiples=add_pca_multiples,
):
    """
    This function subtracts the mean activity of all training images from the image,
    crops a 224 x 224 image portion based on the provided left-top coordinate, 
    adds PCA multiples to R, G, & B pixel values of the image, and optionally 
    horizontally flips the tensor. 

    :param image_path: Path to the image as a string
    :param corners: A tuple with the left-top coordinates of the cropped image
    :param orientation_bool: True or False value for flipping the image
    :param subtract_mean_activity: Function that subtracts the mean activity from the tensor
    :param add_pca_multiples: Function that adds [P1 P2 P3] [α1 λ1, α2 λ2, α3 λ3] to the tensor
    :return: A (224 x 224)-sized, randomly cropped, and potentially flipped tensor
    """
    image_as_tensor = F.pil_to_tensor(Image.open(image_path))

    # performing mean activity/image subtraction
    image_as_tensor_subracted = subtract_mean_activity(image_as_tensor)

    # retrieving the left-top corners and cropping the image 
    start_x, start_y = corners
    cropped_tensor = image_as_tensor_subracted[:, start_x:start_x + 224, start_y:start_y + 224]
    
    # adding PCA multiples to the images
    # defining the alpha here so that a new set of numbers are drawn for each image
    # as done in the paper by the authors
    new_random_alpha = torch.from_numpy(
        np.random.normal(loc=0, scale=0.1, size=3)
    ).to(torch.float32)
    
    pca_added_tensor = add_pca_multiples(cropped_tensor, new_random_alpha)

    # optionally flipping the tensor based on the boolean value
    if orientation_bool:
        pca_added_tensor = torch.flip(pca_added_tensor, dims=(2,))

    # converting to float32
    final_tensor = pca_added_tensor.to(torch.float32)

    return final_tensor


class AlexNetTrainingSet(Dataset):
    
    """
    This custom dataset class returns the labels and augmented images for
    training/validation. 
    
    The method augment_params generates unique random corners for each 
    image using a random seed generated from a hash value, which is generated
    from the image's name.
    
    The method update_epoch_number should be called after each training run 
    to increment the augmentation index; this ensures that a new randomly
    cropped image is returned during each training run.
    """
    
    def __init__(
        self, 
        image_directory=training_images_output_path, 
        image_names=labels_w_information["image_name"],
        label_from_image_name=label_from_name_dictionary,
        transform=extract_augment_image_co,
        total_augmentations=90,
        epoch_number=0,
    ):
        self.image_directory = image_directory
        self.image_names = image_names
        self.label_from_image_name = label_from_image_name
        self.transform = transform
        self.total_augmentations = total_augmentations
        self.epoch_number = epoch_number

    def __len__(self):
        # total number of training images
        return len(self.image_names)

    def __getitem__(self, index):
        # the augmentation index; every epoch of the run picks a unique crop/flip
        augment_index = self.epoch_number

        # assigning the image name, label, and path
        image_name = self.image_names[index]
        label = self.label_from_image_name.get(image_name)
        
        # get parameters for augmentation
        corner_coord, flip_bool = self.augment_params(image_name, augment_index)

        # getting the image path and a 224 x 224 random crop
        image_path = os.path.join(self.image_directory, image_name)
        augmented_image = self.transform(
            image_path,
            corner_coord,
            flip_bool,
        )
        
        return augmented_image, label

    def update_epoch_number(self):
        self.epoch_number += 1
        return self.epoch_number

    def augment_params(self, image_name, augment_index):
        # getting a unique hash value from image name -> unique random seed
        image_hash = hashlib.md5(f"{image_name}".encode('utf-8')).digest()
        rng_from_hash = np.random.default_rng(int.from_bytes(image_hash))
        
        # 256 - 224 = 32 is the max value to crop a 224 x 224 image
        all_coords = rng_from_hash.integers(0, 32 + 1, size=(self.total_augmentations, 2))

        # sampling a left-top corner coordinate
        corner_coord = all_coords[augment_index]
        
        # randomly choose whether or not to flip the image
        flip_bool = rng_from_hash.choice([True, False])

        return corner_coord, flip_bool


#### TRAINING PREPARATION BLOCK

# instantiating the dataset and dataloader for training data
alexnet_training_set = AlexNetTrainingSet()

train_loader = DataLoader(
    alexnet_training_set, 
    batch_size=128,
    shuffle=True, 
    num_workers=0
)

# instantiating the dataset and dataloader for validation data
alexnet_validation_set = AlexNetTrainingSet(
    image_directory=validation_images_output_path, 
    image_names=validation_images_processed_names,
    label_from_image_name=validation_label_from_name_dictionary,
    transform=extract_augment_image_co,
)

validation_loader = DataLoader(
    alexnet_validation_set, 
    batch_size=128,
    shuffle=True,
    num_workers=0
)

initial_learning_rate = 0.01
momentum=0.9
weight_decay=0.0005

# instantiating the model and applying the custom weights and biases
alexnet = AlexNet()
alexnet.apply(initialize_weights_biases)

# defining the loss and the optimizer
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    alexnet.parameters(), 
    lr=initial_learning_rate,
    momentum=momentum,
    weight_decay=weight_decay,
)

# learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode="max",
    factor=0.1,
    patience=10,
    threshold=0.01,
    cooldown=4,
    min_lr=initial_learning_rate / 1000
)

num_epochs = 90

# create string with date or date & time for the log file name
current_date_formatted = date.today().strftime("%Y%m%d")
#current_datetime_formatted = datetime.today().strftime("%Y%m%d_%H_%M")

log_file_name = f"./logs/training_log_full_{current_date_formatted}.csv"

def write_to_csv(epoch, train_loss, train_acc, val_loss, val_acc, lr):
    with open(log_file_name, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc, lr])

with open(log_file_name, "w", newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
        "epoch", 
        "training_loss", 
        "training_accuracy",
        "validation_loss", 
        "validation_accuracy",
        "learning_rate",
    ])


#### TRAINING BLOCK

for epoch in range(num_epochs):
    # training block
    
    # layers like dropout behave differently during training and validation
    # so, setting the model to training mode
    alexnet.train()
    
    running_loss = 0
    correct_train = 0
    total_train = 0
    
    for inputs, labels in train_loader:
        inputs = inputs.to(device2)
        labels = labels.to(device2)

        optimizer.zero_grad()
        outputs = alexnet(inputs)
        loss = criterion(outputs, labels)

        # backpropagation
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
    
    # computing training metrics
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct_train / total_train
    
    # validation block
    alexnet.eval()
    correct_val = 0
    total_val = 0
    val_loss_count = 0
    
    with torch.no_grad():
        for val_inputs, val_labels in validation_loader:
            val_inputs = val_inputs.to(device2)
            val_labels = val_labels.to(device2)
            
            val_outputs = alexnet(val_inputs)
            val_loss = criterion(val_outputs, val_labels)
            val_loss_count += val_loss.item()
            
            _, val_predicted = torch.max(val_outputs, 1)
            total_val += val_labels.size(0)
            correct_val += (val_predicted == val_labels).sum().item()
    
    # computing validation metrics
    val_epoch_loss = val_loss_count / len(validation_loader)
    val_epoch_acc = correct_val / total_val

    # executing the learning rate (lr) scheduler and saving the current lr value
    scheduler.step(val_epoch_acc)
    current_lr = scheduler.get_last_lr()[0]

    print(
        f"{str(epoch + 1).zfill(2)}/{num_epochs}, "
        f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.4f}, "
        f"Val Loss: {val_epoch_loss:.4f}, Val Accuracy: {val_epoch_acc:.4f}, "
        f"Learning Rate: {current_lr}"
    )

    # updating the epoch number/augmentation index in the custom DataSet class
    alexnet_training_set.update_epoch_number()
    alexnet_validation_set.update_epoch_number()
    
    write_to_csv(epoch + 1, epoch_loss, epoch_acc, val_epoch_loss, val_epoch_acc, current_lr)

print()
print(f"Trial training done! Loss and accuracy values were saved to {log_file_name}")

# saving the model and the optimizer states
torch.save(alexnet.state_dict(), f"./logs/full_runs/alexnet_wb_{current_date_formatted}.pt")
torch.save(optimizer.state_dict(), f"./logs/full_runs/alexnet_optimizer_{current_date_formatted}.pt")