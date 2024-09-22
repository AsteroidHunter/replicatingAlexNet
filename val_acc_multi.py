import os
import torch
from PIL import Image
import torch.nn as nn
from tqdm import tqdm
import multiprocessing
from functools import partial
import torch.nn.functional as nnF
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader, Dataset

device1 = torch.device("cpu")
device2 = device1


class AlexNet(nn.Module):

    """
    A 2-GPU AlexNet model. 
    
    The top layers of the model reside on the first GPU, and the bottom layers
    reside on the second GPU. The input image is fed to both the layers. 
    
    There are a few redundant function calls & 
    repeated output concatenations in this model; I have left them in place as
    they are good for visually tracking how the layers are set up and avoiding
    GPU-layer mismatch errors.
    """
    
    def __init__(self, number_of_classes=1000):
        super().__init__()

        self.number_of_classes = number_of_classes
        
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

    #def mask_input(self, x, mask_value=None):
    #    if mask_value is None:
    #        mask_value = self.mask_value
    #        
    #    x_device = x.device
    #    *_, height, width = x.shape
    #    
    #    upper_mask = torch.ones((height, width))
    #    upper_mask[int(height / 2):, :] = mask_value
    #    
    #    lower_mask = torch.zeros((height, width))
    #    lower_mask[:int(height / 2), :] = mask_value
    #
    #    upper_mask = upper_mask.to(x_device)
    #    lower_mask = lower_mask.to(x_device)
    #    
    #    return x * upper_mask, x * lower_mask
    
    def forward(self, x):
        
        # top_image, bottom_image = self.mask_input(x)
        top_image = x.to(device1)
        bottom_image = x.to(device2)


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


# loading the saved mean activity from section 1.1 of augment_ingest.ipynb
mean_activity_all_training = torch.load(
    "./images/mean_activity_of_all_training_images.pt",
    weights_only=False
)

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

validation_label_from_name_dictionary = {
    image_name: label for image_name, label 
    in zip(validation_images_processed_names, validation_true_labels)
}



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
        image_names=None,
        label_from_image_name=None,
        transform=None,
    ):
        self.image_directory = image_directory
        self.image_names = image_names
        self.label_from_image_name = label_from_image_name
        self.transform = transform

    def __len__(self):
        # total number of training images
        return len(self.image_names)

    def __getitem__(self, index):
        # assigning the image name, label, and path
        image_name = self.image_names[index]
        label = self.label_from_image_name.get(image_name)

        # getting the image path and center crop
        image_path = os.path.join(self.image_directory, image_name)

        image_as_tensor = F.pil_to_tensor(Image.open(image_path))
        
        return image_as_tensor, label

def extract_augment_image_ten(
    image_as_tensor, 
    subtract_mean_activity
):
    corners_flips = [
        (0, 0, True), (0, 0, False),
        (0, 32, True), (0, 32, False),
        (32, 32, True), (32, 32, False),
        (32, 0, True), (32, 0, False),
        (16, 16, True), (16, 16, False),
    ]
    
    evaluated_out = []
    for corners_flips_tuple in corners_flips:
        start_x, start_y, flip = corners_flips_tuple
        
        image_as_tensor_subtracted = subtract_mean_activity(image_as_tensor)
        cropped_tensor = image_as_tensor_subtracted[:, start_x:start_x + 224, start_y:start_y + 224]
    
        if flip:
            cropped_tensor = torch.flip(cropped_tensor, dims=(2,))
    
        final_tensor = cropped_tensor.to(torch.float32)
        evaluated_out.append(final_tensor)
    
    return torch.stack(evaluated_out)

def process_batch(batch, model, subtract_mean_activity):
    inputs, labels = batch
    augmented_inputs = torch.vstack(
        [extract_augment_image_ten(i, subtract_mean_activity) for i in inputs]
    )
    
    with torch.no_grad():
        outputs = model(augmented_inputs)
    outputs = outputs.view(inputs.size(0), 10, -1).mean(dim=1)
    
    # getting top-5 predictions
    _, top5_pred = torch.topk(outputs, 5, dim=1)
    
    return top5_pred, labels

def parallel_validation(model, validation_loader, num_cores):
    model.eval()
    
    correct_top1 = 0
    correct_top5 = 0
    total_val = 0
    
    pool = multiprocessing.Pool(processes=num_cores)
    
    process_batch_partial = partial(
        process_batch, model=model, subtract_mean_activity=subtract_mean_activity
    )
    
    for batch_results in tqdm(
        pool.imap(
            process_batch_partial, validation_loader
        ), 
        total=len(validation_loader)
    ):
        top5_pred, labels = batch_results
        total_val += labels.size(0)
        
        # top-1 accuracy
        correct_top1 += (top5_pred[:, 0] == labels).sum().item()
        
        # top-5 accuracy
        correct_top5 += sum(labels.unsqueeze(1) == top5_pred).sum().item()
    
    pool.close()
    pool.join()
    
    top1_accuracy = correct_top1 / total_val
    top5_accuracy = correct_top5 / total_val
    
    return top1_accuracy, top5_accuracy

# multi-processing for computing the top-1 and top-5 accuracies
if __name__ == "__main__":
    num_cores = 10
    
    alexnet = AlexNet()
    alexnet.load_state_dict(
        torch.load(
            "./logs/full_runs/alexnet_wb_20240912.pt",
            map_location="cpu"
        )
    )
    alexnet.eval()
    
    # instantiating the dataset and dataloader for validation data
    alexnet_validation_set = AlexNetTrainingSet(
        image_directory=validation_images_output_path, 
        image_names=validation_images_processed_names,
        label_from_image_name=validation_label_from_name_dictionary,
    )
    
    batch_size_is = 32
    
    validation_loader = DataLoader(
        alexnet_validation_set, 
        batch_size=batch_size_is,
        shuffle=False,
        num_workers=0,
    )
    
    # parallel validation
    top1_accuracy, top5_accuracy = parallel_validation(alexnet, validation_loader, num_cores)
    print(f"Top-1 Accuracy: {top1_accuracy:.4f}")
    print(f"Top-5 Accuracy: {top5_accuracy:.4f}")