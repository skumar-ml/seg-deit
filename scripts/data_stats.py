from torchvision import datasets, transforms
from timm.data import create_transform
import torch
import numpy as np
from fast_slic.avx2 import SlicAvx2
import os
import shutil
import math
from tqdm import tqdm
import cv2

def process_segment(x, n_segments=196):
    B, C, H, W = x.shape
    x = x.permute(0, 2, 3, 1) # Change channels to be last dimension
    x = x.cpu().numpy()

    if not x.flags['C_CONTIGUOUS']:
        x = x.copy(order='C')

    assert x.flags['C_CONTIGUOUS']

     # Iterate over each image in batch and get segmentation mask
    save_mask = torch.zeros((B, H, W), device=DEVICE)
    for i, img in enumerate(x):
        cp_img = np.squeeze(img)

        # slic = Slic(num_components=n_segments, min_size_factor=0)
        # slic = SlicAvx2(num_components=n_segments, min_size_factor=0)
        # segmentation_mask = slic.iterate(cp_img)
        # print(i, len(np.unique(segmentation_mask)))
        # assert len(np.unique(segmentation_mask)) == n_segments, f"Got {len(np.unique(segmentation_mask))} segments from SLIC, but expected {n_segments}"

        segmentation_mask = np.zeros((224, 224))

        for w in range(0, 224, 16):
            for j in range(0, 224, 16):
                segmentation_mask[w:w+16, j:j+16] = int(w/16 + (j/16)*14)

        save_mask[i, :, :] = torch.from_numpy(segmentation_mask).to(DEVICE)
        
    return save_mask

def process_ft(x, save_mask, n_segments=196, n_points=64, grayscale=True):

    # Format
    x = x.permute(0, 2, 3, 1).squeeze()
    B = x.shape[0]
    
    # Allocate
    seg_out = torch.zeros((B, n_segments, int(n_points * (n_points/2 + 1) * 2)))
    pos_out = torch.zeros((B, n_segments, 5))

    magnitude_sum = 0
    magnitude_sum_2 = 0

    phase_sum = 0
    phase_sum_2 = 0

    # TODO: Verify implementation
    for i in range(n_segments):
        # print(i)
        # Get mask for segment i for all images in batch
        binary_mask = (save_mask == i)
        # print(binary_mask.shape)

        for j, image_mask in enumerate(binary_mask):
            image_mask = np.uint8(image_mask.cpu().numpy())
            coords = cv2.findNonZero(image_mask)
            bbox_x, bbox_y, bbox_w, bbox_h = cv2.boundingRect(coords)
            cropped_image = x[j, bbox_y:bbox_y+bbox_h, bbox_x:bbox_x+bbox_w]

            # Take FT and separate magnitude and phase info
            # fourier_transform = torch.fft.fft2(segmented_imgs, s=(n_points, n_points))
            fourier_transform = torch.fft.rfft2(cropped_image, s=(n_points, n_points))
            # print(fourier_transform.shape)
            magnitude = torch.abs(fourier_transform) 
            phase = torch.angle(fourier_transform) 

            assert torch.sum(torch.isnan(magnitude)).item() == 0, "NaN element in the magnitude before normalization"

            # Normalize scales & standardize data according to dataset statistics
            # magnitude = magnitude / (1 if torch.max(magnitude)==0 else torch.max(magnitude))  # [0 1] -- divide by 1 if max is already 0.
            phase = phase / torch.tensor(math.pi) # [-1 1]
        
            magnitude_sum += torch.sum(magnitude)
            magnitude_sum_2 += torch.sum(torch.square(magnitude))

            phase_sum = torch.sum(phase)
            phase_sum_2 = torch.sum(torch.square(phase))

        return magnitude_sum, magnitude_sum_2, phase_sum, phase_sum_2

# User variables
data_path = "/home/sk138/data/"
data_folder_name = "cifar-100-python/"

input_size = 224
num_workers = 10
n_segments=196
pin_mem = True
n_points=64
batch_size=2000
DEVICE='cuda:0'

# Transforms
train_transform = create_transform(input_size, is_training=True, no_aug=True)
test_transform = create_transform(input_size, is_training=False, no_aug=True)
grayscale_transform = transforms.Grayscale()

print(train_transform)
print(test_transform)

# Create data loaders
train_dataset = datasets.CIFAR100(data_path, train=True, transform=train_transform, download=True)
test_dataset = datasets.CIFAR100(data_path, train=False, transform=test_transform, download=True)

class_names = train_dataset.classes

train_sampler = torch.utils.data.SequentialSampler(train_dataset)
test_sampler = torch.utils.data.SequentialSampler(test_dataset)

data_loader_train = torch.utils.data.DataLoader(
    train_dataset,
    sampler=train_sampler,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=pin_mem,
    drop_last=False,
)

data_loader_test = torch.utils.data.DataLoader(
    test_dataset,
    sampler=test_sampler,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=pin_mem,
    drop_last=False,
)

# Get stats
data_type = "train"
use = data_loader_train if data_type=="train" else data_loader_test

magnitude_sum_tot = 0
magnitude_sum_2_tot = 0
phase_sum_tot = 0
phase_sum_2_tot = 0

# Iterate over dataloader
for i, (inputs, labels) in tqdm(enumerate(use)):
    inputs = inputs.to(DEVICE)
    inputs = inputs*255
    inputs = inputs.to(torch.uint8)

    # Segment image, take FT, and find pos embed. 
    seg_mask = process_segment(inputs, n_segments=n_segments)
    input_gray = grayscale_transform(inputs)
    temp1, temp2, temp3, temp4 = process_ft(input_gray, seg_mask, n_segments=n_segments, n_points=n_points)

    magnitude_sum_tot += temp1
    magnitude_sum_2_tot += temp2
    phase_sum_tot += temp3
    phase_sum_2_tot += temp4

# Save
save_vals = torch.Tensor([magnitude_sum_tot, magnitude_sum_2_tot, phase_sum_tot, phase_sum_2_tot])
print(save_vals)
torch.save(save_vals, f'{data_type}_sums.pt')

count = (10000 * 192 * 64 * 33)
mag_mean = magnitude_sum_tot / count
mag_var  = (magnitude_sum_2_tot / count) - (mag_mean ** 2)
mag_std  = torch.sqrt(mag_var)

phase_mean = phase_sum_tot / count
phase_var  = (phase_sum_2_tot / count) - (phase_mean ** 2)
phase_std  = torch.sqrt(phase_var)

stat_vals = torch.Tensor([mag_mean, mag_std, phase_mean, phase_std])
print(stat_vals)
torch.save(stat_vals, f'{data_type}_stats.pt')
    