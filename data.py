import torch
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode, functional as TF #added this
from torch.utils.data import Dataset, random_split
import os
import numpy as np


# Define data transform(You can freely modify this part to suit your needs)
class SegmentationDataset(Dataset):  #added this class
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, mask = self.dataset[idx]

        mask = TF.resize(mask, (224, 224), interpolation=InterpolationMode.NEAREST)

        # Convert to a safe integer type, subtract 1 to match the submission format,
        # and use .clip(0, 2) to guarantee all values are exactly 0, 1, or 2.
        mask_np = np.array(mask, dtype=np.int16)
        mask_np = (mask_np - 1).clip(0, 2)

        mask = torch.as_tensor(mask_np, dtype=torch.long)

        return image, mask



transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# Download and load the entire dataset
data_dir = '/root/CAS3120/oxford_pet_data'

# Load train split, changed train_dataset to train_dataset_raw
train_dataset_raw = datasets.OxfordIIITPet(
    root=data_dir,
    split='trainval',  # This includes both train and validation from original split
    download=True,
    transform=transform,
    target_types='segmentation' # might change from category to segmentation
)

# Load test split, changed test_dataset to test_dataset_raw
test_dataset_raw = datasets.OxfordIIITPet(
    root=data_dir,
    split='test',
    download=True,
    transform=transform,
    target_types='segmentation' # might change from category to segmentation
)

# added the segmentationdataset below
train_dataset = SegmentationDataset(train_dataset_raw)
test_dataset = SegmentationDataset(test_dataset_raw)

# Combine train and test datasets
class FullDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.len1 = len(dataset1)
        self.len2 = len(dataset2)
        
    def __len__(self):
        return self.len1 + self.len2
    
    def __getitem__(self, idx):
        if idx < self.len1:
            return self.dataset1[idx]
        else:
            return self.dataset2[idx - self.len1]

# Combine datasets
full_dataset = FullDataset(train_dataset, test_dataset)

# Calculate split sizes (90% train, 10% val)
total_size = len(full_dataset)
train_size = int(0.9 * total_size)
val_size = total_size - train_size

print(f"Total samples: {total_size}")
print(f"Train samples: {train_size}")
print(f"Validation samples: {val_size}")

# Random split with fixed seed for reproducibility
torch.manual_seed(42)
train_set, val_set = random_split(full_dataset, [train_size, val_size])

print(f"\nFinal split:")
print(f"Train set size: {len(train_set)}")
print(f"Validation set size: {len(val_set)}")

# Create data loaders
batch_size = 32

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4 # was 4 before
)

val_loader = torch.utils.data.DataLoader(
    val_set,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4 # was 4 before
)

print(f"\nData loaders created successfully!")
print(f"Number of training batches: {len(train_loader)}")
print(f"Number of validation batches: {len(val_loader)}")