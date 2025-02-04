"""Custom dataset to read embeddings"""
import os
import numpy as np
import pandas as pd

# torch
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import yaml


# Custom dataset to read embeddings and group (condition) from a CSV
class AudioDataset(Dataset):
    def __init__(self, csv_file, split, embeddings_folder):
        # Load the CSV file
        self.data = pd.read_csv(csv_file)

        # Ensure there's a split column and filter data
        if 'split' in self.data.columns:
            self.data = self.data[self.data['split'] == split]
        else:
            raise ValueError("CSV file must contain a 'split' column to separate train/val/test data.")

        self.split = split
        self.split = split # Store the split (train, val, test)
        self.folder = embeddings_folder


    def __len__(self):
        return len(self.data)  # Total number of samples in the dataset


    def __getitem__(self, idx):
        # Get the embeddings and group for the given index
        file_name = os.path.basename(self.data.iloc[idx]['id'].lower())
        file_name = file_name + "__día_típico.npy"

        embeddings_path = os.path.join(self.folder, file_name) #TODO mejorar esta lectura

        # Check if the embeddings file exists
        if not os.path.isfile(embeddings_path):
            raise FileNotFoundError(f"Embeddings file not found at {embeddings_path}")

        embeddings = np.load(embeddings_path)

        group = self.data.iloc[idx]['group']

        # Convert the numpy array of embeddings to a tensor
        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
                
        # group score should be a tensor with a single element
        group_tensor = torch.tensor(group, dtype=torch.float32)
        
        return embeddings_tensor, group_tensor


if __name__ == "__main__":
    # Test the training DataLoader
    base_path = path = os.getcwd()
    config_path = os.path.join(base_path, "configs", "config.yaml")
    with open(config_path, 'r', encoding="utf-8") as f:
        config = yaml.safe_load(f)

    db_path = os.path.join(base_path, "data", config["project"]["name"], "db.csv")
    embeddings_path = os.path.join(base_path, "data", config["project"]["name"], "audio_embeddings")

    batch_size = config["training"]["batch_size"]
    shuffle = config["data"]["shuffle"]
    num_workers = config["data"]["num_workers"]

    train_dataset = AudioDataset(db_path, split='train', embeddings_folder = embeddings_path)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)  # DataLoader with batching and shuffling

    # Create the validation DataLoader}
    val_dataset = AudioDataset(db_path, split='val', embeddings_folder = embeddings_path)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    # Create the testing DataLoader
    test_dataset = AudioDataset(db_path, split='test',embeddings_folder = embeddings_path)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    # Print the number of samples in each split
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")
    print(f"Number of testing samples: {len(test_dataset)}")

    # print example of data
    for i, (embeddings_example, group_example) in enumerate(train_loader):
        print(f"Batch {i} embeddings shape: {embeddings_example.shape}, group shape: {group_example.shape}")
        break
