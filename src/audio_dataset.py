"""Custom dataset to read embeddings"""
import os
import numpy as np
import pandas as pd

# torch
import torch
from torch.utils.data import Dataset

# Custom dataset to read embeddings and group (condition) from a CSV
class AudioDataset(Dataset):
    def __init__(self, csv_file, split, embeddings_folder):
        # Load the CSV file
        self.data = pd.read_csv(csv_file)
        self.split = split # Store the split (train, val, test)
        self.folder = embeddings_folder

        
    def __len__(self):
        return len(self.data)  # Total number of samples in the dataset
    
    def __getitem__(self, idx):
        # Get the embeddings and group for the given index
        file_name = os.path.basename(self.data.iloc[idx]['stimuli'])
        file_folder = os.path.basename(os.path.dirname(self.data.iloc[idx]['stimuli']))

        embeddings_path = os.path.join(self.folder, f"{self.split}", f"{file_folder}", f"{file_name.split('.')[0]}", ".npy") #TODO mejorar esta lectura

        # Check if the embeddings file exists
        if not os.path.isfile(embeddings_path):
            raise FileNotFoundError(f"Embeddings file not found at {embeddings_path}")

        embeddings = np.load(embeddings_path)

        group = self.data.iloc[idx]['group']

        # Convert the numpy array of embeddings to a tensor
        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
                
        # group score should be a single value? or label? #TODO
        group_tensor = torch.tensor([group], dtype=torch.float32)
        
        return embeddings_tensor, group_tensor


# # Create the training DataLoader
# train_csv_path = "/home/aleph/tesis/classifier/train.csv"
# train_dataset = AudioDataset(train_csv_path, split='train')
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)  # DataLoader with batching and shuffling

# # Create the validation DataLoader
# val_csv_path = "/home/aleph/tesis/classifier/val.csv"
# val_dataset = AudioDataset(val_csv_path, split='val')
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

# # Create the testing DataLoader
# test_csv_path = "/home/aleph/tesis/classifier/test.csv"
# test_dataset = AudioDataset(test_csv_path, split='test')
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2) 