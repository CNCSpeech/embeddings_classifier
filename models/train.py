import sys
import os
from tqdm.auto import tqdm
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.embedding_mlp import EmbeddingMLP
from src.audio_dataloader import AudioDataset
from src.utils.config import get_config

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = get_config()
    train_dataset = AudioDataset(config.data.train_data_path, split='train', embeddings_folder = config.data.audio_embeddings_path)
    train_loader = DataLoader(train_dataset, 
                              batch_size=config.training.batch_size, 
                              shuffle=config.data.shuffle, 
                              num_workers=config.data.num_workers)  # DataLoader with batching and shuffling

    # Create the validation DataLoader}
    val_dataset = AudioDataset(config.data.val_data_path, split='val', embeddings_folder = config.data.audio_embeddings_path)
    val_loader = DataLoader(val_dataset, 
                            batch_size=config.training.batch_size, 
                            shuffle=config.data.shuffle, 
                            num_workers=config.data.num_workers)

    # Initialize Model
    mlp = EmbeddingMLP(config.training.input_dim, 
                       config.training.hidden_dim, 
                       config.training.dropout_prob, 
                       config.training.num_layers, 
                       config.model.num_classes).to(device)
    print(summary(mlp, input_size=(config.training.num_layers, config.training.input_dim)))

    # Training Configuration
    num_epochs = config.training.num_epochs
    loss_fn = nn.BCEWithLogitsLoss() if config.model.num_classes == 1 else nn.CrossEntropyLoss()
    optimizer = optim.Adam(mlp.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # Reduce LR every 10 epochs

    # Track Loss
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    worst_val_loss = float('-inf')  # Track the worst loss
    patience = 20
    no_improvement_count = 0

    # Training Loop
    for epoch in range(num_epochs):
        mlp.train()
        train_loss = 0.0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                targets = targets.view(-1, 1)  # Reshape targets to match output shape

                optimizer.zero_grad()
                outputs = mlp(inputs)  # Raw logits

                # Compute loss correctly
                loss = loss_fn(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(mlp.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()
                pbar.update(1)

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation
        mlp.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                targets = targets.view(-1, 1)  # Reshape targets to match output shape
                outputs = mlp(inputs)
                loss = loss_fn(outputs, targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            #save into /home/aleph/embbedings_classifier/models/checkpoints
            torch.save(mlp.state_dict(), os.path.join(base_path, "models", "checkpoints", "audio_mlp_best_model.pth"))
            no_improvement_count = 0  # Reset early stopping counter
        else:
            no_improvement_count += 1  # Increment counter if no improvement

        # Save checkpoint every epoch 10 epochs
        if epoch % 10 == 0:
            torch.save(mlp.state_dict(),  os.path.join(base_path, "models", "checkpoints", f"audio_mlp_checkpoint_{epoch}.pth"))

        # Early Stopping
        if no_improvement_count >= patience:
            print(f"Stopping early after {epoch + 1} epochs due to no improvement in validation loss.")
            break

        # Step Learning Rate Scheduler
        scheduler.step()

    # Save Train & Validation Loss History
    losses = pd.DataFrame({'train_loss': train_losses, 'val_loss': val_losses})
    losses.to_csv(os.path.join(base_path, "models", "checkpoints", 'losses.csv'), index=False)
