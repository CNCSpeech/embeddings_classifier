import torch
from torch.utils.data import DataLoader
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.embbedding_mlp import EmbeddingMLP
from src.audio_dataset import AudioDataset
loss_fn = torch.nn.CrossEntropyLoss()  # Loss function for classification

if __name__ == "__main__":
    # Load the best model
    best_model_path = "/home/aleph/tesis/classifier/best_model.pth"
    dense_mlp = EmbeddingMLP(input_dim=768, hidden_dim=128, dropout_prob=0.2, num_layers=13)  # Ensure correct model initialization
    dense_mlp.load_state_dict(torch.load(best_model_path))  # Load the saved model
    dense_mlp.eval()  # Set model to evaluation mode

    # Evaluate the model on the test set
    test_csv_path = "test_set_embeddings.csv" #TODO change this path
    test_dataset = AudioDataset(test_csv_path)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)  # DataLoader for testing
    test_loss = 0.0
    with torch.no_grad():  # No gradients needed during evaluation
        for inputs, targets in test_loader:
            outputs = dense_mlp(inputs)  # Forward pass
            loss = loss_fn(outputs, targets)  # Calculate loss
            test_loss += loss.item()  # Accumulate the loss

    # Average test loss over all batches
    avg_test_loss = test_loss / len(test_loader)
    print("Test Loss:", avg_test_loss)  # Evaluate the model's performance on the test set
