import torch
from torch.utils.data import DataLoader
import sys
import os
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.embedding_mlp import EmbeddingMLP
from src.audio_dataloader import AudioDataset

# Binary classification loss function
loss_fn = torch.nn.BCEWithLogitsLoss()

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the best model
    best_model_path = os.path.join(os.getcwd(), "models","checkpoints", "audio_mlp_best_model.pth")
    mlp = EmbeddingMLP(input_dim=768, hidden_dim=128, dropout_prob=0.3, num_layers=13, num_classes=1)
    mlp.load_state_dict(torch.load(best_model_path, map_location=device))  # Load model on the correct device
    mlp.to(device)  # Move model to GPU if available
    mlp.eval()  # Set model to evaluation mode

    # Load test dataset
    base_path = path = os.getcwd()
    config_path = os.path.join(base_path, "configs", "config.yaml")
    with open(config_path, 'r', encoding="utf-8") as f:
        config = yaml.safe_load(f)

    db_path = os.path.join(base_path, "data", config["project"]["name"], "db.csv")
    embeddings_path = os.path.join(base_path, "data", config["project"]["name"], "audio_embeddings")

    test_dataset = AudioDataset(db_path, split='test', embeddings_folder=embeddings_path)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    test_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():  # No gradients needed during evaluation
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)  # Move to GPU

            targets = targets.view(-1, 1)  # Reshape targets to match output shape
            outputs = mlp(inputs)  # Forward pass

            # Compute loss
            loss = loss_fn(outputs, targets)
            test_loss += loss.item()

            # Convert logits to probabilities & round to get class labels
            predictions = torch.sigmoid(outputs).round()

            # Compute accuracy
            total_correct += (predictions == targets).sum().item()
            total_samples += targets.size(0)

    # Compute final test loss & accuracy
    avg_test_loss = test_loss / len(test_loader)
    test_accuracy = total_correct / total_samples

    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")


    # Access the learned weights from the WeightedAverage layer
    learned_weights = mlp.weighted_average.weights  # This is a trainable parameter
    normalized_weights = torch.softmax(learned_weights, dim=0)  # Normalize using softmax

    print("Learned Weighted Average for Each Layer:")
    for i, weight in enumerate(normalized_weights.cpu().detach().numpy()):
        print(f"Layer {i}: {weight:.4f}")
