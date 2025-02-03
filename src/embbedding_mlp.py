import torch
import torch.nn as nn


class WeightedAverage(torch.nn.Module):
    # Weighted average layer, learns weights (importance) for each layer and combines them
    def __init__(self, num_layers=12):
        super().__init__()
        self.weights = torch.nn.Parameter(data=torch.ones((num_layers,)))
       
    def forward(self, x):
        w = torch.nn.functional.softmax(self.weights, dim=0)
        x_weighted = x*w[None,:,None]
        return torch.sum(x_weighted, dim=1)

class EmbeddingMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_prob, num_layers):
        super(EmbeddingMLP, self).__init__()

        # add a weighted average layer
        self.weighted_average = WeightedAverage(num_layers)
        
        # First dense layer with 128 neurons, ReLU activation, and dropout
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # Linear layer for dense transformation
            nn.ReLU(),  # ReLU activation
            nn.Dropout(dropout_prob),  # Dropout with 0.2
        )

        # Second dense layer, same design
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
        )

        # Final dense layer for group score prediction
        self.output_layer = nn.Linear(hidden_dim, 1)  # Linear layer to predict group score

    def forward(self, x):
        # Apply the weighted average to combine 12 layers into 1
        x = self.weighted_average(x)  # Apply WeightedAverage

        # Pass the input through the first dense layer
        x = self.layer1(x)

        # Pass through the second dense layer
        x = self.layer2(x)

        # Pass through the final dense layer to get the group
        x = self.output_layer(x)  # Output layer

        return x


#possible model parameters
# input_dim = 768  # Single 768-dimensional input
# hidden_dim = 128  # Hidden dimension for dense layers
# dropout_prob = 0.6  # Dropout probability
# num_layers = 13  # Number of layers in the Wav2Vec2 model