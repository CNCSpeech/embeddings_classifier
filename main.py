from src.utils.config import get_config

# Get config instance
config = get_config()

# Access config values
batch_size = config.training.batch_size
learning_rate = config.training.learning_rate
train_data_path = config.data.train_data_path

print(f"Batch Size: {batch_size}")
print(f"Learning Rate: {learning_rate}")
print(f"Train Data Path: {train_data_path}")

