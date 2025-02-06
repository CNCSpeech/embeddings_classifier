from dataclasses import dataclass
from typing import Optional
import yaml
from pathlib import Path
import os

@dataclass
class ProjectConfig:
    name: str
    description: str
    author: str

@dataclass
class ModelConfig:
    name: str
    num_classes: int

@dataclass
class TrainingConfig:
    batch_size: int
    num_epochs: int
    learning_rate: float
    weight_decay: float
    optimizer: str
    scheduler: str
    step_size: int
    input_dim: int
    hidden_dim: int
    dropout_prob: float
    num_layers: int # Number of layers in the transformer model

@dataclass
class DataConfig:
    train_data_path: str
    val_data_path: str
    test_data_path: str
    num_workers: int
    shuffle: bool
    audio_embeddings_path: str
    text_embeddings_path: str
    audio_path: str

@dataclass
class LoggingConfig:
    log_dir: str
    log_interval: int

@dataclass
class CheckpointConfig:
    save_dir: str
    save_interval: int
    resume_from: str

@dataclass
class DeviceConfig:
    gpu: bool
    gpu_id: int

@dataclass
class Config:
    project: ProjectConfig
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    logging: LoggingConfig
    checkpoint: CheckpointConfig
    device: DeviceConfig
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        return cls(
            project=ProjectConfig(**config_dict['project']),
            model=ModelConfig(**config_dict['model']),
            training=TrainingConfig(**config_dict['training']),
            data=DataConfig(**config_dict['data']),
            logging=LoggingConfig(**config_dict['logging']),
            checkpoint=CheckpointConfig(**config_dict['checkpoint']),
            device=DeviceConfig(**config_dict['device'])
        )

# Singleton instance
_config: Optional[Config] = None

def get_config(yaml_path: str = None) -> Config:
    global _config
    if _config is None:
        if yaml_path is None:
            yaml_path = os.path.join(Path(__file__).parents[2], 'configs', 'config.yaml')
        _config = Config.from_yaml(yaml_path)
    return _config
