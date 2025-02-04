# embbedings_classifier
Audio and text embbedings classifier scripts, tests, modules and experiments

## Project Structure

embedding_classification/
+ data/  # Raw and processed datasets
  + raw/  # Unprocessed data files
  + processed/  # Preprocessed data files (e.g., embeddings, features)
  + external/  # Any external datasets or pre-trained embeddings
+ models/  # Model definitions and saved models
  + checkpoints/  # Saved model checkpoints
  + embedding_mlp.py  # Model architecture
  + train.py  # Training script
  + evaluate.py  # Evaluation script
+ notebooks/  # Jupyter notebooks for exploration and prototyping
  + data_analysis.ipynb
  + model_experiments.ipynb
+ src/  # Source code for the project
  + x_data_loader.py  # Data preprocessing and loading functions
  + extract_wav2vec2.py  # Embedding extraction methods
  + classifier.py  # Classification logic
  + utils.py  # Helper functions
+ configs/  # Configuration files
  + config.yaml  # Hyperparameters, paths, etc.
+ scripts/  # Utility scripts for running tasks
  + run_training.sh  # Bash script for training
+ tests/  # Unit tests for the project
  + test_data.py
  + test_model.py
+ results/  # Outputs, logs, and visualizations
  + logs/  # Training logs
  + plots/  # Evaluation plots
+ requirements.txt  # Python dependencies
+ README.md  # Project documentation
+ .gitignore  # Files to ignore in version control
