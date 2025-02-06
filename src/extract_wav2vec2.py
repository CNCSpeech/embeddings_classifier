import os
import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model, Wav2Vec2Config
import librosa
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.config import get_config


def extract_wav2vec_embeddings(input_dir, output_dir, model_name='facebook/wav2vec2-base'):
    """
    Extracts embeddings from all .wav files in all subfolders of input_dir using Wav2Vec2.
    Uses GPU if available, otherwise falls back to CPU.
    
    Parameters:
        input_dir (str): Path to the root folder containing .wav files.
        output_dir (str): Path to save the extracted embeddings.
        model_name (str): Pretrained Wav2Vec2 model name.
    """

    os.makedirs(output_dir, exist_ok=True)

    # Detect GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Wav2Vec2 model and feature extractor
    model_config = Wav2Vec2Config.from_pretrained(model_name, output_hidden_states=True)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = Wav2Vec2Model(model_config).to(device)

    file_count = sum(len([file for file in files if file.endswith('.wav')]) for _, _, files in os.walk(input_dir))  # Get the number of .wav files
    with tqdm(total=file_count) as pbar:
        # Iterate through all .wav files in subdirectories
        for root, _, files in os.walk(input_dir):
            for file in files:
                try:
                    if not file.endswith('.wav'):
                        continue
                    
                    audio_file = os.path.join(root, file)
                    input_audio, sample_rate = librosa.load(audio_file, sr=16000)
                    
                    inputs = feature_extractor(input_audio, return_tensors="pt", sampling_rate=sample_rate).to(device)

                    with torch.no_grad():
                        outputs = model(**inputs)

                    # Extract embeddings from all hidden states, then compute mean
                    all_layer_embeddings = torch.cat(outputs.hidden_states, dim=0).mean(dim=1)
                    
                    # Convert to NumPy and save
                    embeddings = all_layer_embeddings.squeeze().detach().cpu().numpy()
                    file_name = os.path.splitext(file)[0]
                    os.makedirs(output_dir, exist_ok=True)
                    np.save(os.path.join(output_dir, f"{file_name}.npy"), embeddings)

                    # Free memory
                    del embeddings
                    pbar.update(1)
                except torch.OutOfMemoryError:
                    #save to txt file
                    with open("out_of_memory.txt", "a", encoding="utf-8") as f:
                        f.write(f"Out of memory. Skipping {file}\n")
                    continue
    print(f"Embeddings successfully extracted and saved in '{output_dir}'.")


if __name__ == '__main__':
    # Perform embeddings extraction
    config = get_config()
    extract_wav2vec_embeddings(config.data.audio_path, config.data.audio_embeddings_path)
