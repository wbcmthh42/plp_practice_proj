
import pandas as pd
from datasets import Dataset, Features, Value
import hydra
from omegaconf import OmegaConf
import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from transformers import pipeline
import warnings
warnings.filterwarnings("ignore")

def load_reddit_csv_to_datasets(path):
    df = pd.read_csv(path)
    dataset = Dataset.from_pandas(df)
    return dataset

def get_keywords(cfg, model_name, dataset):

    if torch.backends.mps.is_available():
        device = torch.device('mps')  # Use MPS if available
        print("Using MPS")
    elif torch.cuda.is_available():
        device = torch.device('cuda:0')  # Use CUDA if available
        print("Using CUDA")
    else:
        device = torch.device('cpu')  # Fall back to CPU
        print("Using CPU")

    pipe = pipeline('summarization', model=model_name, truncation=True, device=device)

    # we are capping it to 400000 rows of the dataset for this POC. otherwise too much compute time needed
    subset_indices = list(range(min(cfg.reddit_inference.inference_row_limit, len(dataset))))
    subset_dataset = Subset(dataset, subset_indices)
    
    # Check if the subset dataset is empty
    if len(subset_dataset) == 0:
        print("No data available in the dataset.")
        return pd.DataFrame()  # Return an empty DataFrame
    
    batch_size = cfg.reddit_inference.batch_size  # Adjust based on your GPU memory
    dataloader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=False)

    updated_data = []
    total_batches = len(dataloader)

    if total_batches == 0:
        print("No batches available for processing.")
        return pd.DataFrame()  # Return an empty DataFrame

    for batch_index, batch in enumerate(tqdm(dataloader, desc="Extracting keywords")):
        texts = batch['sentence']
        
        # Process each text in the batch individually
        keywords_batch = []
        for text in texts:
            max_length = max(10, min(512, len(text.split())))
            min_length = max(1, min(5, len(text.split()) // 2))
            
            if len(text.split()) <= 10:
                keywords_batch.append(text)
            else:
                summary = pipe(text, max_length=max_length, min_length=min_length)[0]['summary_text']
                keywords_batch.append(summary)

        for i, keywords in enumerate(keywords_batch):
            entry = {key: batch[key][i] for key in batch}
            entry['keywords'] = keywords
            updated_data.append(entry)

        # Save progress every 10% of total batches, avoiding division by zero
        if total_batches > 0 and batch_index % (total_batches // 10) == 0:
            temp_df = pd.DataFrame(updated_data)
            tmp_dir = os.path.join(os.getcwd(), 'tmp')
            os.makedirs(tmp_dir, exist_ok=True)
            temp_file_name = f'{tmp_dir}/reddit_keywords_hybrid_temp_{(batch_index // (total_batches // 10) + 1) * 10}.csv'
            temp_df.to_csv(temp_file_name, index=False)

    # Final save after processing all batches
    df = pd.DataFrame(updated_data)
    return df
            
def save_to_csv(dataset, path):
    dataset.to_csv(path, index=False)

@hydra.main(config_path="../conf", config_name="config", version_base="1.1")
def main(cfg):

    model_name = cfg.saved_model_in_hf
    reddit_dataset_path = cfg.reddit_dataset
    
    os.makedirs(os.path.dirname(reddit_dataset_path), exist_ok=True)
    reddit_dataset = load_reddit_csv_to_datasets(reddit_dataset_path)
    
    dataset = get_keywords(cfg, model_name, reddit_dataset)
    
    save_to_csv(dataset, cfg.reddit_results_file_for_ui)

if __name__ == '__main__':
    '''python -m src.extract_reddit_keywords'''
    main()