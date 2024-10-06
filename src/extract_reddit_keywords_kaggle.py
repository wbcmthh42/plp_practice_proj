
import pandas as pd
from datasets import Dataset, Features, Value
from transformers import pipeline
# import hydra
# from omegaconf import OmegaConf
import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")
from torch.utils.data import Subset

from tqdm import tqdm
import pandas as pd
import os
from torch.utils.data import DataLoader, Subset
from transformers import pipeline
import torch


def load_reddit_csv_to_datasets(path):
    df = pd.read_csv(path)
    dataset = Dataset.from_pandas(df)
    return dataset

def get_keywords(model_name, dataset):
    
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print("Using GPU")
    else:
        device = torch.device('cpu')  
        print("Using CPU")

    pipe = pipeline('summarization', model=model_name, truncation=True, device='cuda:0')

    # Use only the first 350,000 rows of the dataset for testing
    subset_indices = list(range(min(400000, len(dataset))))
    subset_dataset = Subset(dataset, subset_indices)
    
    # Check if the subset dataset is empty
    if len(subset_dataset) == 0:
        print("No data available in the dataset.")
        return pd.DataFrame()  # Return an empty DataFrame
    
    batch_size = 256  # Adjust based on your GPU memory
    dataloader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=False)

    updated_data = []
    total_batches = len(dataloader)

    for batch_index, batch in enumerate(tqdm(dataloader, desc="Extracting keywords")):
        texts = batch['sentence']
        
        # Process each text in the batch individually
        keywords_batch = []
        for text in texts:
            if text is None:  # Check for None values
                keywords_batch.append("")  # Append an empty string or handle as needed
                continue
            
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
            temp_file_name = f'/kaggle/working/reddit_keywords_hybrid_temp_{(batch_index // (total_batches // 10) + 1) * 10}.csv'
            temp_df.to_csv(temp_file_name, index=False)

    # Final save after processing all batches
    df = pd.DataFrame(updated_data)
    return df
            
def save_to_csv(dataset, path):
    dataset.to_csv(path, index=False)

def main():
    model_name = 'wbcmthh42/bart_tech_keywords_model2'
    reddit_dataset_path = '/kaggle/input/vader-distilbert-hybrid/sentiment_analysis_results_hybrid_160k.csv'
    
    os.makedirs(os.path.dirname(reddit_dataset_path), exist_ok=True)
    reddit_dataset = load_reddit_csv_to_datasets(reddit_dataset_path)
    
    dataset = get_keywords(model_name, reddit_dataset)
    
    save_to_csv(dataset, '/kaggle/working/reddit_keywords_vader.csv')

if __name__ == '__main__':
    '''python -m src.extract_reddit_keywords'''
    main()
