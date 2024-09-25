
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

def load_reddit_csv_to_datasets(path):
    df = pd.read_csv(path)
    dataset = Dataset.from_pandas(df)
    return dataset

# def get_keywords(model_name, dataset):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     pipe = pipeline('summarization', model=model_name, max_length=512, truncation=True, device=device)
    
#     # Create a list to hold the updated data
#     updated_data = []

#     # Use tqdm to create a progress bar
#     for i in tqdm(range(len(dataset)), desc="Extracting keywords"):
#         entry = dataset[i]
#         text = entry['sentence']
#         keywords = pipe(text)[0]['summary_text']
#         entry['keywords'] = keywords
#         updated_data.append(entry)

#     # Create a new feature schema including 'keywords'
#     features = Features({
#         'subreddit': Value(dtype='string'),
#         'created_utc': Value(dtype='string'),
#         'sentence': Value(dtype='string'),
#         'sentiment': Value(dtype='string'),
#         'sentiment_score': Value(dtype='float32'),
#         'keywords': Value(dtype='string'),
#     })

#     updated_dataset = Dataset.from_dict({key: [d[key] for d in updated_data] for key in updated_data[0]})

#     return updated_dataset.with_format(type='torch', columns=['subreddit', 'created_utc', 'sentence', 'sentiment', 'sentiment_score', 'keywords'])

def get_keywords(model_name, dataset):
    print("Using 1 GPU")

    # Create a pipeline for a single GPU without specifying max_length
    pipe = pipeline('summarization', model=model_name, truncation=True, device='cuda:0')

    # Use only the first 1000 rows of the dataset for testing
#     subset_indices = list(range(1000))
#     subset_dataset = Subset(dataset, subset_indices)
    
    batch_size = 256  # Adjust this based on your GPU memory
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    updated_data = []

    for batch in tqdm(dataloader, desc="Extracting keywords"):
        texts = batch['sentence']
        
        # Process each text in the batch individually
        keywords_batch = []
        for text in texts:
            # Calculate max_length and min_length for each text
            max_length = max(10, min(512, len(text.split())))
            min_length = max(1, min(5, len(text.split()) // 2))
            
            if len(text.split()) <= 10:
                # For very short texts, just use the text as is
                keywords_batch.append(text)
            else:
                # For longer texts, use the summarization pipeline
                summary = pipe(text, max_length=max_length, min_length=min_length)[0]['summary_text']
                keywords_batch.append(summary)

        for i, keywords in enumerate(keywords_batch):
            entry = {key: batch[key][i] for key in batch}
            entry['keywords'] = keywords
            updated_data.append(entry)
    
    # Convert updated_data to a DataFrame
    df = pd.DataFrame(updated_data)
    return df
            
def save_to_csv(dataset, path):
    dataset.to_csv(path, index=False)

def main():
    model_name = 'wbcmthh42/t5_tech_keywords_model'
#     model_name = 'wbcmthh42/t5_tech_keywords_model'
    reddit_dataset = '/kaggle/input/sent-distilbert/sentiment_analysis_results_distillbert.csv'
    os.makedirs(os.path.dirname(reddit_dataset), exist_ok=True)
    reddit_dataset = load_reddit_csv_to_datasets(reddit_dataset)
    dataset = get_keywords(model_name, reddit_dataset)
    save_to_csv(dataset, '/kaggle/working/reddit_keywords_t5.csv')

if __name__ == '__main__':
    '''python -m src.extract_reddit_keywords'''
    main()
