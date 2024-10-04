import pandas as pd
from datasets import Dataset, Features, Value
from transformers import pipeline
import hydra
from omegaconf import OmegaConf
import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import logging

def load_reddit_csv_to_datasets(path):
    df = pd.read_csv(path)
    dataset = Dataset.from_pandas(df)
    return dataset

def get_keywords(model_name, dataset):
    # Check if CUDA (GPU) is available, otherwise CPU
    if torch.cuda.is_available():
        device = torch.device('cuda:0')  
    else:
        device = torch.device('cpu')  
        print("Using CPU")

    pipe = pipeline('summarization', model=model_name, max_length=512, truncation=True, device=device)

    # Create a DataLoader to batch the data
    batch_size = 64 
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    updated_data = []

    for batch in tqdm(dataloader, desc="Extracting keywords"):
        texts = batch['sentence']

        # Process the batch using the single pipeline
        keywords_batch = pipe(texts)

        # Extract and store the keywords
        for i, keywords in enumerate(keywords_batch):
            entry = {key: batch[key][i] for key in batch}
            entry['keywords'] = keywords['summary_text']
            updated_data.append(entry)

    # Create a new feature schema including 'keywords'
    features = Features({
        'subreddit': Value(dtype='string'),
        'created_utc': Value(dtype='string'),
        'sentence': Value(dtype='string'),
        'sentiment': Value(dtype='string'),
        'sentiment_score': Value(dtype='float32'),
        'keywords': Value(dtype='string'),
    })

    updated_dataset = Dataset.from_dict({key: [d[key] for d in updated_data] for key in updated_data[0]})

    return updated_dataset.with_format(type='torch', columns=['subreddit', 'created_utc', 'sentence', 'sentiment', 'sentiment_score', 'keywords'])

def save_to_csv(dataset, path):
    dataset.to_csv(path, index=False)

@hydra.main(config_path="../conf", config_name="config", version_base="1.1")
def main(cfg):
    model_name = cfg.extract.extraction_model_name
    reddit_dataset = cfg.extract.reddit_dataset
    reddit_dataset = load_reddit_csv_to_datasets(cfg.extract.reddit_dataset)
    dataset = get_keywords(model_name, reddit_dataset)
    save_to_csv(dataset, cfg.extract.reddit_results_file)
    logging.info("Output file save to " + cfg.extract.reddit_results_file)

if __name__ == '__main__':
    '''python -m src.extract_reddit_keywords'''
    main()
