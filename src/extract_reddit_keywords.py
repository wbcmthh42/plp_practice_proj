import pandas as pd
from datasets import Dataset, Features, Value
from transformers import pipeline
import hydra
from omegaconf import OmegaConf
import os
import torch
from tqdm import tqdm

def load_reddit_csv_to_datasets(path):
    df = pd.read_csv(path)
    dataset = Dataset.from_pandas(df)
    return dataset

def get_keywords(model_name, dataset):
    # Set up for 2 GPUs
    num_gpus = 2
    print(f"Using {num_gpus} GPUs")

    # Create a pipeline for each GPU
    pipes = [
        pipeline('summarization', model=model_name, max_length=512, truncation=True, device=f'cuda:{i}')
        for i in range(num_gpus)
    ]

    # Create a DataLoader to batch the data
    batch_size = 64  # Adjust this based on your GPU memory
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    updated_data = []

    for batch in tqdm(dataloader, desc="Extracting keywords"):
        texts = batch['sentence']
        
        # Split the batch between the two GPUs
        mid = len(texts) // 2
        texts_split = [texts[:mid], texts[mid:]]
        
        # Process on both GPUs simultaneously
        keywords_chunks = [
            pipes[i](chunk)
            for i, chunk in enumerate(texts_split)
        ]

        # Combine results
        keywords_batch = [
            item['summary_text']
            for chunk in keywords_chunks
            for item in chunk
        ]

        for i, keywords in enumerate(keywords_batch):
            entry = {key: batch[key][i] for key in batch}
            entry['keywords'] = keywords
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


# def get_keywords(model_name, dataset):
#     pipe = pipeline('summarization', model=model_name, max_length=512, truncation=True, device=0 if torch.backends.mps.is_available() else -1)

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

def save_to_csv(dataset, path):
    dataset.to_csv(path, index=False)

@hydra.main(config_path="../conf", config_name="config")
def main(cfg):
    model_name = cfg.eval.evaluation_model_name
    reddit_dataset = cfg.reddit_dataset
    os.makedirs(os.path.dirname(reddit_dataset), exist_ok=True)
    reddit_dataset = load_reddit_csv_to_datasets(cfg.reddit_dataset)
    dataset = get_keywords(model_name, reddit_dataset)
    save_to_csv(dataset, cfg.reddit_results_file)

if __name__ == '__main__':
    '''python -m src.extract_reddit_keywords'''
    main()
