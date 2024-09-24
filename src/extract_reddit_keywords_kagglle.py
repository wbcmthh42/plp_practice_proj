def load_reddit_csv_to_datasets(path):
    df = pd.read_csv(path)
    dataset = Dataset.from_pandas(df)
    return dataset

def get_keywords(model_name, dataset):
    num_gpus = 2
    print(f"Using {num_gpus} GPUs")

    # Create a pipeline for each GPU without specifying max_length
    pipes = [
        pipeline('summarization', model=model_name, truncation=True, device=f'cuda:{i}')
        for i in range(num_gpus)
    ]

    batch_size = 64  # Adjust this based on your GPU memory
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    updated_data = []

    for batch in tqdm(dataloader, desc="Extracting keywords"):
        texts = batch['sentence']
        
        # Split the batch between the two GPUs
        mid = len(texts) // 2
        texts_split = [texts[:mid], texts[mid:]]
        
        # Process on both GPUs simultaneously
        keywords_chunks = []
        for i, chunk in enumerate(texts_split):
            # Calculate max_length and min_length for each text in the chunk
            max_lengths = [max(10, min(512, len(text.split()))) for text in chunk]
            min_lengths = [max(1, min(5, len(text.split()) // 2)) for text in chunk]
            
            # Process each text in the chunk individually
            chunk_keywords = []
            for text, max_length, min_length in zip(chunk, max_lengths, min_lengths):
                if len(text.split()) <= 10:
                    # For very short texts, just use the text as is
                    chunk_keywords.append(text)
                else:
                    # For longer texts, use the summarization pipeline
                    summary = pipes[i](text, max_length=max_length, min_length=min_length)[0]['summary_text']
                    chunk_keywords.append(summary)
            
            keywords_chunks.append(chunk_keywords)

        # Combine results
        keywords_batch = keywords_chunks[0] + keywords_chunks[1]

        for i, keywords in enumerate(keywords_batch):
            entry = {key: batch[key][i] for key in batch}
            entry['keywords'] = keywords
            updated_data.append(entry)
            
def save_to_csv(dataset, path):
    dataset.to_csv(path, index=False)

def main():
    model_name = 'wbcmthh42/bart_tech_keywords'
    reddit_dataset = '/kaggle/input/sent-distilbert/sentiment_analysis_results_distillbert.csv'
    os.makedirs(os.path.dirname(reddit_dataset), exist_ok=True)
    reddit_dataset = load_reddit_csv_to_datasets(reddit_dataset)
    dataset = get_keywords(model_name, reddit_dataset)
    save_to_csv(dataset, '/kaggle/working/')

if __name__ == '__main__':
    '''python -m src.extract_reddit_keywords'''
    main()