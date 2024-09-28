import pandas as pd
import torch
from tqdm import tqdm
import re
import nltk
from nltk.tokenize import sent_tokenize
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import logging
nltk.download('punkt')
nltk.download('stopwords')
import os
import hydra
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Check if GPU is available
if torch.cuda.is_available():
    logging.info("GPU is available")
    device = 0
else:
    logging.info("GPU is not available")
    device = -1

# Define helper functions
def truncate_to_max_length(text, tokenizer, max_length):
    encoded_input = tokenizer(text, truncation=True, max_length=max_length, return_tensors='pt')
    return tokenizer.decode(encoded_input['input_ids'][0], skip_special_tokens=True)


def process_sentiment_analysis(df, sentiment_analyzer, tokenizer, candidate_labels, output_file, batch_size=16, max_length=512):
    start_time = time.time()
    logging.info(f"Sentiment analysis started at {start_time}")
    
    results = []  # Initialize list to store results
    sentences_batch = []  # List to collect sentences in batches

    # Process each row in the DataFrame
    for index, row in tqdm(df.iterrows(), total=len(df), ncols=100):
        review = row['body']
        subreddit = row['subreddit']
        created_utc = row['created_utc']

        # Sentence tokenization
        sentences = sent_tokenize(review)

        # Truncate sentences and collect them in a batch
        for sentence in sentences:
            truncated_sentence = truncate_to_max_length(sentence, tokenizer, max_length=max_length)
            sentences_batch.append(truncated_sentence)

        # Process in batches
        if len(sentences_batch) >= batch_size:
            sentiment_results = sentiment_analyzer(sentences_batch, candidate_labels)

            # Append results for the current batch
            for sentiment_result in sentiment_results:
                results.append({
                    "subreddit": subreddit,
                    "created_utc": created_utc,
                    "sentence": sentiment_result['sequence'],
                    "sentiment": sentiment_result['labels'],
                    "sentiment_score": sentiment_result['scores'],
                })

            sentences_batch = []  # Clear the batch after processing

    # Process any remaining sentences in the last batch
    if sentences_batch:
        sentiment_results = sentiment_analyzer(sentences_batch, candidate_labels)
        for sentiment_result in sentiment_results:
            results.append({
                "subreddit": subreddit,
                "created_utc": created_utc,
                "sentence": sentiment_result['sequence'],
                "sentiment": sentiment_result['labels'],
                "sentiment_score": sentiment_result['scores'],
            })

    # Convert results to DataFrame and post-process
    sentiment_df = pd.DataFrame(results)
    sentiment_df['sentiment'] = sentiment_df['sentiment'].apply(lambda x: x[0])  # Extract the top sentiment
    sentiment_df['sentiment_score'] = sentiment_df['sentiment_score'].apply(lambda x: x[0])  # Extract the top score
    
    end_time = time.time()
    logging.info(f"Sentiment analysis ended at {end_time}")
    duration = end_time - start_time
    logging.info(f"Total sentiment analysis duration: {duration}")
    
    # Save the results to a CSV file
    sentiment_df.to_csv(output_file, index=False)
    logging.info(f"Sentiment analysis results saved to {output_file}")

def preprocess_text(text):
    # Set up stopwords list for word cloud
    stopwords_list = stopwords.words('english')  
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stopwords_list])
    return text

def generate_wordcloud(df, sentiment, output_dir):
    # Filter the DataFrame for the specified sentiment
    filtered_df = df[df['sentiment'] == sentiment]

    # Create bigrams
    vectorizer = CountVectorizer(ngram_range=(2, 2), stop_words= stopwords.words('english'))
    bigram_counts = vectorizer.fit_transform(filtered_df['sentence'])

    # Get bigram frequencies
    feature_names = vectorizer.get_feature_names_out()
    bigram_freq = dict(zip(feature_names, bigram_counts.sum(axis=0).tolist()[0]))

    # Create WordCloud from the bigram frequencies
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Blues').generate_from_frequencies(bigram_freq)

    # Plot and save the WordCloud
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'Word Cloud for {sentiment} Sentiment')
    plt.axis('off')

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the figure instead of showing it
    output_path = f"{output_dir}/wordcloud_{sentiment}.png"
    plt.savefig(output_path, format='png', bbox_inches='tight')
    plt.close()  # Close the plot to free up memory

    # Log the saved location
    logging.info(f"Word cloud saved to {output_path}")

@hydra.main(config_path="../conf", config_name="config")
def main(cfg):
    # Define variables for input/output files and model
    input_file = cfg.sentiment.input_file  # Input dataset file
    output_file = cfg.sentiment.output_file  # Output file for results
    model_name = cfg.sentiment.model_name  # Model to be used for zero-shot classification
    output_dir = cfg.sentiment.output_dir
    # Load the dataset
    df = pd.read_csv(input_file)
    
    # Set up sentiment analyzer using zero-shot classification
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    sentiment_analyzer = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer, device=device)
    candidate_labels = ["POSITIVE", "NEUTRAL", "NEGATIVE"]

    # Run sentiment analysis
    process_sentiment_analysis(df, sentiment_analyzer, tokenizer, candidate_labels, output_file=output_file, batch_size=16)

    # Load the sentiment analysis results
    sentiment_df = pd.read_csv(output_file)
    sentiment_df['sentence'] = sentiment_df['sentence'].astype(str).apply(lambda x: preprocess_text(x))

    # Generate word clouds for each sentiment
    for sentiment in candidate_labels:
        generate_wordcloud(sentiment_df, sentiment,output_dir)


if __name__ == '__main__':
    '''python -m src.sentiment_analysis'''
    main()