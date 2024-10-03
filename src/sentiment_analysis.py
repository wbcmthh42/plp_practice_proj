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
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

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

def get_sentiment(text):
    vader_obj = SentimentIntensityAnalyzer()
    if isinstance(text, str):
        return vader_obj.polarity_scores(text)
    else:
        return {'neg': 0, 'neu': 0, 'pos': 0, 'compound': 0}  # Return neutral for non-strings
def get_sentiment_label(compound_score):
    """Assign sentiment label based on VADER sentiment scores."""
    if compound_score > 0.05:
        return 'POSITIVE'
    elif compound_score < -0.05:
        return 'NEGATIVE'
    else:
        return 'NEUTRAL'

def hybrid_sentiment(row):
    if row['d_prediction'] == row['v_prediction']:
        return row['d_prediction']  # If both are the same, return the common sentiment
    else:
        return 'NEUTRAL'  # If they are different, return 'NEUTRAL'

def process_sentiment_analysis(df, sentiment_analyzer,tokenizer, output_file):
    start_time = time.time()
    logging.info(f"Sentiment analysis started.")
    
    results = []  # Initialize list to store results
 
    # Process each row in the DataFrame
    for index, row in tqdm(df.iterrows(), total=len(df), ncols=100):
        review = row['body']
        subreddit = row['subreddit']
        created_utc = row['created_utc']
        for sentence in sent_tokenize(review):
            truncated_sentence = truncate_to_max_length(sentence, tokenizer, max_length=512)
            # Get sentiment for the sentence
            sentiment_result = sentiment_analyzer(truncated_sentence)[0]

            # Extract sentiment label and score
            sentiment_label = sentiment_result['label']
            sentiment_score = sentiment_result['score']

            # Append results
            results.append({
                "subreddit": subreddit,
                "created_utc": created_utc,
                "sentence": sentence,
                "d_prediction": sentiment_label,
                "sentiment_score": sentiment_score
            })

    # Convert results to DataFrame and post-process
    sentiment_df = pd.DataFrame(results)
    
    ### run Vader
    sentiment_df[['neg', 'neu', 'pos', 'compound']] = sentiment_df['sentence'].apply(get_sentiment).apply(pd.Series)
    sentiment_df['v_prediction'] = sentiment_df['compound'].apply(get_sentiment_label).apply(pd.Series)
    sentiment_df['sentiment'] = sentiment_df.apply(hybrid_sentiment, axis=1)
    
    end_time = time.time()
    logging.info(f"Sentiment analysis ended.")
    duration = end_time - start_time
    logging.info(f"Total sentiment analysis duration: {duration}")
    
    # Save the results to a CSV file
    sentiment_df = sentiment_df[['subreddit', 'created_utc', 'sentence', 'd_prediction', 'v_prediction', 'sentiment','sentiment_score']]
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

@hydra.main(config_path="../conf", config_name="config", version_base="1.1")
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
    sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device = -1)

    # Run sentiment analysis
    process_sentiment_analysis(df, sentiment_analyzer, tokenizer, output_file=output_file)

    # Load the sentiment analysis results
    sentiment_df = pd.read_csv(output_file)
    sentiment_df['sentence'] = sentiment_df['sentence'].astype(str).apply(lambda x: preprocess_text(x))

    # Generate word clouds for each sentiment
    for sentiment in sentiment_df['sentiment'].unique():
        generate_wordcloud(sentiment_df, sentiment,output_dir)


if __name__ == '__main__':
    '''python -m src.sentiment_analysis'''
    main()