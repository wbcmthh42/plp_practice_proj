import pandas as pd
import re
from collections import Counter
import matplotlib.pyplot as plt
import json
import os
import hydra
from omegaconf import OmegaConf
import logging

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def clean_keywords(keywords):
    # Split the keywords
    keyword_list = keywords.split(',')
    
    # Clean each keyword
    cleaned_keywords = []
    for keyword in keyword_list:
        # Remove numbers, punctuation, html tags, etc.
        keyword = re.sub(r'[^\w\s]|[\d]|<.*?>', '', keyword)
        
        # Convert to lowercase
        keyword = keyword.lower().strip()
        
        # Check if keyword has 3 or fewer words
        if len(keyword.split()) <= 3 and keyword != '':
            cleaned_keywords.append(keyword)
    
    # Remove duplicates while preserving order
    cleaned_keywords = list(dict.fromkeys(cleaned_keywords))
    
    return ','.join(cleaned_keywords)

def get_trending_keywords(df, output_dir, top_n=20):

    # Assuming the DataFrame has a 'keywords' column
    df['keywords_cleaned'] = df['keywords'].apply(clean_keywords)
    
    # Flatten the list of keywords
    all_keywords = [keyword.strip() for keywords in df['keywords_cleaned'] for keyword in keywords.split(',')]
    
    # Count word frequencies
    word_freq = Counter(all_keywords)

    # Remove empty string from the counter
    word_freq.pop('', None)

    # Sort the word frequencies in descending order
    sorted_word_freq = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True))

    # Select top 20 words for the plot
    top_n_words = dict(list(sorted_word_freq.items())[:top_n])

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a bar plot
    plt.figure(figsize=(12, 6))
    plt.bar(top_n_words.keys(), top_n_words.values())
    plt.title('Top 20 Keywords Frequency')
    plt.xlabel('Keywords')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Save the plot as PNG
    plot_path = os.path.join(output_dir, 'keyword_frequency_plot.png')
    plt.savefig(plot_path)
    logging.info(f"Plot saved as '{plot_path}'")

    # Close the plot to free up memory
    plt.close()

    # Save the top 20 words and their frequencies as JSON
    json_path = os.path.join(output_dir, f'top_{top_n}_keywords.json')
    with open(json_path, 'w') as json_file:
        json.dump(top_n_words, json_file, indent=2)
    logging.info(f"Top {top_n} keywords saved as '{json_path}'")

@hydra.main(config_path="../conf", config_name="config")
def main(cfg):
    df = load_data(cfg.reddit_results_file)
    get_trending_keywords(df, cfg.get_trending_keywords.output_dir, top_n=cfg.get_trending_keywords.top_n)

if __name__ == "__main__":
    '''python -m src.get_trending_keywords'''
    main()