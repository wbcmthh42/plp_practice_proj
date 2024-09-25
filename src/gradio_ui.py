import gradio as gr
import pandas as pd
import re
from collections import Counter
import plotly.express as px

def clean_keyword(keyword):
    # Remove HTML tags
    keyword = re.sub(r'<.*?>', '', keyword)
    # Remove numbers and punctuation
    keyword = re.sub(r'[^a-zA-Z\s]', '', keyword)
    # Strip extra spaces
    keyword = keyword.strip()
    return keyword

def display_unique_keywords():
    # Hardcoded path to the CSV file
    file_path = 'reddit_keywords_results/reddit_keywords_full_distilbert.csv'
    # Read the CSV file
    df = pd.read_csv(file_path)
    # Extract the 'keywords' column
    unique_keywords_list = []
    keyword_counter = Counter()
    for keywords in df['keywords']:
        # Split the keywords into individual words, clean them, remove duplicates, and join them back into a string
        unique_keywords = set()
        for word in keywords.split(','):
            cleaned_word = clean_keyword(word)
            if cleaned_word and len(cleaned_word.split()) <= 3:
                unique_keywords.add(cleaned_word)
        # Join unique keywords into a string
        unique_keywords_str = ", ".join(unique_keywords)
        # Append to the list if not empty
        if unique_keywords_str:
            unique_keywords_list.append(unique_keywords_str)
            keyword_counter.update(unique_keywords)
    # Join all rows into a single string for display
    unique_keywords_display = "\n".join(unique_keywords_list)
    # Format the frequency distribution for display
    frequency_distribution = "\n".join([f"{word}: {count}" for word, count in keyword_counter.items()])
    
    # Sort the keyword_counter by frequency in descending order
    sorted_keywords = sorted(keyword_counter.items(), key=lambda item: item[1], reverse=True)
    sorted_words, sorted_counts = zip(*sorted_keywords)
    
    # Create the interactive plot using Plotly
    fig = px.bar(x=sorted_words, y=sorted_counts, labels={'x': 'Keywords', 'y': 'Frequency'}, title='Keyword Frequency Distribution')
    fig.update_layout(xaxis_tickangle=-45)
    
    return unique_keywords_display, frequency_distribution, fig

# Create the Gradio interface
iface = gr.Interface(
    fn=display_unique_keywords,
    inputs=None,  # No input needed since the file path is hardcoded
    outputs=["text", "text", gr.Plot()],  # Two text outputs and one Plot output
    title="Reddit Unique Keywords Display",
    description="Displays the unique keywords column from a hardcoded CSV file, their frequency distribution, and an interactive plot of the word distribution."
)

# Launch the interface
iface.launch(share=True)