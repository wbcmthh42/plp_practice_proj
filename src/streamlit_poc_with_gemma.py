"""
Module Description
================

TechPulse Streamlit POC with Gemma
------------------------

This module provides a Streamlit-based proof-of-concept (POC) application that retrieves Reddit tech posts from selected date ranges, extracts relevant tech keywords, and allows users to search for related arXiv research papers. The application aims to assist users in researching course materials for implementing new courses.

Key Features
------------

* Retrieves Reddit tech posts from user-selected date ranges
* Extracts relevant tech keywords from the posts
* Allows users to select and filter keywords
* Searches for related arXiv research papers based on the selected keywords
* Utilizes a Large Language Model (LLM) built with Gemma to generate insights from the keywords

Dependencies
------------

* Streamlit
* Pandas
* Plotly
* Gemma (language model)
* arXiv API

Usage
-----

To run the application, simply execute the script using `python src/streamlit_poc_with_gemma.py`. This will launch the Streamlit app, allowing users to interact with the application and generate insights.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import re
from typing import List, Tuple
import sys
import os
import requests
import json
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
import hydra
from omegaconf import DictConfig

# Add the directory containing RAG_V3.py to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Import necessary functions from RAG_V3
from rag import load_vector_store, hybrid_search, truncate_summary, setup_rag
# Add Hugging Face API token (make sure to keep this secure in production)
HUGGINGFACE_API_TOKEN = "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"

st.set_page_config(layout="wide")

def load_and_preprocess_data(file_path: str) -> pd.DataFrame:

    """
    Loads a CSV file containing sentiment analysis results and preprocesses the data.

    Parameters
    ----------
    file_path : str
        Path to the CSV file containing sentiment analysis results.

    Returns
    -------
    pd.DataFrame
        Preprocessed dataframe containing only rows with negative sentiment.
    """
    df = pd.read_csv(file_path)
    df['created_utc'] = pd.to_datetime(df['created_utc'])
    # return df[df['polarity'].isin(['POSITIVE', 'NEUTRAL'])]
    return df

def preprocess_keywords(keyword: str, stop_words: set) -> str:
    """
    Preprocesses a single keyword by removing punctuation, converting to lower case, splitting into words, removing stop words and words shorter than 1 character, and then joining the remaining words back together with a space in between each word.

    Parameters
    ----------
    keyword : str
        Keyword to preprocess.
    stop_words : set
        Set of stop words to remove from the keyword.

    Returns
    -------
    str
        Preprocessed keyword.
    """
    keyword = re.sub(r'[^a-z\s]', '', keyword.lower())
    words = keyword.split()
    return ' '.join(set(word for word in words if word not in stop_words and len(word) > 1)).upper()

def process_keywords(df: pd.DataFrame, stop_words: set) -> pd.DataFrame:
    """
    Processes a dataframe containing sentiment analysis results by splitting the keywords column into individual keywords, removing empty strings, '[deleted]', '[removed]', and stop words, and then joining the remaining words back together with a space in between each word.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing sentiment analysis results.
    stop_words : set
        Set of stop words to remove from the keywords.

    Returns
    -------
    pd.DataFrame
        Processed dataframe containing only rows with non-empty keywords.
    """
    df['keywords'] = df['keywords'].str.split(',')
    exploded_df = df.explode('keywords')
    exploded_df = exploded_df[~exploded_df['keywords'].isin(['', '[deleted]', '[removed]'])]
    exploded_df['keywords'] = exploded_df['keywords'].apply(lambda x: preprocess_keywords(x, stop_words))
    return exploded_df[exploded_df['keywords'] != '']

def get_top_keywords(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Gets the top n keywords from a dataframe by frequency and includes polarity counts.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing sentiment analysis results.
    n : int
        Number of top keywords to return.

    Returns
    -------
    pd.DataFrame
        Dataframe containing the top n keywords and their sentiment counts.
    """
    # Group by keywords and count occurrences of each polarity
    keyword_counts = df.groupby(['keywords', 'polarity']).size().unstack(fill_value=0)
    keyword_counts = keyword_counts.reset_index()
    keyword_counts['total'] = keyword_counts[['POSITIVE', 'NEUTRAL', 'NEGATIVE']].sum(axis=1)
    
    # Sort by total frequency and get top n
    return keyword_counts.sort_values(by='total', ascending=False).head(n)

def create_keyword_chart(top_keywords: pd.DataFrame, n: int) -> px.bar:
    """
    Creates a horizontal stacked bar chart of the top n keywords by frequency of sentiments.

    Parameters
    ----------
    top_keywords : pd.DataFrame
        Dataframe containing the top n keywords and their sentiment counts.
    n : int
        Number of top keywords to display.

    Returns
    -------
    px.bar
        Horizontal stacked bar chart of the top n keywords by sentiment frequency.
    """
    # Melt the DataFrame to long format for stacked bar chart
    melted_df = top_keywords.melt(id_vars='keywords', 
                                   value_vars=['POSITIVE', 'NEUTRAL', 'NEGATIVE'], 
                                   var_name='polarity', 
                                   value_name='count')

    # Create a horizontal stacked bar chart
    fig = px.bar(melted_df, 
                 y='keywords',  # Set y to keywords for horizontal bars
                 x='count',     # Set x to count for horizontal bars
                 color='polarity', 
                 title=f'Top {n} Keywords by Sentiment',
                 labels={'count': 'Count', 'keywords': 'Keywords'},
                 text='count',
                 color_discrete_sequence=['green', 'gray', 'red'])  # Specify colors for each sentiment

    fig.update_layout(barmode='stack', 
                      height=800,  # Adjust height as needed
                      width=1000,  # Adjust width as needed
                      yaxis={'categoryorder':'total ascending'},
                      clickmode='event+select')
    return fig

def display_keyword_details(df: pd.DataFrame, keyword: str):
    """
    Displays the top 10 research papers related to a given keyword.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the keyword data.
    keyword : str
        The keyword to search for.

    Returns
    -------
    None
    """
    st.write("Related Research Papers:")
    results = hybrid_search(keyword, top_n=10)

    # Sort results by date, most recent first
    sorted_results = sorted(results, key=lambda x: datetime.strptime(x['Updated'], '%Y-%m-%d'), reverse=True)

    for result in sorted_results:
        st.write(f"**Title:** {result['Title']}")
        st.write(f"**Category:** {result['Category']}")
        st.write(f"**Updated:** {result['Updated']}")
        st.write(f"**Summary:** {truncate_summary(result['Summary'])}")
        st.write(f"**Link:** {result['Link']}")
        st.markdown("---")

@st.cache_resource
def load_model_and_tokenizer():
    """
    Loads the Gemma model and tokenizer, and sets up the device (either MPS, CUDA, or CPU).

    Returns
    -------
    tokenizer : transformers.AutoTokenizer
        The tokenizer for the Gemma model.
    model : transformers.AutoModelForCausalLM
        The Gemma model.
    device : torch.device
        The device to use for the model (either MPS, CUDA, or CPU).
    """
    model_name = "google/gemma-2b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Check if MPS is available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device.type in ["cuda", "mps"] else torch.float32
    )
    model = model.to(device)
    return tokenizer, model, device

def get_llm_summary(keywords: List[str]) -> str:
    """
    Uses the Gemma LLM to generate a concise summary of the emerging tech trends
    represented by the given keywords, and proposes 5 potential topics that would
    be relevant and engaging for students.

    Parameters
    ----------
    keywords : List[str]
        The keywords to analyze.

    Returns
    -------
    str
        A summary of the emerging tech trends represented by the given keywords,
        and 5 potential topics that would be relevant and engaging for students.
    """
    tokenizer, model, device = load_model_and_tokenizer()

    prompt = f"""Analyze the following top keywords from recent Reddit tech discussions and provide a concise summary of the emerging tech trends they might represent:

Keywords: {', '.join(keywords)}

Based on these trends, propose 5 potential topics that would be relevant and engaging for students. Keep your response to lesser than 50 words in total.

Topics:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=250, temperature=0.7, top_p=0.95, do_sample=True)
    
    # Decode the entire output
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the generated summary (everything after the prompt)
    summary = full_output[len(prompt):].strip()

    # Ensure the summary starts with "Possible Topics:"
    if not summary.startswith("Possible Topics:"):
        summary = "Possible Topics: \n" + summary
    
    return summary
    
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # Set up RAG system
    """
    Main function to set up the RAG system, define a list of common stop words,
    and display a Streamlit app to analyze the top keywords from recent Reddit tech discussions.

    Parameters:
        cfg (omegaconf.DictConfig): Configuration for the RAG model.

    Returns:
        None
    """
    setup_rag(cfg)
    # Define a list of common stop words
    default_stop_words = set([
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
        'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were',
        'will', 'with', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
        'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself',
        'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
        'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs',
        'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll",
        'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
        'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
        'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
        'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
        'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
        'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
        'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
        'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
        'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll',
        'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't",
        'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't",
        'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',
        "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't",
        'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
    ])
    
    st.title('Reddit Keyword Frequency Analysis')

    # Load the vector store
    load_vector_store()

    file_path = '/Users/tayjohnny/Documents/My_MTECH/PLP/plp_practice_proj/reddit_keywords_results/reddit_keywords_91%_vader.csv'
    df = load_and_preprocess_data(file_path)

    min_date, max_date = df['created_utc'].min().date(), df['created_utc'].max().date()

    col1, col2 = st.columns([1, 1])

    with col2:
        date_range = st.date_input('Select a date range from reddit:', 
                                   [min_date, max_date], 
                                   min_value=min_date, 
                                   max_value=max_date)
        
        if len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date = end_date = date_range[0]
        
        st.text(f"Selected date range: {start_date} to {end_date}")
        st.text("")

        # Custom stop words input
        custom_stop_words = st.text_input("Add custom words to be filtered out (Note: comma separate each word):")
        if custom_stop_words:
            custom_stop_words = set(word.strip().lower() for word in custom_stop_words.split(','))
            stop_words = default_stop_words.union(custom_stop_words)
        else:
            stop_words = default_stop_words

    # Process keywords after stop_words are defined
    processed_df = process_keywords(df, stop_words)

    # Filter the DataFrame based on the selected date range
    filtered_df = processed_df[
        (processed_df['created_utc'].dt.date >= start_date) & 
        (processed_df['created_utc'].dt.date <= end_date)
    ]

    with col1:
        n_keywords = st.slider("Select number of top keywords to display", min_value=5, max_value=50, value=20, step=5)
        top_n_keywords = get_top_keywords(filtered_df, n_keywords)
        fig = create_keyword_chart(top_n_keywords, n_keywords)
        st.plotly_chart(fig, use_container_width=True, key="keyword_chart")

    with col2:
        # Add LLM summary here
        st.subheader("Tech Trends Insight Based on Top Keywords")
        if st.button("Generate Insight"):
            with st.spinner("Generating insight..."):
                insight = get_llm_summary(top_n_keywords['keywords'].tolist()[:10])  # Limit to top 10 keywords
                st.write(insight)

        selected_keywords = st.multiselect("Selected Keywords", options=top_n_keywords['keywords'].tolist())

        if selected_keywords:
            for keyword in selected_keywords:
                display_keyword_details(filtered_df, keyword)  # Use filtered_df here
        else:
            st.write("Select keywords from the multiselect box above to see details.")

if __name__ == "__main__":
    main()