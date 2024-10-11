import streamlit as st
import pandas as pd
import plotly.express as px
import re
from typing import List, Tuple
import sys
import os
from datetime import datetime

# Add the directory containing RAG_V3.py to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'arxiv')))

# Import necessary functions from RAG_V3
from RAG_V3 import load_vector_store, hybrid_search, truncate_summary

st.set_page_config(layout="wide")

def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df['created_utc'] = pd.to_datetime(df['created_utc'])
    return df[~df['polarity'].isin(['POSITIVE', 'NEUTRAL'])]

def preprocess_keywords(keyword: str, stop_words: set) -> str:
    keyword = re.sub(r'[^a-z\s]', '', keyword.lower())
    words = keyword.split()
    return ' '.join(set(word for word in words if word not in stop_words and len(word) > 1)).upper()

def process_keywords(df: pd.DataFrame, stop_words: set) -> pd.DataFrame:
    df['keywords'] = df['keywords'].str.split(',')
    exploded_df = df.explode('keywords')
    exploded_df = exploded_df[~exploded_df['keywords'].isin(['', '[deleted]', '[removed]'])]
    exploded_df['keywords'] = exploded_df['keywords'].apply(lambda x: preprocess_keywords(x, stop_words))
    return exploded_df[exploded_df['keywords'] != '']

def get_top_keywords(df: pd.DataFrame, n: int) -> pd.DataFrame:
    keyword_counts = df.groupby('keywords').size().reset_index(name='frequency')
    return keyword_counts.sort_values(by='frequency', ascending=False).head(n)

def create_keyword_chart(top_keywords: pd.DataFrame, n: int) -> px.bar:
    fig = px.bar(top_keywords, 
                 x='frequency', 
                 y='keywords', 
                 orientation='h',
                 title=f'Top {n} Keywords by Frequency',
                 labels={'frequency': 'Frequency', 'keywords': 'Keywords'},
                 color='frequency',
                 color_continuous_scale='Blues')
    fig.update_layout(height=1000, 
                      width=800,
                      yaxis={'categoryorder':'total ascending'},
                      clickmode='event+select')
    return fig

def display_keyword_details(df: pd.DataFrame, keyword: str):
    # keyword_data = df[df['keywords'] == keyword]
    # st.write(f"Statistics for keyword: **{keyword}**")
    # st.write(f"Total occurrences: {len(keyword_data)}")
    
    # top_subreddits = keyword_data['subreddit'].value_counts().head(5)
    # st.write("Top 5 subreddits for this keyword:")
    # st.dataframe(top_subreddits)
    
    # if 'body' in keyword_data.columns:
    #     st.write("Sample comments containing this keyword:")
    #     sample_comments = keyword_data['body'].sample(min(3, len(keyword_data))).tolist()
    #     for comment in sample_comments:
    #         st.text(comment[:200] + "..." if len(comment) > 200 else comment)
    # else:
    #     st.write("Comment body not available in the dataset.")
    
    # st.write("Other available information:")
    # for column in keyword_data.columns:
    #     if column not in ['keywords', 'subreddit', 'body']:
    #         st.write(f"{column}: {keyword_data[column].iloc[0]}")
    
    # st.markdown("---")
    
    # Add this section to display related research papers
    st.write("Related Research Papers:")
    results = hybrid_search(keyword, top_n=10)
    sorted_results = sorted(results, key=lambda x: datetime.strptime(x['Updated'], '%Y-%m-%d'), reverse=True)
    for result in sorted_results:
        st.write(f"**Title:** {result['Title']}")
        st.write(f"**Category:** {result['Category']}")
        st.write(f"**Updated:** {result['Updated']}")
        st.write(f"**Summary:** {truncate_summary(result['Summary'])}")
        st.write(f"**Link:** {result['Link']}")
        st.markdown("---")

def main():
    
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

        # Move custom_stop_words input here
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
        top_n_keywords = get_top_keywords(filtered_df, n_keywords)  # Use filtered_df here
        fig = create_keyword_chart(top_n_keywords, n_keywords)
        st.plotly_chart(fig, use_container_width=True, key="keyword_chart")

    with col2:
        selected_keywords = st.multiselect("Selected Keywords", options=top_n_keywords['keywords'].tolist())

        if selected_keywords:
            for keyword in selected_keywords:
                display_keyword_details(filtered_df, keyword)  # Use filtered_df here
        else:
            st.write("Select keywords from the multiselect box above to see details.")

if __name__ == "__main__":
    main()