import streamlit as st
import pandas as pd
import plotly.express as px
import re

# Set page config to wide mode
st.set_page_config(layout="wide")

# Define a list of common stop words
stop_words = set([
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

# Sample DataFrame
file_path = '/Users/tayjohnny/Documents/My_MTECH/PLP/plp_practice_proj/reddit_keywords_results/reddit_keywords_full_distilbert.csv'
df = pd.read_csv(file_path)

st.title('Reddit Keyword Frequency Analysis')

# Filter out rows with sentiment 'POSITIVE' or 'NEUTRAL'
filtered_df = df[~df['sentiment'].isin(['POSITIVE', 'NEUTRAL'])]

# Convert 'created_utc' to datetime format
df['created_utc'] = pd.to_datetime(df['created_utc'])
min_date = df['created_utc'].min().date()
max_date = df['created_utc'].max().date()

# Count the frequency of each keyword
filtered_df['keywords'] = filtered_df['keywords'].str.split(',')
exploded_df = filtered_df.explode('keywords')

# simple data pre-processing
exploded_df = exploded_df[~exploded_df['keywords'].isin(['', '[deleted]', '[removed]'])]

def preprocess_keywords(keyword):
    # Convert to lowercase and remove all non-alphabetic characters
    keyword = re.sub(r'[^a-z\s]', '', keyword.lower())
    # Split into words
    words = keyword.split()
    # Remove stopwords and get unique words
    return ' '.join(set(word for word in words if word not in stop_words and len(word) > 1)).upper()

exploded_df['keywords'] = exploded_df['keywords'].apply(preprocess_keywords)
exploded_df = exploded_df[exploded_df['keywords'] != '']  # Remove empty strings after preprocessing

# Count the frequency of each keyword under each subreddit
df1 = exploded_df.groupby(['subreddit', 'keywords']).size().reset_index(name='frequency')

# top frequency keywords regardless of subreddit
keyword_counts = exploded_df.groupby('keywords').size().reset_index(name='frequency')
top_20_keywords = keyword_counts.sort_values(by='frequency', ascending=False).head(30)

# Create two columns with adjusted ratio
col1, col2 = st.columns([2, 1])  # Adjust the ratio as needed

with col1:
    # Allow user to choose the number of top keywords
    n_keywords = st.slider("Select number of top keywords to display", min_value=5, max_value=50, value=20, step=5)

    # Update top keywords based on user selection
    top_n_keywords = keyword_counts.sort_values(by='frequency', ascending=False).head(n_keywords)

    # Create the bar chart using Plotly
    fig = px.bar(top_n_keywords, 
                 x='frequency', 
                 y='keywords', 
                 orientation='h',
                 title=f'Top {n_keywords} Keywords by Frequency',
                 labels={'frequency': 'Frequency', 'keywords': 'Keywords'},
                 color='frequency',
                 color_continuous_scale='Blues')

    # Customize the layout
    fig.update_layout(height=1000, 
                      width=800,  # You may need to adjust this
                      yaxis={'categoryorder':'total ascending'},
                      clickmode='event+select')

    # Display the chart
    selected_points = st.plotly_chart(fig, use_container_width=True, key="keyword_chart")

with col2:
    # Date range selector
    start_date, end_date = st.date_input('Select a date range from reddit:', [min_date, max_date], min_value=min_date, max_value=max_date)

    st.text("")

    # Create a container for the selected keywords
    selected_keywords = st.multiselect("Selected Keywords", options=top_n_keywords['keywords'].tolist())

    if selected_keywords:
        for keyword in selected_keywords:
            # Filter data for the selected keyword
            keyword_data = exploded_df[exploded_df['keywords'] == keyword]
            
            # Display keyword statistics
            st.write(f"Statistics for keyword: **{keyword}**")
            st.write(f"Total occurrences: {len(keyword_data)}")
            
            # Top subreddits for the keyword
            top_subreddits = keyword_data['subreddit'].value_counts().head(5)
            st.write("Top 5 subreddits for this keyword:")
            st.dataframe(top_subreddits)
            
            # Sample comments containing the keyword (if 'body' column exists)
            if 'body' in keyword_data.columns:
                st.write("Sample comments containing this keyword:")
                sample_comments = keyword_data['body'].sample(min(3, len(keyword_data))).tolist()
                for comment in sample_comments:
                    st.text(comment[:200] + "..." if len(comment) > 200 else comment)
            else:
                st.write("Comment body not available in the dataset.")
            
            # Display other available information
            st.write("Other available information:")
            for column in keyword_data.columns:
                if column not in ['keywords', 'subreddit', 'body']:
                    st.write(f"{column}: {keyword_data[column].iloc[0]}")
            
            st.markdown("---")  # Add a separator between keywords
    else:
        st.write("Select keywords from the multiselect box above to see details.")
