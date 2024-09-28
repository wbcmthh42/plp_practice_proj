import streamlit as st
import pandas as pd
import altair as alt


# Sample DataFrame
file_path = '/Users/suqiulin/Desktop/plp_practice_proj/reddit_keywords_results/reddit_keywords_full_distilbert.csv'
df=pd.read_csv(file_path)

st.title('Reddit Keyword Frequency Analysis')

# Filter out rows with sentiment 'POSITIVE' or 'NEUTRAL'
filtered_df = df[~df['sentiment'].isin(['POSITIVE', 'NEUTRAL'])]

# Convert 'created_utc' to datetime format

df['created_utc'] = pd.to_datetime(df['created_utc'])
min_date = df['created_utc'].min().date()
max_date = df['created_utc'].max().date()
start_date, end_date = st.date_input('Select a date range from reddit:', [min_date, max_date], min_value=min_date, max_value=max_date)

# Count the frequency of each keyword
filtered_df = df[(df['created_utc'] >= pd.to_datetime(start_date)) & (df['created_utc'] <= pd.to_datetime(end_date))]
filtered_df['keywords'] = filtered_df['keywords'].str.split(',')
exploded_df = filtered_df.explode('keywords')

# simple data pre-processing
exploded_df = exploded_df[~exploded_df['keywords'].isin(['', '[deleted]'])]


# Count the frequency of each keyword under each subreddit
df1 = exploded_df.groupby(['subreddit', 'keywords']).size().reset_index(name='frequency')

# top frequency keywords regardless of subreddit
keyword_counts = exploded_df.groupby('keywords').size().reset_index(name='frequency')
top_20_keywords = keyword_counts.sort_values(by='frequency', ascending=False).head(30)

st.text("")

# Display the top 20 keywords chart using Altair

chart = alt.Chart(top_20_keywords).mark_bar().encode(
    x=alt.X('frequency:Q', title='Frequency'),
    y=alt.Y('keywords:N', sort='-x', title='Keywords'),
    color=alt.Color('frequency:Q', scale=alt.Scale(scheme='blues'), legend=None)
).properties(
    width=700,
    height=800,
    title='Top Keywords by Frequency'
)

# Display the chart
st.altair_chart(chart, use_container_width=False)


# Dropdown list to select a subreddit
subreddit_list = df1['subreddit'].unique()
selected_subreddit = st.selectbox('Select a subreddit to view top keywords:', subreddit_list)

# Filter and display top 5 keywords based on frequency for the selected subreddit
if selected_subreddit:
    filtered_df1 = df1[df1['subreddit'] == selected_subreddit].sort_values(by='frequency', ascending=False)
    top_keywords = filtered_df1.head(5)
    
    # Display the results
    st.text("")
    st.write(f"Top 5 keywords for subreddit: **{selected_subreddit}** from {start_date} to {end_date}")
    st.dataframe(top_keywords)