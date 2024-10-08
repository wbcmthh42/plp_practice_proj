"""
Module Description:
This script, `scrape_reddit.py`, is designed to scrape comments from specified subreddits within a date range. 
It uses the PRAW (Python Reddit API Wrapper) library to retrieve submission and comment data from Reddit, 
filters comments by date, and saves the results to a CSV file.
"""

import pandas as pd
import praw
from praw.models import MoreComments
from dotenv import load_dotenv
import os
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import date, timedelta, datetime
from prawcore.exceptions import RequestException
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import time
import logging
import hydra
from omegaconf import OmegaConf
from datetime import datetime
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
env_path = '.env'
load_dotenv(dotenv_path=env_path)
env_client_id = os.getenv("CLIENT_ID")
env_client_secret = os.getenv("SECRET_KEY")
env_user_agent = os.getenv("REDDIT_GRANT_TYPE")
env_username = os.getenv("REDDIT_USERNAME")
env_password = os.getenv("REDDIT_PASSWORD")

# Initialize Reddit instance
reddit = praw.Reddit(
    client_id=env_client_id,
    client_secret=env_client_secret,
    user_agent=env_user_agent,
    username=env_username,
    password=env_password
)

# Function to retrieve a list of submission IDs from given subreddits
def retrieve_list_of_submission_id(subreddit_name_list,limit):
    """
    Retrieves a list of submission IDs from the specified subreddits.

    Args:
        subreddit_name_list (list): A list of subreddit names to retrieve submissions from.
        limit (int): The number of submissions to retrieve per subreddit.

    Returns:
        list: A list of submission IDs.
    """
    submissions = []
    for subreddit_name in subreddit_name_list:
        for submission in reddit.subreddit(subreddit_name).new(limit=limit):
            submissions.append(submission.id)
    return submissions

# Function to fetch comments from a given submission within a date range
def fetch_comments_from_submission(submission_id, start_date, end_date):
    """
    Fetches comments from a given submission within the specified date range.

    This function extracts all comments from the submission, converts the timestamps to a readable format,
    and filters them by the given start and end dates.

    Args:
        submission_id (str): The ID of the submission to extract comments from.
        start_date (datetime): The start date to filter comments.
        end_date (datetime): The end date to filter comments.

    Returns:
        pd.DataFrame: A DataFrame containing the extracted comments.
    """
    comments_data = []
    logging.info('PRAW extraction in progress...')
    
    try:
        submission = reddit.submission(id=submission_id)

        submission.comments.replace_more(limit=None)

        for comment in submission.comments.list():
            comments_data.append({
                'author': comment.author.name if comment.author else '[deleted]',
                'body': comment.body,
                'created_utc': comment.created_utc,
                'id': comment.id,
                'submission': submission.id,
                'subreddit': submission.subreddit.display_name,
                'subreddit_id': submission.subreddit_id
            })
    
    except RequestException as e:
        logging.error(f"Waiting for 120 seconds due to Reddit API error: {e}")
        time.sleep(120)
    
    except Exception as e:
        logging.error(f"An unknown error occurred: {e}")
    
    if comments_data:
        dfComment = pd.DataFrame(comments_data)
        dfComment['created_utc'] = pd.to_datetime(dfComment['created_utc'], unit='s')
        dfComment = dfComment[(dfComment['created_utc'] >= start_date) & (dfComment['created_utc'] <= end_date)]
    else:
        dfComment = pd.DataFrame(columns=[
            'author', 'body', 'created_utc', 'id', 
            'submission', 'subreddit', 'subreddit_id'
        ])

    return dfComment

def scrape_reddit_comments(start_date_str, end_date_str, reddit_list, limit, file_name):
    """
    Scrapes comments from the specified subreddits within a given date range and saves them to a CSV file.

    This function retrieves submission IDs from the specified subreddits, fetches comments from each submission,
    filters the comments by date, and saves the resulting DataFrame to a CSV file.

    Args:
        start_date_str (str): The start date in 'YYYY-MM-DD' format.
        end_date_str (str): The end date in 'YYYY-MM-DD' format.
        reddit_list (list): A list of subreddit names to scrape.
        limit (int): The number of submissions to scrape from each subreddit.
        file_name (str): The name of the CSV file to save the comments.
    
    Returns:
        None
    """
    # Convert dates to datetime format
    start_date = pd.to_datetime(start_date_str)
    end_date = pd.to_datetime(end_date_str)

    # Initialize an empty DataFrame to store all comments
    final_df = pd.DataFrame(columns=[
        'author', 'body', 'created_utc', 'id', 
        'submission', 'subreddit', 'subreddit_id'
    ])

    # Measure time for the entire scraping process
    start_time = time.time()
    

    # Loop over the subreddits and fetch comments
    for submission_id in retrieve_list_of_submission_id(reddit_list, limit):
        comments_df = fetch_comments_from_submission(submission_id, start_date, end_date)
        final_df = pd.concat([final_df, comments_df], ignore_index=True)

    # Remove duplicates and save the final DataFrame to CSV
    final_df['body'] = final_df['body'].astype(str)
    final_df.drop_duplicates().to_csv(file_name, index=False)
    
 
    logging.info(f"Data successfully saved to {file_name}")
    
    # Calculate and log the time taken
    end_time = time.time()
    logging.info(f'Time taken: {end_time - start_time:.2f} seconds')
    logging.info(f'Length of final DataFrame: {len(final_df)}')

   
@hydra.main(config_path="../conf", config_name="config", version_base="1.1")
def main(cfg):
    """
    Main entry point for scraping Reddit comments.

    This function reads the configuration file, sets up the date range and subreddits to scrape, 
    and calls the `scrape_reddit_comments` function to extract and save comments.

    Args:
        cfg (DictConfig): A Hydra configuration object containing PRAW parameters such as the start date, 
                          subreddits, and output file name.
    
    Returns:
        None
    """
    # Define date range and subreddits to scrape
    # there is a limitation by PRAW API as to how far back can comments be retrieved
    start_date = cfg.praw.start_date
    end_date = datetime.today().strftime('%Y-%m-%d')
    file_name = cfg.praw.praw_output
    reddit_list = cfg.praw.subreddits
    limit= 5
    # Call the scraping function
    scrape_reddit_comments(start_date, end_date, reddit_list, limit, file_name)
 
if __name__ == '__main__':
    '''python -m src.scrape_reddit'''
    main()