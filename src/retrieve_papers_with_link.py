"""
Module for retrieving arXiv papers with links.

This module provides functions for retrieving arXiv papers and saving them to a CSV file.

Functions:
    save_to_csv: Saves the retrieved papers to a CSV file.
    main: Retrieves arXiv papers and saves them to a CSV file.
"""
import requests
import xml.etree.ElementTree as ET
import pandas as pd
import time
import logging
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import hydra
from omegaconf import DictConfig

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def save_to_csv(data, filename):
    """
    Saves the given data to a CSV file with the given filename.

    Args:
        data: List of dictionaries containing the data to be saved.
        filename: String representing the filename of the CSV file to be saved.
    """
    df = pd.DataFrame(data)
    df['Updated'] = pd.to_datetime(df['Updated'])
    df.to_csv(filename, index=False)
    logger.info(f"Saved {len(data)} records to {filename}")

def retrieve_papers_with_link(from_date, until_date, max_results=None):
    # Base OAI-PMH URL for arXiv
    """
    Retrieves the list of papers from arXiv with links for the given date range.

    Args:
        from_date: String representing the start date (YYYY-MM-DD).
        until_date: String representing the end date (YYYY-MM-DD).
        max_results: Optional integer representing the maximum number of results to retrieve.

    Returns:
        List of dictionaries containing the paper title, summary, category, link, and updated date.
    """
    base_url = 'http://export.arxiv.org/oai2?verb=ListRecords'

    # Search parameters for OAI-PMH (for Computer Science category and metadata format 'oai_dc')
    metadata_format = 'oai_dc'
    category = 'cs'  # Computer Science category

    # List to store the data
    data = []

    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount('http://', HTTPAdapter(max_retries=retries))

    resumption_token = None
    batch_count = 0
    logger.info(f"Starting retrieval of papers from {from_date} to {until_date}")

    while True:
        try:
            if resumption_token:
                url = f'{base_url}&resumptionToken={resumption_token}'
            else:
                url = f'{base_url}&metadataPrefix={metadata_format}&from={from_date}&until={until_date}&set={category}'

            logger.info(f"Fetching batch {batch_count + 1}")
            response = session.get(url)
            response.raise_for_status()
            root = ET.fromstring(response.text)

            records = root.findall('.//{http://www.openarchives.org/OAI/2.0/}record')
            
            for record in records:
                metadata = record.find('.//{http://www.openarchives.org/OAI/2.0/oai_dc/}dc')
                
                if metadata is not None:
                    title = metadata.find('.//{http://purl.org/dc/elements/1.1/}title').text if metadata.find('.//{http://purl.org/dc/elements/1.1/}title') is not None else 'N/A'
                    summary = metadata.find('.//{http://purl.org/dc/elements/1.1/}description').text if metadata.find('.//{http://purl.org/dc/elements/1.1/}description') is not None else 'N/A'
                    updated = record.find('.//{http://www.openarchives.org/OAI/2.0/}datestamp').text
                    category = metadata.find('.//{http://purl.org/dc/elements/1.1/}subject').text if metadata.find('.//{http://purl.org/dc/elements/1.1/}subject') is not None else 'N/A'
                    
                    # Extract the identifier (article link)
                    identifier = metadata.find('.//{http://purl.org/dc/elements/1.1/}identifier').text if metadata.find('.//{http://purl.org/dc/elements/1.1/}identifier') is not None else 'N/A'
                    
                    # Convert arXiv identifier to URL
                    if identifier.startswith('http://arxiv.org/abs/'):
                        link = identifier
                    elif identifier.startswith('arXiv:'):
                        link = f'http://arxiv.org/abs/{identifier[6:]}'
                    else:
                        link = f'http://arxiv.org/abs/{identifier}'
                    
                    # Append the data to the list
                    data.append({
                        'Title': title,
                        'Summary': summary,
                        'Updated': updated,
                        'Category': category,
                        'Link': link
                    })

                    # Save to CSV every 100 rows
                    if len(data) % 100 == 0:
                        save_to_csv(data, 'arxiv_papers_2022_2024_with_links_partial.csv')

                    if max_results and len(data) >= max_results:
                        logger.info(f"Reached maximum results limit of {max_results}")
                        break

            batch_count += 1
            logger.info(f"Processed batch {batch_count}. Total records: {len(data)}")

            if max_results and len(data) >= max_results:
                break

            resumption_token_elem = root.find('.//{http://www.openarchives.org/OAI/2.0/}resumptionToken')
            if resumption_token_elem is None or not resumption_token_elem.text:
                logger.info("No more results to fetch")
                break
            
            resumption_token = resumption_token_elem.text
            logger.info(f"Waiting before next request. Resumption token: {resumption_token}")
            time.sleep(1)  # Add a delay between requests

        except requests.exceptions.RequestException as e:
            logger.error(f"Error occurred: {e}")
            logger.info("Waiting 5 seconds before retrying...")
            time.sleep(5)  # Wait before retrying

    logger.info(f"Retrieval complete. Total records: {len(data)}")

    return data

@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """
    Main function.

    Retrieves arXiv papers based on the configuration provided by Hydra.
    Saves the results to the specified output file.
    """
    logger.info("Starting main function")
    data = retrieve_papers_with_link(
        cfg.date_range.from_date,
        cfg.date_range.until_date,
        max_results=cfg.max_results
    )
    
    output_file = cfg.output_file
    logger.info(f"Saving final results to {output_file}")
    save_to_csv(data, output_file)
    logger.info(f"Final results saved to {output_file}")

if __name__ == "__main__":
    main()