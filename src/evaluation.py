"""
Module for evaluating the performance of a trained seq2seq model.

This script loads a trained model, loads a dataset, and evaluates the model's performance on the dataset.
It also provides functions for loading data, loading models, and getting features for evaluation.

Usage:
    python -m src.evaluation
"""
from datasets import load_dataset
from transformers import pipeline
from bert_score import score
import pandas as pd
import logging
import csv
import hydra
from omegaconf import OmegaConf
import os


def evaluate(model_name, dataset, results_file):
    """
    Evaluate a model's performance on a dataset.

    Args:
        model_name (str): The name of the model to evaluate.
        dataset (DatasetDict): A dataset containing test data.
        results_file (str): The file where the results should be written to.

    This function evaluates a model's performance by generating summaries of all the text in the dataset and comparing them to the ground truth. The results are written to a csv file.
    """
    pipe = pipeline('summarization', model=model_name, max_length=512, truncation=True)

    with open(results_file, 'w', newline='') as csvfile:
        fieldnames = ['text', 'keywords', 'ground_truth', 'P', 'R', 'F1']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for i in range(len(dataset['test'])):
            keywords = pipe(dataset['test'][i]['text'])[0]['summary_text']
            ground_truth = dataset['test'][i]['keywords']

            P, R, F1 = score([keywords], [ground_truth], lang="en", verbose=True)

            writer.writerow({
                'text': dataset['test'][i]['text'],
                'keywords': keywords,
                'ground_truth': ground_truth,
                'P': P.item(),
                'R': R.item(),
                'F1': F1.item()
            })

def evaluate_bert(model_name, dataset, results_file):
    """
    Evaluate a BERT model's performance on a dataset.

    Args:
        model_name (str): The name of the BERT model to evaluate.
        dataset (DatasetDict): A dataset containing test data.
        results_file (str): The file where the results should be written to.

    This function evaluates a BERT model's performance by generating summaries of all the text in the dataset and comparing them to the ground truth. The results are written to a csv file.
    """
    pipe = pipeline('token-classification', model=model_name)

    with open(results_file, 'w', newline='') as csvfile:
        fieldnames = ['text', 'keywords', 'ground_truth', 'P', 'R', 'F1']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for i in range(len(dataset['test'])):
            text = dataset['test'][i]['text']
            results = pipe(text)
            
            # Extract keywords from the token classification results
            keywords = ' '.join(set([item['word'] for item in results if item['score'] > 0.5]))  # Adjust threshold as needed

            ground_truth = dataset['test'][i]['keywords']

            P, R, F1 = score([keywords], [ground_truth], lang="en", verbose=True)

            writer.writerow({
                'text': text,
                'keywords': keywords,
                'ground_truth': ground_truth,
                'P': P.item(),
                'R': R.item(),
                'F1': F1.item()
            })

def get_dataset_average_score(csv_data):
    """
    Calculate and log the average P, R, and F1 scores of a set of summaries in a csv file.

    Args:
        csv_data (str): The path to the csv file containing the results.

    This function reads a csv file containing the results of evaluating a model on a dataset, calculates the average P, R, and F1 scores, and logs the results.
    """
    results = pd.read_csv(csv_data, header=0)
    average_P, average_R, average_F1 = results[['P', 'R', 'F1']].mean()
    logging.info(f"dataset_average_score_results from {csv_data}:")
    logging.info(f"System level average P score: {average_P:.3f}")
    logging.info(f"System level average R score: {average_R:.3f}")
    logging.info(f"System level average F1 score: {average_F1:.3f}")


@hydra.main(config_path="../conf", config_name="config")
def main(cfg):
    """
    Evaluate a model on a dataset.

    This function loads a dataset and a model, evaluates the model on the dataset, and logs the results. The results are written to a csv file. The function then calculates the average P, R, and F1 scores of the results and logs them.

    Args:
        cfg (OmegaConf): The Hydra configuration object. The following fields are used:
            - eval.evaluation_model_name (str): The name of the model to evaluate.
            - dataset_name (str): The name of the dataset to evaluate the model on.
            - eval.results_file (str): The file where the results should be written to.

    This function is the main entry point of the script when it is run as a Python module. Hydra is used to handle configuration and logging.
    """
    model_name = cfg.eval.evaluation_model_name
    dataset = load_dataset(cfg.dataset_name)
    results_file = cfg.eval.results_file
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    # Determine which evaluation function to use based on the model name
    if 'bert' in model_name.lower():
        evaluate_bert(model_name, dataset, results_file)
    elif 't5' in model_name.lower() or 'bart' in model_name.lower():
        evaluate(model_name, dataset, results_file)
    else:
        raise ValueError(f"Unsupported model type: {model_name}. Please use a BERT, T5, or BART model.")
    
    get_dataset_average_score(results_file)

if __name__ == '__main__':
    '''python -m src.evaluation'''
    main()