from datasets import load_dataset
from transformers import pipeline
from bert_score import score
import pandas as pd
import logging
import csv
import hydra
from omegaconf import OmegaConf

def evaluate(model_name, dataset, results_file):
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

def get_dataset_average_score(csv_data):
    results = pd.read_csv(csv_data, header=0)
    average_P, average_R, average_F1 = results[['P', 'R', 'F1']].mean()
    logging.info(f"dataset_average_score_results from {csv_data}:")
    logging.info(f"System level average P score: {average_P:.3f}")
    logging.info(f"System level average R score: {average_R:.3f}")
    logging.info(f"System level average F1 score: {average_F1:.3f}")


@hydra.main(config_path="../conf", config_name="config")
def main(cfg):
    model_name = cfg.eval.evaluation_model_name
    dataset = load_dataset(cfg.dataset_name)
    results_file = cfg.eval.results_file
    evaluate(model_name, dataset, results_file)
    get_dataset_average_score(results_file)

if __name__ == '__main__':
    '''python -m src.evaluation'''
    main()