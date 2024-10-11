"""
Module for training a sequence-to-sequence (seq2seq) model on a dataset.

This script loads a dataset, loads a pre-trained model and tokenizer, and trains the model on the dataset.
It also provides functions for loading data, loading models, and getting features for training.

Usage:
    python -m src.model_training
"""
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from transformers import TrainingArguments, Trainer
import torch
import hydra
from omegaconf import OmegaConf


def load_data(data):
    """
    Loads a dataset from Hugging Face datasets library.

    Args:
        data (str): Name of the dataset to load.

    Returns:
        Dataset: Loaded dataset.
    """
    dataset = load_dataset(data)
    return dataset

def load_model(model_name):
    """
    Loads a pre-trained model and tokenizer from the Hugging Face model hub.

    Args:
        model_name (str): Name of the model to load.

    Returns:
        tuple: A tuple containing a pre-trained tokenizer and model.
    """
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer,model

def get_feature(tokenizer,batch):
    """
    Encodes a batch of text and target keywords into a format suitable for
    training a seq2seq model.

    Args:
        tokenizer (transformers.AutoTokenizer): Tokenizer to use for encoding.
        batch (dict): A batch of data, containing 'text' and 'keywords' keys.

    Returns:
        dict: A dictionary containing 'input_ids', 'attention_mask', and 'labels'
              keys, suitable for training a seq2seq model.
    """
    encodings = tokenizer(batch['text'], text_target=batch['keywords'],
                            max_length=1024, truncation=True)

    encodings = {'input_ids': encodings['input_ids'],
                'attention_mask': encodings['attention_mask'],
                'labels': encodings['labels']}

    return encodings

def train_model(tokenizer, model, dataset, save_model_name, output_dir, cfg):
    """
    Trains a seq2seq model on a dataset.

    Args:
        tokenizer (transformers.AutoTokenizer): Tokenizer to use for encoding.
        model (transformers.AutoModelForSeq2SeqLM): Model to train.
        dataset (datasets.Dataset): Dataset to train on.
        save_model_name (str): Name to save the trained model as.
        output_dir (str): Directory to save the model in.
        cfg (omegaconf.DictConfig): Configuration for training.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f'device used: !!!!!{device}!!!!!')
    model.to(device)
    dataset_pt = dataset.map(lambda batch: get_feature(tokenizer, batch), batched=True)

    columns = ['input_ids', 'labels', 'attention_mask']
    dataset_pt.set_format(type='torch', columns=columns)
    # print(dataset_pt)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = TrainingArguments(
        output_dir = output_dir,
        num_train_epochs=cfg.training.num_train_epochs,
        warmup_steps = cfg.training.warmup_steps,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size, 
        per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
        weight_decay = cfg.training.weight_decay,
        logging_steps = cfg.training.logging_steps,
        evaluation_strategy = cfg.training.evaluation_strategy,
        eval_steps=cfg.training.eval_steps, 
        save_steps=cfg.training.save_steps,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps 
    )

    trainer = Trainer(model=model, args=training_args, tokenizer=tokenizer, data_collator=data_collator,
                  train_dataset = dataset_pt['train'], eval_dataset = dataset_pt['validation'])

    trainer.train()

    trainer.save_model(save_model_name)

@hydra.main(config_path="../conf", config_name="config")
def main(cfg):
    """
    Main function to train a model.

    Args:
        cfg (omegaconf.DictConfig): Configuration for training.

    Returns:
        None
    """
    model_name = cfg.base_model_name
    save_model_name = cfg.save_model_name
    output_dir = cfg.output_dir
    dataset = load_data(cfg.dataset_name)
    tokenizer, model = load_model(model_name)
    train_model(tokenizer, model, dataset, save_model_name, output_dir, cfg)

if __name__ == '__main__':
    '''python -m src.model_training'''
    main()