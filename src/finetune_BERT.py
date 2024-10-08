import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast, BertForTokenClassification, AdamW, get_linear_schedule_with_warmup
import string
import time
import os
import json
import datetime
import sys
import hydra
from omegaconf import DictConfig
from datasets import load_dataset

class BERTFineTuner:
    """
    A class for fine-tuning BERT models for token classification tasks.

    This class handles the entire process of fine-tuning a BERT model, including
    data loading, preprocessing, model setup, training, and evaluation.

    Attributes:
        tokenizer (BertTokenizerFast): The BERT tokenizer.
        model (BertForTokenClassification): The BERT model for token classification.
        device (torch.device): The device (CPU/GPU) to run the model on.
        labels_to_ids (dict): Mapping from label names to IDs.
        ids_to_labels (dict): Mapping from IDs to label names.
        MAX_LEN (int): Maximum sequence length for input.
        pretrained_model (str): Name of the pretrained BERT model to use.
        max_len (int): Maximum length for tokenization.
        batch_size (int): Batch size for training and evaluation.
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for optimization.
        output_dir (str): Directory to save the fine-tuned model.
    """

    def __init__(self, cfg):
        """
        Initializes the BERTFineTuner with default settings.
        """
        self.cfg = cfg  # Store the configuration as an instance variable
        self.tokenizer = None
        self.model = None
        self.device = None
        self.labels_to_ids = {'B-TEC': 1, 'I-TEC': 2, 'O': 0}
        self.ids_to_labels = {1: 'B-TEC', 2: 'I-TEC', 0: 'O'}
        self.MAX_LEN = None
        
        # Configuration parameters from Hydra
        self.pretrained_model = self.cfg.bert.pretrained_model
        self.max_len = self.cfg.bert.max_len
        self.batch_size = self.cfg.bert.batch_size
        self.num_epochs = self.cfg.bert.num_epochs
        self.learning_rate = self.cfg.bert.learning_rate
        self.output_dir = self.cfg.bert.output_dir


    def format_time(self, elapsed):
        """
        Formats a time duration into a string.

        Args:
            elapsed (float): Time duration in seconds.

        Returns:
            str: Formatted time string in the format "hh:mm:ss".
        """
        elapsed_rounded = int(round((elapsed)))
        return str(datetime.timedelta(seconds=elapsed_rounded))
    
    def load_datasets(self, cfg):
        """
        Loads the train, validation, and test datasets.

        Returns:
            tuple: Containing train_dataset, validation_dataset, and test_dataset.
        """
        splits = {
            'train': cfg.bert.datasets.train,
            'validation': cfg.bert.datasets.validation,
            'test': cfg.bert.datasets.test
        }
        train_dataset = pd.read_parquet(splits["train"])
        validation_dataset = pd.read_parquet(splits["validation"])
        test_dataset = pd.read_parquet(splits["test"])
        return train_dataset, validation_dataset, test_dataset

    def preprocess_text(self, row):
        """
        Preprocesses the text by removing punctuation.

        Args:
            row (pandas.Series): A row from the dataset.

        Returns:
            str: Preprocessed text with punctuation removed.
        """
        text = row['text']
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text

    def label_keywords(self, row):
        """
        Labels the keywords in the text.

        Args:
            row (pandas.Series): A row from the dataset.

        Returns:
            list: A list of labels for each word in the text.
        """
        text = self.preprocess_text(row).lower().split()
        keywords = row['keywords'].lower()
        keywords = [keyword.split() for keyword in keywords.split(', ')]
        labels = ['O'] * len(text)
        
        for keyword_parts in keywords:
            keyword_len = len(keyword_parts)
            for i in range(len(text) - keyword_len + 1):
                if text[i:i + keyword_len] == keyword_parts:
                    if i < len(labels):
                        labels[i] = 'B-TEC'
                    for j in range(1, keyword_len):
                        if i + j < len(labels):
                            labels[i + j] = 'I-TEC'
        
        return labels

    def process_datasets(self, train_dataset, validation_dataset, test_dataset):
        """
        Processes the datasets by applying preprocessing and labeling.

        Args:
            train_dataset (pandas.DataFrame): The training dataset.
            validation_dataset (pandas.DataFrame): The validation dataset.
            test_dataset (pandas.DataFrame): The test dataset.

        Returns:
            tuple: Containing processed train, validation, and test datasets, and filtered train dataset.
        """
        # Apply the functions to your dataset
        train_dataset['text'] = train_dataset.apply(self.preprocess_text, axis=1)
        train_dataset['labels'] = train_dataset.apply(self.label_keywords, axis=1)
        validation_dataset['text'] = validation_dataset.apply(self.preprocess_text, axis=1)
        validation_dataset['labels'] = validation_dataset.apply(self.label_keywords, axis=1)
        test_dataset['text'] = test_dataset.apply(self.preprocess_text, axis=1)
        test_dataset['labels'] = test_dataset.apply(self.label_keywords, axis=1)

        # Filter out rows where 'labels' contains only 'O'
        train_dataset_filtered = train_dataset[train_dataset['labels'].apply(lambda labels: any(label != 'O' for label in labels))]
        train_dataset_filtered = train_dataset_filtered.reset_index(drop=True)

        # Process all datasets
        train_dataset_processed = self.process_dataset(train_dataset_filtered)
        validation_dataset_processed = self.process_dataset(validation_dataset)
        test_dataset_processed = self.process_dataset(test_dataset)

        return train_dataset_processed, validation_dataset_processed, test_dataset_processed, train_dataset_filtered

    def process_dataset(self, dataset):
        """
        Processes a single dataset by converting labels to tuples and removing duplicates.

        Args:
            dataset (pandas.DataFrame): The dataset to process.

        Returns:
            pandas.DataFrame: The processed dataset.
        """
        # Convert the 'labels' column from a list to a tuple for deduplication
        dataset['labels'] = dataset['labels'].apply(tuple)
        # Drop duplicates and reset the index
        return dataset[["text", "labels"]].drop_duplicates().reset_index(drop=True)

    def setup_tokenizer_and_model(self):
        """
        Sets up the BERT tokenizer and model.
        """
        self.tokenizer = BertTokenizerFast.from_pretrained(self.pretrained_model)
        self.model = BertForTokenClassification.from_pretrained(
            self.pretrained_model, 
            num_labels=len(self.labels_to_ids),
            ignore_mismatched_sizes=True
        )

    def create_datasets(self, train_dataset_processed, validation_dataset_processed, test_dataset_processed):
        """
        Creates CustomDataset objects for train, validation, and test sets.

        Args:
            train_dataset_processed (pandas.DataFrame): Processed training dataset.
            validation_dataset_processed (pandas.DataFrame): Processed validation dataset.
            test_dataset_processed (pandas.DataFrame): Processed test dataset.

        Returns:
            tuple: Containing train, validation, and test CustomDataset objects.
        """
        self.MAX_LEN = int(1.5 * train_dataset_processed['text'].str.split().str.len().max())
        train_dataset = CustomDataset(train_dataset_processed, self.tokenizer, self.MAX_LEN, self.labels_to_ids)
        val_dataset = CustomDataset(validation_dataset_processed, self.tokenizer, self.MAX_LEN, self.labels_to_ids)
        test_dataset = CustomDataset(test_dataset_processed, self.tokenizer, self.MAX_LEN, self.labels_to_ids)
        return train_dataset, val_dataset, test_dataset

    def create_dataloaders(self, train_dataset, val_dataset, test_dataset):
        """
        Creates DataLoader objects for train, validation, and test sets.

        Args:
            train_dataset (CustomDataset): Training dataset.
            val_dataset (CustomDataset): Validation dataset.
            test_dataset (CustomDataset): Test dataset.

        Returns:
            tuple: Containing train, validation, and test DataLoader objects.
        """
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size)
        return train_dataloader, val_dataloader, test_dataloader

    def setup_training(self, train_dataloader):
        """
        Sets up the optimizer and learning rate scheduler for training.

        Args:
            train_dataloader (DataLoader): DataLoader for the training set.

        Returns:
            tuple: Containing the optimizer and scheduler.
        """
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        total_steps = len(train_dataloader) * self.num_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        return optimizer, scheduler

    def setup_device(self):
        """
        Sets up the device (CPU/GPU) for training.
        """
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
            print("CUDA and MPS devices not found. Using CPU instead.")
        self.model.to(self.device)

    def train(self, train_dataloader, val_dataloader, optimizer, scheduler):
        """
        Trains the model using the provided data loaders.

        Args:
            train_dataloader (DataLoader): DataLoader for the training set.
            val_dataloader (DataLoader): DataLoader for the validation set.
            optimizer (torch.optim.Optimizer): The optimizer for training.
            scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.

        Returns:
            list: A list of dictionaries containing training statistics for each epoch.
        """
        training_stats = []
        total_t0 = time.time()

        for epoch in range(self.num_epochs):
            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch + 1, self.num_epochs))
            print('Training...')

            t0 = time.time()
            total_train_loss = 0
            self.model.train()

            for step, batch in enumerate(train_dataloader):
                if step % 40 == 0 and not step == 0:
                    elapsed = self.format_time(time.time() - t0)
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

                b_input_ids = batch['input_ids'].to(self.device)
                b_input_mask = batch['attention_mask'].to(self.device)
                b_labels = batch['labels'].to(self.device)

                self.model.zero_grad()

                try:
                    outputs = self.model(b_input_ids,
                                    token_type_ids=None,
                                    attention_mask=b_input_mask,
                                    labels=b_labels)
                except RuntimeError as e:
                    if "Placeholder storage has not been allocated on MPS device" in str(e):
                        print("MPS device error encountered. Falling back to CPU.")
                        self.device = torch.device("cpu")
                        self.model.to(self.device)
                        b_input_ids = b_input_ids.to(self.device)
                        b_input_mask = b_input_mask.to(self.device)
                        b_labels = b_labels.to(self.device)
                        outputs = self.model(b_input_ids,
                                        token_type_ids=None,
                                        attention_mask=b_input_mask,
                                        labels=b_labels)
                    else:
                        raise e

                loss = outputs.loss
                total_train_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            avg_train_loss = total_train_loss / len(train_dataloader)
            training_time = self.format_time(time.time() - t0)

            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epoch took: {:}".format(training_time))

            print("")
            print("Running Validation...")

            t0 = time.time()
            self.model.eval()

            total_eval_accuracy = 0
            total_eval_loss = 0
            nb_eval_steps = 0

            for batch in val_dataloader:
                b_input_ids = batch['input_ids'].to(self.device)
                b_input_mask = batch['attention_mask'].to(self.device)
                b_labels = batch['labels'].to(self.device)

                with torch.no_grad():        
                    try:
                        outputs = self.model(b_input_ids, 
                                        token_type_ids=None, 
                                        attention_mask=b_input_mask,
                                        labels=b_labels)
                    except RuntimeError as e:
                        if "Placeholder storage has not been allocated on MPS device" in str(e):
                            print("MPS device error encountered. Falling back to CPU.")
                            self.device = torch.device("cpu")
                            self.model.to(self.device)
                            b_input_ids = b_input_ids.to(self.device)
                            b_input_mask = b_input_mask.to(self.device)
                            b_labels = b_labels.to(self.device)
                            outputs = self.model(b_input_ids,
                                            token_type_ids=None,
                                            attention_mask=b_input_mask,
                                            labels=b_labels)
                        else:
                            raise e

                loss = outputs.loss
                total_eval_loss += loss.item()

                logits = outputs.logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                pred_flat = np.argmax(logits, axis=2).flatten()
                labels_flat = label_ids.flatten()
                
                min_len = min(len(pred_flat), len(labels_flat))
                pred_flat = pred_flat[:min_len]
                labels_flat = labels_flat[:min_len]
                
                total_eval_accuracy += np.sum(pred_flat == labels_flat) / len(labels_flat)
                nb_eval_steps += 1
                
            avg_val_accuracy = total_eval_accuracy / nb_eval_steps
            print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

            avg_val_loss = total_eval_loss / len(val_dataloader)
            
            validation_time = self.format_time(time.time() - t0)
            
            print("  Validation Loss: {0:.2f}".format(avg_val_loss))
            print("  Validation took: {:}".format(validation_time))

            training_stats.append(
                {
                    'epoch': epoch + 1,
                    'Training Loss': avg_train_loss,
                    'Valid. Loss': avg_val_loss,
                    'Valid. Accur.': avg_val_accuracy,
                    'Training Time': training_time,
                    'Validation Time': validation_time
                }
            )

        print("")
        print("Training complete!")
        print("Total training took {:} (h:mm:ss)".format(self.format_time(time.time()-total_t0)))

        return training_stats

    def save_model(self):
        """
        Saves the fine-tuned model, tokenizer, and label mapping.
        """
        output_dir = self.output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        self.model.config.save_pretrained(output_dir)

        with open(os.path.join(output_dir, "label_mapping.json"), "w") as f:
            json.dump(self.labels_to_ids, f)

        print(f"Model checkpoint and artifacts saved to {output_dir}")

    def run(self, cfg):
        """
        Runs the entire fine-tuning process.
        """
        train_dataset, validation_dataset, test_dataset = self.load_datasets(cfg)
        train_dataset_processed, validation_dataset_processed, test_dataset_processed, train_dataset_filtered = self.process_datasets(train_dataset, validation_dataset, test_dataset)
        self.setup_tokenizer_and_model()
        train_dataset, val_dataset, test_dataset = self.create_datasets(train_dataset_processed, validation_dataset_processed, test_dataset_processed)
        train_dataloader, val_dataloader, test_dataloader = self.create_dataloaders(train_dataset, val_dataset, test_dataset)
        optimizer, scheduler = self.setup_training(train_dataloader)
        self.setup_device()
        training_stats = self.train(train_dataloader, val_dataloader, optimizer, scheduler)
        self.save_model()

class CustomDataset(Dataset):
    """
    A custom dataset for token classification tasks.

    This dataset handles the tokenization and label encoding for BERT input.

    Attributes:
        len (int): The length of the dataset.
        data (pandas.DataFrame): The dataset.
        tokenizer (BertTokenizerFast): The BERT tokenizer.
        max_len (int): Maximum sequence length.
        labels_to_ids (dict): Mapping from label names to IDs.
    """

    def __init__(self, dataframe, tokenizer, max_len, labels_to_ids):
        """
        Initializes the CustomDataset.

        Args:
            dataframe (pandas.DataFrame): The dataset.
            tokenizer (BertTokenizerFast): The BERT tokenizer.
            max_len (int): Maximum sequence length.
            labels_to_ids (dict): Mapping from label names to IDs.
        """
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.labels_to_ids = labels_to_ids

    def __getitem__(self, index):
        """
        Gets an item from the dataset.

        Args:
            index (int): The index of the item.

        Returns:
            dict: A dictionary containing the tokenized input and labels.
        """
        # Get the text and labels
        text = self.data.text.iloc[index]
        word_labels = self.data.labels.iloc[index]

        # Convert string labels to ids
        word_labels = [self.labels_to_ids[label] for label in word_labels]

        # Tokenize the text
        encoding = self.tokenizer(text.split(),
                                  is_split_into_words=True,
                                  return_offsets_mapping=True,
                                  padding='max_length',
                                  truncation=True,
                                  max_length=self.max_len)

        # Create the labels for each token
        labels = []
        word_idx = 0
        for offset in encoding.offset_mapping:
            if offset[0] == 0 and offset[1] != 0:
                if word_idx < len(word_labels):
                    labels.append(word_labels[word_idx])
                    word_idx += 1
                else:
                    labels.append(-100)  # Use padding label if we've run out of word labels
            elif offset[0] == offset[1]:
                labels.append(-100)
            else:
                if word_idx < len(word_labels):
                    labels.append(word_labels[word_idx])
                else:
                    labels.append(-100)  # Use padding label if we've run out of word labels

        # Ensure labels list is the same length as the input_ids
        if len(labels) < len(encoding['input_ids']):
            labels.extend([-100] * (len(encoding['input_ids']) - len(labels)))
        elif len(labels) > len(encoding['input_ids']):
            labels = labels[:len(encoding['input_ids'])]

        # Convert to tensor
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item['labels'] = torch.as_tensor(labels)

        return item

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return self.len

@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    fine_tuner = BERTFineTuner(cfg)  # Pass the cfg to the BERTFineTuner
    fine_tuner.run(cfg)  # Call the run method to start the fine-tuning process

if __name__ == "__main__":
    # fine_tuner = BERTFineTuner()
    main()