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

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

# Load and preprocess the dataset
splits = {'train': 'data/train-00000-of-00001.parquet', 'validation': 'data/validation-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}
train_dataset = pd.read_parquet("hf://datasets/ilsilfverskiold/tech-keywords-topics-summary/" + splits["train"])
validation_dataset = pd.read_parquet("hf://datasets/ilsilfverskiold/tech-keywords-topics-summary/" + splits["validation"])
test_dataset = pd.read_parquet("hf://datasets/ilsilfverskiold/tech-keywords-topics-summary/" + splits["test"])

def preprocess_text(row):
    # Remove punctuation from text
    text = row['text']
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def label_keywords(row):
    text = preprocess_text(row).lower().split()
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

# Apply the functions to your dataset
train_dataset['text'] = train_dataset.apply(preprocess_text, axis=1)
train_dataset['labels'] = train_dataset.apply(label_keywords, axis=1).tolist()
validation_dataset['text'] = validation_dataset.apply(preprocess_text, axis=1)
validation_dataset['labels'] = validation_dataset.apply(label_keywords, axis=1).tolist()
test_dataset['text'] = test_dataset.apply(preprocess_text, axis=1)
test_dataset['labels'] = test_dataset.apply(label_keywords, axis=1).tolist()

# Filter out rows where 'labels' contains only 'O'
def contains_only_O(labels):
    return all(label == 'O' for label in labels)

# Apply the function and filter out rows where 'labels' contains only 'O'
train_dataset_filtered = train_dataset[~train_dataset['labels'].apply(contains_only_O)]

# Reset the index after filtering, if needed
train_dataset_filtered.reset_index(drop=True, inplace=True)

# Function to process dataset
def process_dataset(dataset):
    # Convert the 'labels' column from a list to a tuple for deduplication
    dataset['labels'] = dataset['labels'].apply(tuple)
    # Drop duplicates and reset the index
    return dataset[["text", "labels"]].drop_duplicates().reset_index(drop=True)

# Process all datasets
train_dataset_processed = process_dataset(train_dataset_filtered)
validation_dataset_processed = process_dataset(validation_dataset)
test_dataset_processed = process_dataset(test_dataset)

# Initialize the BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained('dbmdz/bert-large-cased-finetuned-conll03-english')

# Define label to id mapping
labels_to_ids = {'B-TEC': 1, 'I-TEC': 2, 'O': 0}

# Define id to label mapping
ids_to_labels = {1: 'B-TEC', 2: 'I-TEC', 0: 'O'}

# Set the maximum length for the input
MAX_LEN = int(1.5 * train_dataset_filtered['text'].str.split().str.len().max())

# Define a custom dataset
class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, labels_to_ids):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.labels_to_ids = labels_to_ids

    def __getitem__(self, index):
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
        return self.len

# Create datasets
train_dataset = CustomDataset(train_dataset_processed, tokenizer, MAX_LEN, labels_to_ids)
val_dataset = CustomDataset(validation_dataset_processed, tokenizer, MAX_LEN, labels_to_ids)
test_dataset = CustomDataset(test_dataset_processed, tokenizer, MAX_LEN, labels_to_ids)

# Create data loaders
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16)
test_dataloader = DataLoader(test_dataset, batch_size=16)

# Initialize the model
model = BertForTokenClassification.from_pretrained(
    'dbmdz/bert-large-cased-finetuned-conll03-english', 
    num_labels=len(labels_to_ids),
    ignore_mismatched_sizes=True
)

# Set up the optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Set up the learning rate scheduler
total_steps = len(train_dataloader) * 3  # 3 epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# We'll store a number of quantities such as training and validation loss,
# validation accuracy, and timings.
training_stats = []

# Measure the total training time for the whole run.
total_t0 = time.time()

# Training loop
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
    print("CUDA and MPS devices not found. Using CPU instead.")

model.to(device)

num_epochs = 3
for epoch in range(num_epochs):
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch + 1, num_epochs))
    print('Training...')

    # Measure how long the training epoch takes.
    t0 = time.time()

    # Reset the total loss for this epoch.
    total_train_loss = 0

    # Put the model into training mode.
    model.train()

    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):
        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        # Unpack this training batch from our dataloader.
        b_input_ids = batch['input_ids'].to(device)
        b_input_mask = batch['attention_mask'].to(device)
        b_labels = batch['labels'].to(device)

        # Clear any previously calculated gradients before performing a backward pass.
        model.zero_grad()

        # Perform a forward pass
        try:
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_labels)
        except RuntimeError as e:
            if "Placeholder storage has not been allocated on MPS device" in str(e):
                print("MPS device error encountered. Falling back to CPU.")
                device = torch.device("cpu")
                model.to(device)
                b_input_ids = b_input_ids.to(device)
                b_input_mask = b_input_mask.to(device)
                b_labels = b_labels.to(device)
                outputs = model(b_input_ids,
                                token_type_ids=None,
                                attention_mask=b_input_mask,
                                labels=b_labels)
            else:
                raise e

        # Accumulate the training loss
        loss = outputs.loss
        total_train_loss += loss.item()

        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters
        optimizer.step()

        # Update the learning rate.
        scheduler.step()

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)

    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epoch took: {:}".format(training_time))

    # ========================================
    #               Validation
    # ========================================

    print("")
    print("Running Validation...")

    t0 = time.time()

    # Put the model in evaluation mode
    model.eval()

    # Tracking variables 
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    # Evaluate data for one epoch
    for batch in val_dataloader:
        # Unpack this training batch from our dataloader.
        b_input_ids = batch['input_ids'].to(device)
        b_input_mask = batch['attention_mask'].to(device)
        b_labels = batch['labels'].to(device)

        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():        
            try:
                # Forward pass, calculate logit predictions.
                outputs = model(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask,
                                labels=b_labels)
            except RuntimeError as e:
                if "Placeholder storage has not been allocated on MPS device" in str(e):
                    print("MPS device error encountered. Falling back to CPU.")
                    device = torch.device("cpu")
                    model.to(device)
                    b_input_ids = b_input_ids.to(device)
                    b_input_mask = b_input_mask.to(device)
                    b_labels = b_labels.to(device)
                    outputs = model(b_input_ids,
                                    token_type_ids=None,
                                    attention_mask=b_input_mask,
                                    labels=b_labels)
                else:
                    raise e

        # Accumulate the validation loss.
        loss = outputs.loss
        total_eval_loss += loss.item()

        # Move logits and labels to CPU
        logits = outputs.logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences.
        # Ensure that logits and label_ids have the same shape
        pred_flat = np.argmax(logits, axis=2).flatten()
        labels_flat = label_ids.flatten()
        
        # Truncate the longer array to match the length of the shorter one
        min_len = min(len(pred_flat), len(labels_flat))
        pred_flat = pred_flat[:min_len]
        labels_flat = labels_flat[:min_len]
        
        total_eval_accuracy += np.sum(pred_flat == labels_flat) / len(labels_flat)
        nb_eval_steps += 1
        
    # Report the final accuracy for this validation run.
    avg_val_accuracy = total_eval_accuracy / nb_eval_steps
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / len(val_dataloader)
    
    # Measure how long the validation run took.
    validation_time = format_time(time.time() - t0)
    
    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    # Record all statistics from this epoch.
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

print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

# Save the model checkpoint
output_dir = "./model_checkpoint2"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save model
model.save_pretrained(output_dir)

# Save tokenizer
tokenizer.save_pretrained(output_dir)

# Save configuration
model.config.save_pretrained(output_dir)

# Save label mapping
with open(os.path.join(output_dir, "label_mapping.json"), "w") as f:
    json.dump(labels_to_ids, f)

print(f"Model checkpoint and artifacts saved to {output_dir}")
