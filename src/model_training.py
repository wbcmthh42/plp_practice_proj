from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from transformers import TrainingArguments, Trainer
import torch
import hydra
from omegaconf import OmegaConf


def load_data(data):
    dataset = load_dataset(data)
    return dataset

def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer,model

def get_feature(tokenizer,batch):
    encodings = tokenizer(batch['text'], text_target=batch['keywords'],
                            max_length=1024, truncation=True)

    encodings = {'input_ids': encodings['input_ids'],
                'attention_mask': encodings['attention_mask'],
                'labels': encodings['labels']}

    return encodings

def train_model(tokenizer, model, dataset, save_model_name, output_dir, cfg):
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
    model_name = cfg.base_model_name
    save_model_name = cfg.save_model_name
    output_dir = cfg.output_dir
    dataset = load_data(cfg.dataset_name)
    tokenizer, model = load_model(model_name)
    train_model(tokenizer, model, dataset, save_model_name, output_dir, cfg)

if __name__ == '__main__':
    '''python -m src.model_training'''
    main()