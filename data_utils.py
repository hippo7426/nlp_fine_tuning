import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
from datasets import Dataset as HFDataset
from sklearn.model_selection import train_test_split
import pandas as pd
from typing import List, Dict, Tuple
from config import TrainingConfig

class PoemDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer: PreTrainedTokenizer, max_length: int):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        # Combine prompt and poem for training
        full_text = item['prompt'] + item['poem']
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # For causal language modeling, labels are the same as input_ids
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone()
        }

def load_and_preprocess_data(config: TrainingConfig) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Load and preprocess the dataset, splitting into train/validation/test sets.
    """
    print(f"Loading data from {config.data_path}...")
    
    with open(config.data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} samples")
    
    # Split data into train/validation/test
    train_data, temp_data = train_test_split(
        data, 
        test_size=(config.validation_ratio + config.test_ratio),
        random_state=42
    )
    
    val_size = config.validation_ratio / (config.validation_ratio + config.test_ratio)
    val_data, test_data = train_test_split(
        temp_data,
        test_size=(1 - val_size),
        random_state=42
    )
    
    print(f"Data split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    return train_data, val_data, test_data

def setup_tokenizer(config: TrainingConfig, model_tokenizer: PreTrainedTokenizer) -> PreTrainedTokenizer:
    """
    Setup tokenizer with special tokens for poem generation.
    """
    print("Setting up tokenizer with special tokens...")
    
    # Add special tokens
    special_tokens = [config.topic_start_token, config.topic_end_token]
    
    # Check if tokens already exist
    new_tokens = []
    for token in special_tokens:
        if token not in model_tokenizer.get_vocab():
            new_tokens.append(token)
    
    if new_tokens:
        print(f"Adding new tokens: {new_tokens}")
        model_tokenizer.add_tokens(new_tokens)
    
    # Set pad token if not exists
    if model_tokenizer.pad_token is None:
        model_tokenizer.pad_token = model_tokenizer.eos_token
    
    print(f"Tokenizer vocab size: {len(model_tokenizer)}")
    
    return model_tokenizer

def create_dataloaders(
    train_data: List[Dict], 
    val_data: List[Dict], 
    tokenizer: PreTrainedTokenizer, 
    config: TrainingConfig
) -> Tuple[DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for training and validation.
    """
    print("Creating datasets and dataloaders...")
    
    train_dataset = PoemDataset(train_data, tokenizer, config.max_length)
    val_dataset = PoemDataset(val_data, tokenizer, config.max_length)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=config.use_gpu
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=config.use_gpu
    )
    
    print(f"Created dataloaders: Train={len(train_loader)}, Val={len(val_loader)}")
    
    return train_loader, val_loader

def prepare_input_text(topic: str, config: TrainingConfig) -> str:
    """
    Prepare input text for poem generation.
    """
    return f"{config.topic_start_token}{topic}{config.topic_end_token}\n"

def extract_generated_poem(generated_text: str, prompt: str) -> str:
    """
    Extract the generated poem from the full generated text.
    """
    # Remove the prompt part
    if generated_text.startswith(prompt):
        poem = generated_text[len(prompt):].strip()
    else:
        poem = generated_text.strip()
    
    return poem 