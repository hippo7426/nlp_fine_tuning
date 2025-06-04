import torch
import torch.nn.functional as F
import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizer
from bert_score import score
from typing import List, Dict, Tuple
import math
from config import TrainingConfig
from data_utils import prepare_input_text, extract_generated_poem

def calculate_perplexity(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, 
                        test_data: List[Dict], config: TrainingConfig) -> float:
    """
    Calculate perplexity on test data.
    """
    print("Calculating perplexity...")
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for item in test_data:
            full_text = item['prompt'] + item['poem']
            
            # Tokenize
            inputs = tokenizer(
                full_text,
                return_tensors='pt',
                truncation=True,
                max_length=config.max_length
            ).to(config.device)
            
            # Forward pass
            outputs = model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss
            
            # Calculate number of tokens (excluding padding)
            num_tokens = (inputs['attention_mask'] == 1).sum().item()
            
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    print(f"Perplexity: {perplexity:.2f}")
    return perplexity

def calculate_bert_score(generated_poems: List[str], reference_poems: List[str]) -> Dict[str, float]:
    """
    Calculate BERTScore between generated and reference poems.
    """
    print("Calculating BERTScore...")
    
    if len(generated_poems) != len(reference_poems):
        raise ValueError("Number of generated and reference poems must match")
    
    # Calculate BERTScore
    P, R, F1 = score(generated_poems, reference_poems, lang='ko', verbose=False)
    
    bert_scores = {
        'precision': P.mean().item(),
        'recall': R.mean().item(),
        'f1': F1.mean().item()
    }
    
    print(f"BERTScore - Precision: {bert_scores['precision']:.3f}, "
          f"Recall: {bert_scores['recall']:.3f}, F1: {bert_scores['f1']:.3f}")
    
    return bert_scores

def generate_poems_for_evaluation(model: PreTrainedModel, tokenizer: PreTrainedTokenizer,
                                test_data: List[Dict], config: TrainingConfig, 
                                num_samples: int = 50) -> Tuple[List[str], List[str]]:
    """
    Generate poems for evaluation and return generated and reference poems.
    """
    print(f"Generating {num_samples} poems for evaluation...")
    
    model.eval()
    generated_poems = []
    reference_poems = []
    
    # Select random samples for evaluation
    import random
    selected_data = random.sample(test_data, min(num_samples, len(test_data)))
    
    with torch.no_grad():
        for item in selected_data:
            # Extract topic from prompt
            prompt = item['prompt']
            reference_poem = item['poem']
            
            # Tokenize input
            inputs = tokenizer(
                prompt,
                return_tensors='pt',
                truncation=True,
                max_length=config.max_length
            ).to(config.device)
            
            # Generate
            outputs = model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_k=config.top_k,
                top_p=config.top_p,
                do_sample=config.do_sample,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            # Decode generated text
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
            generated_poem = extract_generated_poem(generated_text, prompt)
            
            generated_poems.append(generated_poem)
            reference_poems.append(reference_poem)
    
    return generated_poems, reference_poems

def evaluate_model(model: PreTrainedModel, tokenizer: PreTrainedTokenizer,
                  test_data: List[Dict], config: TrainingConfig) -> Dict[str, float]:
    """
    Comprehensive model evaluation including perplexity and BERTScore.
    """
    print("Starting comprehensive evaluation...")
    
    results = {}
    
    # Calculate perplexity
    perplexity = calculate_perplexity(model, tokenizer, test_data, config)
    results['perplexity'] = perplexity
    
    # Generate poems and calculate BERTScore
    generated_poems, reference_poems = generate_poems_for_evaluation(
        model, tokenizer, test_data, config
    )
    
    bert_scores = calculate_bert_score(generated_poems, reference_poems)
    results.update(bert_scores)
    
    # Save some example generations
    print("\n=== Example Generations ===")
    for i in range(min(3, len(generated_poems))):
        print(f"\nExample {i+1}:")
        print(f"Generated: {generated_poems[i][:100]}...")
        print(f"Reference: {reference_poems[i][:100]}...")
    
    return results

def save_evaluation_results(results: Dict[str, float], config: TrainingConfig, 
                          filename: str = "evaluation_results.txt"):
    """
    Save evaluation results to a file.
    """
    filepath = f"{config.output_dir}/{filename}"
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=== Model Evaluation Results ===\n\n")
        for metric, value in results.items():
            f.write(f"{metric.capitalize()}: {value:.4f}\n")
        
        f.write(f"\nConfiguration used:\n")
        f.write(f"- Model: {config.model_name}\n")
        f.write(f"- Max length: {config.max_length}\n")
        f.write(f"- Full fine-tuning: {config.full_finetuning}\n")
        f.write(f"- Learning rate: {config.learning_rate}\n")
        f.write(f"- Epochs: {config.num_epochs}\n")
    
    print(f"Evaluation results saved to {filepath}") 