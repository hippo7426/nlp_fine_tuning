#!/usr/bin/env python3
"""
KoGPT2 Korean Poetry Generation Fine-tuning Script

This script fine-tunes a KoGPT2 model for Korean poetry generation.
Usage:
    python main.py [--full-finetuning] [--gpu] [--epochs 3] [--lr 5e-5]
"""

import argparse
import sys
import os
import torch
from config import TrainingConfig
from trainer import KoGPT2Trainer
from data_utils import load_and_preprocess_data
from evaluation import evaluate_model, save_evaluation_results

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='KoGPT2 Poetry Fine-tuning')
    
    # Training strategy
    parser.add_argument('--full-finetuning', action='store_true', default=True,
                      help='Use full fine-tuning (default: True)')
    parser.add_argument('--head-only', action='store_true', default=False,
                      help='Use head-only fine-tuning')
    parser.add_argument('--trainable-layers', type=int, default=0,
                      help='Number of top transformer layers to keep trainable in head-only mode (default: 0)')
    
    # Hardware
    parser.add_argument('--gpu', action='store_true', default=True,
                      help='Use GPU if available (default: True)')
    parser.add_argument('--cpu', action='store_true', default=False,
                      help='Force CPU usage')
    
    # A100 optimization
    parser.add_argument('--a100-optimized', action='store_true', default=False,
                      help='Use A100 optimized settings (larger batch size, mixed precision)')
    parser.add_argument('--mixed-precision', action='store_true', default=False,
                      help='Enable mixed precision training')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=3,
                      help='Number of training epochs (default: 3)')
    parser.add_argument('--lr', type=float, default=5e-5,
                      help='Learning rate (default: 5e-5)')
    parser.add_argument('--batch-size', type=int, default=4,
                      help='Batch size (default: 4)')
    parser.add_argument('--max-length', type=int, default=512,
                      help='Maximum sequence length (default: 512)')
    
    # Logging
    parser.add_argument('--quiet', action='store_true', default=False,
                      help='Quiet mode with minimal logging')
    
    # Model
    parser.add_argument('--model-name', type=str, default='skt/kogpt2-base-v2',
                      help='Model name or path (default: skt/kogpt2-base-v2)')
    
    # Mode
    parser.add_argument('--mode', type=str, default='train', 
                      choices=['train', 'evaluate', 'generate', 'all'],
                      help='Mode: train, evaluate, generate, or all (default: all)')
    parser.add_argument('--model-path', type=str, default=None,
                      help='Path to saved model for evaluation or generation')
    
    return parser.parse_args()

def main():
    """Main function."""
    print("=== KoGPT2 Korean Poetry Fine-tuning ===\n")
    
    # Parse arguments
    args = parse_arguments()
    
    # Create configuration
    config = TrainingConfig()
    
    # Update config based on arguments
    if args.head_only:
        config.full_finetuning = False
    else:
        config.full_finetuning = args.full_finetuning
    
    config.trainable_layers = args.trainable_layers
    
    if args.cpu:
        config.use_gpu = False
        config.device = 'cpu'
    else:
        config.use_gpu = args.gpu
        
    config.num_epochs = args.epochs
    config.learning_rate = args.lr
    config.batch_size = args.batch_size
    config.max_length = args.max_length
    config.model_name = args.model_name
    
    # Logging settings
    config.quiet_mode = args.quiet
    
    # A100 optimization settings
    if args.a100_optimized:
        print("🚀 A100 optimization enabled!")
        config.batch_size = 16  # Increase batch size for A100
        config.use_mixed_precision = True
        config.dataloader_num_workers = 4
        print(f"   - Batch size increased to {config.batch_size}")
        print(f"   - Mixed precision enabled")
        print(f"   - DataLoader workers: {config.dataloader_num_workers}")
    
    if args.mixed_precision:
        config.use_mixed_precision = True
    
    # Store command line arguments in config for saving
    config.command_line_args = vars(args)
    
    # Print configuration
    print("Configuration:")
    print(f"- Model: {config.model_name}")
    print(f"- Device: {config.device}")
    print(f"- Full fine-tuning: {config.full_finetuning}")
    print(f"- Epochs: {config.num_epochs}")
    print(f"- Learning rate: {config.learning_rate}")
    print(f"- Batch size: {config.batch_size}")
    print(f"- Max length: {config.max_length}")
    print()
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    train_data, val_data, test_data = load_and_preprocess_data(config)
    
    # Initialize trainer
    trainer = KoGPT2Trainer(config)
    
    if args.mode in ['train', 'all']:
        print("\n=== Starting Training ===")
        
        # Setup model and tokenizer
        trainer.setup_model_and_tokenizer()
        
        # Train the model
        trainer.train(train_data, val_data)
        
        print("Training completed!")
        
    if args.mode in ['evaluate', 'all']:
        print("\n=== Starting Evaluation ===")
        
        # Load model if path is provided
        if args.model_path:
            trainer.load_model(args.model_path)
        elif not hasattr(trainer, 'model') or trainer.model is None:
            # Load best model from training
            best_model_path = os.path.join(config.model_save_dir, "best_model")
            if os.path.exists(best_model_path):
                trainer.load_model(best_model_path)
            else:
                print("No trained model found. Please run training first or provide --model-path")
                return
        
        # Evaluate model
        results = evaluate_model(trainer.model, trainer.tokenizer, test_data, config)
        
        # Save results
        save_evaluation_results(results, config)
        
        print("Evaluation completed!")
        
    if args.mode in ['generate', 'all']:
        print("\n=== Testing Poetry Generation ===")
        
        # Load model if path is provided
        if args.model_path:
            trainer.load_model(args.model_path)
        elif not hasattr(trainer, 'model') or trainer.model is None:
            # Load best model from training
            best_model_path = os.path.join(config.model_save_dir, "best_model")
            if os.path.exists(best_model_path):
                trainer.load_model(best_model_path)
            else:
                print("No trained model found. Please run training first or provide --model-path")
                return
        
        # Test poem generation
        test_topics = ["자연", "사랑", "그리움", "가을", "달빛"]
        
        print("Generated poems:")
        print("=" * 50)
        
        for topic in test_topics:
            print(f"\nTopic: {topic}")
            print("-" * 30)
            try:
                poem = trainer.generate_poem(topic)
                print(poem)
            except Exception as e:
                print(f"Error generating poem: {e}")
            print("-" * 30)
        
        # Interactive generation
        print("\n=== Interactive Generation ===")
        print("Enter topics to generate poems (type 'quit' to exit):")
        
        while True:
            try:
                topic = input("\nEnter topic: ").strip()
                if topic.lower() in ['quit', 'exit', 'q']:
                    break
                
                if topic:
                    print(f"\nGenerating poem for '{topic}'...")
                    poem = trainer.generate_poem(topic)
                    print("\nGenerated poem:")
                    print("-" * 30)
                    print(poem)
                    print("-" * 30)
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
    
    print("\nAll tasks completed!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1) 