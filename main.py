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
    
    # LoRA settings
    parser.add_argument('--lora', action='store_true', default=False,
                      help='Use LoRA (Low-Rank Adaptation) fine-tuning')
    parser.add_argument('--lora-r', type=int, default=32,
                      help='LoRA rank (default: 32)')
    parser.add_argument('--lora-alpha', type=int, default=64,
                      help='LoRA scaling parameter (default: 64)')
    parser.add_argument('--lora-dropout', type=float, default=0.05,
                      help='LoRA dropout rate (default: 0.05)')
    parser.add_argument('--lora-target-modules', nargs='+', default=['c_attn', 'c_proj', 'c_fc'],
                      help='Target modules for LoRA (default: c_attn c_proj c_fc)')
    
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
    
    # LoRA settings
    config.use_lora = args.lora
    config.lora_r = args.lora_r
    config.lora_alpha = args.lora_alpha
    config.lora_dropout = args.lora_dropout
    config.lora_target_modules = args.lora_target_modules
    
    # LoRA optimization: 자동으로 학습률 조정
    if config.use_lora:
        # LoRA는 일반적으로 더 높은 학습률이 필요함
        if args.lr == 5e-5:  # 기본값인 경우에만 자동 조정
            config.learning_rate = 1e-4  # 2배 증가
            print(f"🎯 LoRA 최적화: 학습률을 {config.learning_rate:.0e}로 자동 증가")
        
        # warmup 단계도 늘려서 안정적인 학습
        config.warmup_steps = 200  # 100 -> 200으로 증가
        print(f"🎯 LoRA 최적화: warmup steps를 {config.warmup_steps}로 증가")
    
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
        if config.use_lora:
            config.batch_size = 64  # LoRA는 메모리를 적게 사용하므로 더 큰 배치 사이즈 가능
            print(f"   - LoRA + A100: Batch size increased to {config.batch_size}")
            # LoRA + A100 조합에서는 더 적극적인 학습률 적용
            if args.lr == 5e-5:  # 기본값인 경우
                config.learning_rate = 2e-4  # 더 높은 학습률
                print(f"   - LoRA + A100: Learning rate increased to {config.learning_rate:.0e}")
        else:
            config.batch_size = 16  # Increase batch size for A100
            print(f"   - Batch size increased to {config.batch_size}")
        config.use_mixed_precision = True
        config.dataloader_num_workers = 4
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
    print(f"- Use LoRA: {config.use_lora}")
    if config.use_lora:
        print(f"- LoRA rank: {config.lora_r}")
        print(f"- LoRA alpha: {config.lora_alpha}")
        print(f"- LoRA dropout: {config.lora_dropout}")
        print(f"- LoRA target modules: {config.lora_target_modules}")
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