import os
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class TrainingConfig:
    # Model settings
    model_name: str = "skt/kogpt2-base-v2"
    max_length: int = 512
    
    # Training settings
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    warmup_steps: int = 100
    weight_decay: float = 0.01
    
    # A100 GPU optimized settings
    use_mixed_precision: bool = True  # Enable FP16/BF16 for A100
    dataloader_num_workers: int = 4   # Increase for A100
    
    # Fine-tuning strategy
    full_finetuning: bool = True
    trainable_layers: int = 0  # Number of top layers to keep trainable in head-only mode
    
    # LoRA settings
    use_lora: bool = False
    lora_r: int = 16  # LoRA rank
    lora_alpha: int = 32  # LoRA scaling parameter
    lora_dropout: float = 0.1  # LoRA dropout
    lora_target_modules: List[str] = field(default_factory=lambda: ["c_attn", "c_proj"])  # Target modules for LoRA
    
    # Data settings
    train_ratio: float = 0.8
    validation_ratio: float = 0.1
    test_ratio: float = 0.1
    data_path: str = "data/prompt_dataset.json"
    
    # Special tokens
    topic_start_token: str = "<|topic:"
    topic_end_token: str = "|>"
    
    # GPU settings
    use_gpu: bool = True
    device: str = "cuda" if use_gpu else "cpu"
    
    # Output settings
    output_dir: str = "outputs"
    model_save_dir: str = "saved_models"
    logs_dir: str = "logs"
    
    # Evaluation settings
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 500  # Increased from 100 to reduce log frequency
    
    # Generation settings for testing
    max_new_tokens: int = 200
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    do_sample: bool = True
    
    # Logging settings
    quiet_mode: bool = False  # If True, minimal logging
    progress_bar: bool = True  # Show progress bars
    
    def __post_init__(self):
        # Create directories if they don't exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.model_save_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Check GPU availability
        if self.use_gpu:
            import torch
            if not torch.cuda.is_available():
                print("Warning: CUDA not available, switching to CPU")
                self.device = "cpu"
                self.use_gpu = False 