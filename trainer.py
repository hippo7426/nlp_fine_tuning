import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    get_linear_schedule_with_warmup
)
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import os
import time
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import json
import datetime
from dataclasses import asdict

from config import TrainingConfig
from data_utils import setup_tokenizer, create_dataloaders, load_and_preprocess_data

class KoGPT2Trainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize model and tokenizer
        self.tokenizer = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer with special tokens."""
        print(f"Loading model and tokenizer: {self.config.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        # Setup special tokens
        self.tokenizer = setup_tokenizer(self.config, self.tokenizer)
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(self.config.model_name)
        
        # Resize model embeddings to match tokenizer vocab size
        if len(self.tokenizer) != self.model.get_input_embeddings().num_embeddings:
            print(f"Resizing model embeddings from {self.model.get_input_embeddings().num_embeddings} to {len(self.tokenizer)}")
            self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Apply LoRA if enabled
        if self.config.use_lora:
            self._setup_lora()
        else:
            # Configure fine-tuning strategy (only if not using LoRA)
            self._setup_finetuning_strategy()
        
        # Move model to device
        self.model.to(self.device)
        
        print(f"Model loaded and configured. Total parameters: {self.count_parameters():,}")
        
    def _setup_finetuning_strategy(self):
        """Configure full fine-tuning vs head-only fine-tuning."""
        if not self.config.full_finetuning:
            print("Setting up head-only fine-tuning...")
            
            total_layers = len(self.model.transformer.h)
            trainable_layers = self.config.trainable_layers
            frozen_layers = total_layers - trainable_layers
            
            print(f"Total transformer layers: {total_layers}")
            print(f"Trainable layers (top): {trainable_layers}")
            print(f"Frozen layers (bottom): {frozen_layers}")
            
            # Freeze transformer layers (keep only top N layers trainable)
            for name, param in self.model.named_parameters():
                if 'transformer.h' in name:
                    # Extract layer number
                    layer_num = int(name.split('.')[2])
                    if layer_num < frozen_layers:  # Freeze bottom layers
                        param.requires_grad = False
                        
            # Count what's actually trainable
            trainable_transformer_params = 0
            frozen_transformer_params = 0
            for name, param in self.model.named_parameters():
                if 'transformer.h' in name:
                    if param.requires_grad:
                        trainable_transformer_params += param.numel()
                    else:
                        frozen_transformer_params += param.numel()
            
            print(f"Frozen transformer parameters: {frozen_transformer_params:,}")
            print(f"Trainable transformer parameters: {trainable_transformer_params:,}")
                        
        else:
            print("Using full fine-tuning...")
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"All {total_params:,} parameters are trainable")
        
    def _setup_lora(self):
        """Setup LoRA configuration."""
        print("Setting up LoRA fine-tuning...")
        
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias="none",
        )
        
        # Apply LoRA to the model
        self.model = get_peft_model(self.model, lora_config)
        
        # Print LoRA information
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"LoRA configuration:")
        print(f"  - Rank: {self.config.lora_r}")
        print(f"  - Alpha: {self.config.lora_alpha}")
        print(f"  - Dropout: {self.config.lora_dropout}")
        print(f"  - Target modules: {self.config.lora_target_modules}")
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")
        print(f"  - Trainable ratio: {100 * trainable_params / total_params:.2f}%")
        
        # 성능 최적화 팁
        print(f"\n💡 LoRA 최적화 팁:")
        if self.config.lora_r < 32:
            print(f"  ⚠️  낮은 rank({self.config.lora_r}) 감지. 성능 향상을 위해 --lora-r 32 이상 권장")
        if self.config.lora_alpha / self.config.lora_r < 2:
            print(f"  ⚠️  alpha/rank 비율이 낮음. --lora-alpha {self.config.lora_r * 2} 권장")
        if len(self.config.lora_target_modules) < 3:
            print(f"  💡 더 많은 모듈 적용 권장: --lora-target-modules c_attn c_proj c_fc")
        
        print(f"  ✅ 현재 설정으로 약 {100 * trainable_params / total_params:.1f}%의 파라미터 학습")
        
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def setup_optimizer_and_scheduler(self, num_training_steps: int):
        """Setup optimizer and learning rate scheduler."""
        print("Setting up optimizer and scheduler...")
        
        # Get parameters that require gradients
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        
        # LoRA 특별 처리: LoRA 파라미터와 일반 파라미터를 분리
        if self.config.use_lora:
            lora_params = []
            base_params = []
            
            for name, param in param_optimizer:
                if param.requires_grad:
                    if 'lora_' in name:
                        lora_params.append((name, param))
                    else:
                        base_params.append((name, param))
            
            print(f"LoRA parameters: {len(lora_params)}, Base parameters: {len(base_params)}")
            
            # LoRA 파라미터에는 더 높은 학습률 적용
            optimizer_grouped_parameters = [
                {
                    'params': [p for n, p in lora_params if not any(nd in n for nd in no_decay)],
                    'weight_decay': self.config.weight_decay,
                },
                {
                    'params': [p for n, p in lora_params if any(nd in n for nd in no_decay)],
                    'weight_decay': 0.0,
                },
                {
                    'params': [p for n, p in base_params if not any(nd in n for nd in no_decay)],
                    'weight_decay': self.config.weight_decay
                },
                {
                    'params': [p for n, p in base_params if any(nd in n for nd in no_decay)],
                    'weight_decay': 0.0
                }
            ]
        else:
            optimizer_grouped_parameters = [
                {
                    'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                    'weight_decay': self.config.weight_decay
                },
                {
                    'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                    'weight_decay': 0.0
                }
            ]
        
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            eps=1e-8,
            betas=(0.9, 0.95) if self.config.use_lora else (0.9, 0.999)  # LoRA에 최적화된 beta값
        )
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=num_training_steps
        )
        
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        # Configure progress bar based on settings
        if self.config.quiet_mode:
            # Minimal progress bar in quiet mode
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", 
                              disable=False, leave=False,
                              mininterval=30.0, maxinterval=60.0)
        else:
            # Normal progress bar
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", 
                              mininterval=10.0, maxinterval=30.0)
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss / self.config.gradient_accumulation_steps
            loss.backward()
            
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            
            # Gradient accumulation
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                # Log learning rate
                current_lr = self.scheduler.get_last_lr()[0]
                self.learning_rates.append(current_lr)
                
            # Update progress bar only at logging intervals
            if (step + 1) % self.config.logging_steps == 0:
                avg_loss = total_loss / (step + 1)
                progress_bar.set_postfix({
                    'step': f'{step+1}/{len(train_loader)}',
                    'loss': f'{avg_loss:.4f}',
                    'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
                })
                
                # Print periodic log only if not in quiet mode
                if not self.config.quiet_mode:
                    print(f"Step {step+1}/{len(train_loader)} - Loss: {avg_loss:.4f}, LR: {self.scheduler.get_last_lr()[0]:.2e}")
                
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            # Reduce validation logging frequency
            progress_bar = tqdm(val_loader, desc="Validating",
                              mininterval=5.0, maxinterval=15.0)
            
            for step, batch in enumerate(progress_bar):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
                
                # Update progress less frequently
                if (step + 1) % 100 == 0 or step == len(val_loader) - 1:
                    avg_loss = total_loss / (step + 1)
                    progress_bar.set_postfix({'val_loss': f'{avg_loss:.4f}'})
        
        return total_loss / len(val_loader)
    
    def train(self, train_data: List[Dict], val_data: List[Dict]):
        """Main training loop."""
        print("Starting training...")
        
        # Create data loaders
        train_loader, val_loader = create_dataloaders(
            train_data, val_data, self.tokenizer, self.config
        )
        
        # Calculate total training steps
        total_steps = len(train_loader) * self.config.num_epochs // self.config.gradient_accumulation_steps
        
        # Setup optimizer and scheduler
        self.setup_optimizer_and_scheduler(total_steps)
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            print(f"\n=== Epoch {epoch + 1}/{self.config.num_epochs} ===")
            
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model("best_model")
                print(f"New best model saved (val_loss: {val_loss:.4f})")
            
            # Save checkpoint
            if (epoch + 1) % 1 == 0:
                self.save_model(f"checkpoint_epoch_{epoch+1}")
        
        print("Training completed!")
        self.plot_training_history()
        
    def save_model(self, name: str):
        """Save model and tokenizer."""
        save_path = os.path.join(self.config.model_save_dir, name)
        os.makedirs(save_path, exist_ok=True)
        
        # Save model (LoRA or full model)
        if self.config.use_lora:
            # Save LoRA adapter
            self.model.save_pretrained(save_path)
            # Also save base model config
            self.model.base_model.model.config.save_pretrained(save_path)
        else:
            # Save full model
            self.model.save_pretrained(save_path)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_path)
        
        # Save training configuration and arguments
        config_dict = asdict(self.config)
        
        # Add additional training information
        training_info = {
            "training_config": config_dict,
            "command_line_args": getattr(self.config, 'command_line_args', {}),
            "model_info": {
                "model_name": self.config.model_name,
                "total_parameters": self.count_parameters(),
                "vocab_size": len(self.tokenizer),
                "max_length": self.config.max_length,
            },
            "training_history": {
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
                "learning_rates": self.learning_rates[-10:] if self.learning_rates else [],  # Last 10 LR values
                "final_train_loss": self.train_losses[-1] if self.train_losses else None,
                "final_val_loss": self.val_losses[-1] if self.val_losses else None,
                "best_val_loss": min(self.val_losses) if self.val_losses else None,
            },
            "timestamp": {
                "saved_at": datetime.datetime.now().isoformat(),
                "model_type": name,
            }
        }
        
        # Save training arguments
        training_args_path = os.path.join(save_path, "training_args.json")
        with open(training_args_path, 'w', encoding='utf-8') as f:
            json.dump(training_info, f, indent=2, ensure_ascii=False)
        
        print(f"Model saved to {save_path}")
        print(f"Training arguments saved to {training_args_path}")
    
    def load_model(self, path: str):
        """Load saved model."""
        print(f"Loading model from {path}")
        
        # Load tokenizer first
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        
        # Re-setup special tokens to ensure consistency
        from data_utils import setup_tokenizer
        self.tokenizer = setup_tokenizer(self.config, self.tokenizer)
        
        # Check if it's a LoRA model by checking for adapter files
        adapter_config_path = os.path.join(path, "adapter_config.json")
        
        if os.path.exists(adapter_config_path):
            # Load LoRA model
            print("Loading LoRA model...")
            # First load base model
            base_model = AutoModelForCausalLM.from_pretrained(self.config.model_name)
            
            # Resize base model embeddings if needed
            if len(self.tokenizer) != base_model.get_input_embeddings().num_embeddings:
                print(f"Resizing base model embeddings from {base_model.get_input_embeddings().num_embeddings} to {len(self.tokenizer)}")
                base_model.resize_token_embeddings(len(self.tokenizer))
            
            # Then load LoRA adapter
            self.model = PeftModel.from_pretrained(base_model, path)
        else:
            # Load full model
            print("Loading full model...")
            self.model = AutoModelForCausalLM.from_pretrained(path)
            
            # Check and resize embeddings if needed
            if len(self.tokenizer) != self.model.get_input_embeddings().num_embeddings:
                print(f"Resizing model embeddings from {self.model.get_input_embeddings().num_embeddings} to {len(self.tokenizer)}")
                self.model.resize_token_embeddings(len(self.tokenizer))
        
        self.model.to(self.device)
        
        # Verify tokenizer and model compatibility
        print(f"✅ Model and tokenizer loaded successfully")
        print(f"   Tokenizer vocab size: {len(self.tokenizer)}")
        print(f"   Model embedding size: {self.model.get_input_embeddings().num_embeddings}")
        
        # Check special tokens
        print(f"   Special tokens: {self.tokenizer.special_tokens_map}")
        if hasattr(self.tokenizer, 'added_tokens_encoder'):
            print(f"   Added tokens: {list(self.tokenizer.added_tokens_encoder.keys())}")
    
    def plot_training_history(self):
        """Plot training and validation losses."""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.learning_rates)
        plt.xlabel('Step')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{self.config.output_dir}/training_history.png", dpi=300, bbox_inches='tight')
        plt.show()
        
    def generate_poem(self, topic: str) -> str:
        """Generate a poem for the given topic."""
        self.model.eval()
        
        try:
            # Prepare input
            from data_utils import prepare_input_text, extract_generated_poem
            prompt = prepare_input_text(topic, self.config)
            
            print(f"🎯 Generating poem for topic: '{topic}'")
            print(f"📝 Prompt: '{prompt.strip()}'")
            
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors='pt',
                truncation=True,
                max_length=self.config.max_length
            ).to(self.device)
            
            print(f"🔧 Input tokens: {inputs['input_ids'].shape}")
            print(f"🔧 Device: {self.device}")
            
            # Generate
            with torch.no_grad():
                # LoRA 모델의 경우 특별한 처리
                if hasattr(self.model, 'base_model'):
                    # LoRA model
                    print("🎛️ Using LoRA model for generation")
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.config.max_new_tokens,
                        temperature=self.config.temperature,
                        top_k=self.config.top_k,
                        top_p=self.config.top_p,
                        do_sample=self.config.do_sample,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        repetition_penalty=1.1  # LoRA에서 반복 방지
                    )
                else:
                    # Regular model
                    print("🤖 Using full model for generation")
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.config.max_new_tokens,
                        temperature=self.config.temperature,
                        top_k=self.config.top_k,
                        top_p=self.config.top_p,
                        do_sample=self.config.do_sample,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
            
            print(f"✅ Generation completed. Output tokens: {outputs.shape}")
            
            # Decode and extract poem
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
            poem = extract_generated_poem(generated_text, prompt)
            
            print(f"📖 Generated text length: {len(generated_text)} chars")
            print(f"🎭 Extracted poem length: {len(poem)} chars")
            
            return poem
            
        except Exception as e:
            print(f"❌ Error during poem generation: {e}")
            print(f"   Model type: {type(self.model)}")
            print(f"   Tokenizer vocab size: {len(self.tokenizer)}")
            print(f"   Model embedding size: {self.model.get_input_embeddings().num_embeddings}")
            
            # 간단한 fallback 생성
            fallback_poem = f"죄송합니다. '{topic}' 주제로 시를 생성하는 중 오류가 발생했습니다.\n오류: {str(e)}"
            return fallback_poem 