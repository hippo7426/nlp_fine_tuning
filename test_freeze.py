#!/usr/bin/env python3
"""
Test script to understand freeze logic
"""

def test_freeze_logic():
    """Test the freeze logic with different trainable_layers values"""
    
    total_layers = 12  # KoGPT2 has 12 layers (0-11)
    
    print("=== Head-only Fine-tuning Freeze Logic ===\n")
    
    for trainable_layers in [0, 2, 4, 6]:
        frozen_layers = total_layers - trainable_layers
        
        print(f"📋 trainable_layers = {trainable_layers}")
        print(f"   - Total layers: {total_layers} (layers 0-{total_layers-1})")
        print(f"   - Frozen layers: {frozen_layers} (layers 0-{frozen_layers-1})")
        print(f"   - Trainable layers: {trainable_layers} (layers {frozen_layers}-{total_layers-1})")
        
        # Show which layers are frozen/trainable
        frozen = []
        trainable = []
        
        for layer_num in range(total_layers):
            if layer_num < frozen_layers:
                frozen.append(layer_num)
            else:
                trainable.append(layer_num)
        
        print(f"   - Frozen: {frozen}")
        print(f"   - Trainable: {trainable}")
        print(f"   - Plus: embedding, positional encoding, layer norm, LM head")
        print()

if __name__ == "__main__":
    test_freeze_logic() 