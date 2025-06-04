#!/usr/bin/env python3
"""
Simple script to test dataset structure
"""

import json
import sys

def test_dataset():
    try:
        print("Loading dataset...")
        with open('data/prompt_dataset.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Total samples: {len(data)}")
        print(f"First sample keys: {list(data[0].keys())}")
        print(f"First sample prompt: '{data[0]['prompt'][:50]}...'")
        print(f"First sample poem: '{data[0]['poem'][:100]}...'")
        
        # Check data consistency
        valid_samples = 0
        for i, sample in enumerate(data[:100]):  # Check first 100 samples
            if 'prompt' in sample and 'poem' in sample:
                if sample['prompt'] and sample['poem']:
                    valid_samples += 1
        
        print(f"Valid samples in first 100: {valid_samples}/100")
        print("Dataset test completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_dataset() 