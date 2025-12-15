#!/usr/bin/env python3
"""Minimal test to isolate bus error"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

print("Step 1: Importing libraries...")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import AutoModelForCausalLMWithValueHead
from datasets import load_dataset
print("✓ Imports successful")

print("\nStep 2: Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print("✓ Tokenizer loaded")

print("\nStep 3: Loading model...")
model = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2")
print("✓ Model loaded")

print("\nStep 4: Loading dataset (small sample)...")
dataset = load_dataset("Anthropic/hh-rlhf", split="train[:5]")
print(f"✓ Dataset loaded: {len(dataset)} items")

print("\nStep 5: Testing generation...")
query = "Human: Hello\nAssistant:"
query_tokens = tokenizer(query, return_tensors="pt", max_length=64, truncation=True)
print("✓ Tokenization successful")

print("\nStep 6: Generating response...")
try:
    # Try without use_cache parameter
    with torch.no_grad():
        outputs = model.pretrained_model.generate(
            **query_tokens,
            max_new_tokens=10,
            do_sample=False,  # Greedy decoding
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"✓ Generation successful: {response[:50]}...")
except Exception as e:
    print(f"✗ Generation failed: {e}")
    print("Trying alternative approach...")
    # Try with base model directly
    base_model = AutoModelForCausalLM.from_pretrained("gpt2")
    with torch.no_grad():
        outputs = base_model.generate(
            **query_tokens,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"✓ Alternative generation successful: {response[:50]}...")

print("\n✅ All tests passed! The issue is likely in the training loop.")

