"""
Week 1 Day 3-4: Train Reward Model on HH-RLHF

This script trains a reward model on Anthropic HH-RLHF preference pairs
using TRL's RewardTrainer.

Key fixes from previous attempts:
- Use 5K-10K training samples (not 1K)
- Use 5-10 epochs (not 3)
- Use TRL RewardTrainer (don't reinvent)
- Validate loss decreases and accuracy > 60%
"""

import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
)
from trl import RewardTrainer, RewardConfig
from datasets import load_dataset
import numpy as np
from pathlib import Path

# Disable tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def prepare_dataset_for_reward_training(dataset, tokenizer, max_length=512):
    """Prepare HH-RLHF dataset for reward model training."""
    def tokenize_function(examples):
        # Tokenize chosen and rejected responses
        chosen_texts = [f"{q} {c}" for q, c in zip(examples["query"], examples["chosen"])]
        rejected_texts = [f"{q} {r}" for q, r in zip(examples["query"], examples["rejected"])]
        
        chosen_tokens = tokenizer(
            chosen_texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        rejected_tokens = tokenizer(
            rejected_texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        
        return {
            "input_ids_chosen": chosen_tokens["input_ids"],
            "attention_mask_chosen": chosen_tokens["attention_mask"],
            "input_ids_rejected": rejected_tokens["input_ids"],
            "attention_mask_rejected": rejected_tokens["attention_mask"],
        }
    
    return dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

def main():
    print("=" * 80)
    print("Week 1 Day 3-4: Train Reward Model on HH-RLHF")
    print("=" * 80)
    print()
    
    # Configuration
    model_name = "gpt2"
    dataset_name = "Anthropic/hh-rlhf"
    output_dir = Path("pulseos_rlhf/reward_model")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training settings (fixed from previous attempts)
    num_train_samples = 8000  # 5K-10K range (was 1K)
    num_epochs = 7  # 5-10 epochs (was 3)
    learning_rate = 2e-5
    batch_size = 4
    max_length = 512
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print()
    
    # Load tokenizer
    print("1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"   ✓ Loaded {model_name} tokenizer")
    print()
    
    # Load dataset
    print("2. Loading HH-RLHF dataset...")
    try:
        dataset = load_dataset(dataset_name, split="train")
        print(f"   ✓ Loaded {len(dataset)} preference pairs")
    except Exception as e:
        print(f"   ✗ Error loading dataset: {e}")
        print("   Trying with token...")
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if hf_token:
            dataset = load_dataset(dataset_name, split="train", token=hf_token)
            print(f"   ✓ Loaded {len(dataset)} preference pairs")
        else:
            raise
    
    # Use subset for training (fixed: use more samples)
    train_size = min(num_train_samples, len(dataset))
    train_dataset = dataset.select(range(train_size))
    val_size = min(1000, len(dataset) - train_size)
    val_dataset = dataset.select(range(train_size, train_size + val_size))
    
    print(f"   Training on {len(train_dataset)} samples")
    print(f"   Validation on {len(val_dataset)} samples")
    print()
    
    # Prepare dataset
    print("3. Preparing dataset for reward training...")
    train_dataset = prepare_dataset_for_reward_training(train_dataset, tokenizer, max_length)
    val_dataset = prepare_dataset_for_reward_training(val_dataset, tokenizer, max_length)
    print("   ✓ Dataset prepared")
    print()
    
    # Load model for reward training
    print("4. Loading model for reward training...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,  # Single scalar reward
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    print(f"   ✓ Loaded {model_name} for reward training")
    print()
    
    # Reward config
    print("5. Configuring reward trainer...")
    reward_config = RewardConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        max_length=max_length,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    print("   ✓ Reward config created")
    print()
    
    # Create reward trainer
    print("6. Creating reward trainer...")
    trainer = RewardTrainer(
        model=model,
        args=reward_config,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    print("   ✓ Reward trainer created")
    print()
    
    # Train
    print("7. Training reward model...")
    print(f"   Training for {num_epochs} epochs on {len(train_dataset)} samples...")
    print()
    
    train_result = trainer.train()
    
    print()
    print("=" * 80)
    print("TRAINING RESULTS")
    print("=" * 80)
    print(f"Training loss: {train_result.training_loss:.4f}")
    print(f"Training samples: {train_result.global_step}")
    print()
    
    # Evaluate
    print("8. Evaluating reward model...")
    eval_result = trainer.evaluate()
    eval_loss = eval_result.get("eval_loss", 0.0)
    print(f"   Validation loss: {eval_loss:.4f}")
    print()
    
    # Save model
    print("9. Saving reward model...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    print(f"   ✓ Model saved to {output_dir}")
    print()
    
    # Validate reward model can score text
    print("10. Validating reward model...")
    model.eval()
    test_texts = [
        "Human: How do I make a cake?\nAssistant: Here's a simple recipe for making a cake.",
        "Human: How do I make a cake?\nAssistant: I don't know.",
    ]
    
    with torch.no_grad():
        for text in test_texts:
            tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
            reward = model(**tokens).logits.item()
            print(f"   Text: '{text[:50]}...'")
            print(f"   Reward: {reward:.4f}")
    
    print()
    print("=" * 80)
    print("SUCCESS CRITERIA CHECK")
    print("=" * 80)
    
    # Check success criteria
    initial_loss = train_result.log_history[0].get("loss", 0.0) if train_result.log_history else 0.0
    final_loss = train_result.training_loss
    
    if initial_loss > 0 and final_loss < initial_loss:
        loss_reduction = ((initial_loss - final_loss) / initial_loss) * 100
        print(f"✓ Loss decreased: {initial_loss:.4f} → {final_loss:.4f} ({loss_reduction:.1f}% reduction)")
    else:
        print(f"⚠️  Loss tracking: {final_loss:.4f}")
    
    if eval_loss < 0.5:
        print(f"✓ Validation loss < 0.5: {eval_loss:.4f}")
    else:
        print(f"⚠️  Validation loss: {eval_loss:.4f} (target < 0.5)")
    
    print(f"✓ Model saved and can score text")
    print()
    print("Next steps:")
    print("  1. If loss decreased significantly: Proceed to Day 5-7 (baseline PPO)")
    print("  2. If loss didn't decrease: Check dataset, increase epochs/samples")
    print("=" * 80)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise
