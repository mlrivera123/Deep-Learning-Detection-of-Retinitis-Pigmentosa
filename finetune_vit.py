from transformers import ViTForImageClassification, ViTFeatureExtractor, ViTImageProcessor, TrainingArguments, Trainer
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import numpy as np
import json
import argparse
import random

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory")
parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory")
parser.add_argument("--seed", type=int, default=1337, help="Random seed for reproducibility")
args = parser.parse_args()

DATA_DIR = args.data_dir
OUTPUT_DIR = args.output_dir

# Set seeds for reproducibility
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

dataset = load_dataset(
    "imagefolder",
    data_dir=DATA_DIR,
)

#Extract number of classes and labels from dataset
num_classes = len(dataset['train'].features['label'].names)
labels = dataset['train'].features['label'].names
print(f"Number of classes: {num_classes}")
print(f"Labels: {labels}")

# Create id2label and label2id mappings which make the results more interpretable
id2label = {i: label for i, label in enumerate(dataset["train"].features["label"].names)}
label2id = {label: i for i, label in id2label.items()}
print(f"id2label: {id2label}")
print(f"label2id: {label2id}")

model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=num_classes,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)

def preprocess_image(examples):
    # ViT expects 3 channel RGB images
    examples["image"] = [img.convert("RGB") for img in examples["image"]]
    # convert images to tensor format

    inputs = image_processor(examples["image"], return_tensors="pt")

    # Add labels for training
    inputs["labels"] = examples["label"]
    return inputs

preprocessed_dataset = dataset.with_transform(preprocess_image)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    
    # Get probabilities for AUC (apply softmax if predictions are logits)
    probabilities = torch.nn.functional.softmax(torch.tensor(predictions), dim=1).numpy()
    
    # Get class predictions for other metrics
    pred_classes = np.argmax(predictions, axis=1)
    
    # Compute standard metrics
    accuracy = accuracy_score(labels, pred_classes)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, pred_classes, average='weighted')
    
    # Compute AUC (for binary classification, use probability of positive class)
    if probabilities.shape[1] == 2:
        auc = roc_auc_score(labels, probabilities[:, 1])
    else:
        # For multi-class, use one-vs-rest
        auc = roc_auc_score(labels, probabilities, multi_class='ovr', average='weighted')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
    }


training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=30,
    learning_rate=5e-05,
    optim="adamw_torch",
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-08,
    lr_scheduler_type="linear",
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=5,
    load_best_model_at_end=True,
    metric_for_best_model="auc",
    greater_is_better=True,
    save_total_limit=3,
    remove_unused_columns=False,
    report_to="none",
    seed=seed,
)




# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=preprocessed_dataset["train"],
    eval_dataset=preprocessed_dataset["validation"],
    compute_metrics=compute_metrics,
    tokenizer=image_processor,  # Save preprocessing config with model
)

# Train the model
print("Starting training...")
train_results = trainer.train()

# Save training metrics
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)

# Evaluate on validation set and save predictions for AUC calculation
print("\nEvaluating on validation set...")
val_results = trainer.evaluate(preprocessed_dataset["validation"])
trainer.log_metrics("eval", val_results)
trainer.save_metrics("eval", val_results)

# Get predictions and save labels.npy and probabilities.npy
print("\nSaving predictions for AUC calculation...")
predictions = trainer.predict(preprocessed_dataset["validation"])
probabilities = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=1).numpy()
labels = predictions.label_ids

# Save for read_prob.py compatibility
np.save(f"{OUTPUT_DIR}/probabilities.npy", probabilities)
np.save(f"{OUTPUT_DIR}/labels.npy", labels)
print(f"Saved probabilities.npy and labels.npy to {OUTPUT_DIR}")

# Print validation results
print("\nValidation Results:")
for key, value in val_results.items():
    print(f"  {key}: {value:.4f}")

# Save the final model and metrics to the same output directory
print("\nSaving model...")
trainer.save_model(OUTPUT_DIR)
image_processor.save_pretrained(OUTPUT_DIR)

# Save all metrics and hyperparameters to a custom JSON file in the same directory
all_metrics = {
    "train": train_results.metrics,
    "validation": val_results,
    "hyperparameters": {
        "learning_rate": training_args.learning_rate,
        "weight_decay": training_args.weight_decay,
        "warmup_ratio": training_args.warmup_ratio,
        "label_smoothing_factor": training_args.label_smoothing_factor,
        "num_train_epochs": training_args.num_train_epochs,
        "per_device_train_batch_size": training_args.per_device_train_batch_size,
        "lr_scheduler_type": str(training_args.lr_scheduler_type),
        "max_grad_norm": training_args.max_grad_norm,
        "seed": training_args.seed,
    },
}

with open(f"{OUTPUT_DIR}/all_metrics.json", "w") as f:
    json.dump(all_metrics, f, indent=2)

print(f"\nTraining complete! Model and metrics saved to {OUTPUT_DIR}")