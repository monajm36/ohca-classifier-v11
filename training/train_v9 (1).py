import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,  # Use standard model!
    Trainer,
    TrainingArguments,
    EvalPrediction,
    set_seed,
)


# =============================================================================
# CONFIG
# =============================================================================

class Config:
    MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
    DATA_PATH = "/Users/monamoukaddem/Desktop/mimic_labelled.xlsx"
    OUTPUT_DIR = "/Users/monamoukaddem/Desktop/ohca_classifier_v9"  # New version!

    MAX_LEN = 512
    NUM_LABELS = 2  # Binary: 0=Non-OHCA, 1=OHCA
    BATCH_SIZE = 4
    GRAD_ACCUM = 2
    LR = 2e-5
    NUM_EPOCHS = 5
    SEED = 42
    
    # Weight decay for regularization
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1

    # Sections to extract
    SECTIONS = ["chief_complaint", "hpi", "ems", "ed_course", "impression"]

    DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


config = Config()
set_seed(config.SEED)
os.makedirs(config.OUTPUT_DIR, exist_ok=True)


# =============================================================================
# HELPER: SECTION EXTRACTION
# =============================================================================

import re
from typing import Optional

SECTION_PATTERNS = {
    "chief_complaint": [
        r"chief complaint[\s:]*",
        r"micu indication/[\s]*chief complaint[\s:]*",
    ],
    "hpi": [
        r"history of present illness[\s:]*",
        r"hpi[\s:]*",
    ],
    "ems": [
        r"ems[\s]+narrative[\s:]*",
        r"pre[-\s]*hospital[\s]+course[\s:]*",
    ],
    "ed_course": [
        r"ed course[\s:]*",
        r"emergency department course[\s:]*",
    ],
    "impression": [
        r"impression[\s:]*",
        r"assessment[\s/&]*plan[\s:]*",
    ],
}


def normalize_text(txt: str) -> str:
    """Basic text cleaning while preserving clinical content"""
    txt = txt.replace("\r", "\n")
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n+", "\n", txt)
    return txt.strip()


def extract_section(note: str, section_name: str) -> Optional[str]:
    """
    Simple heuristic section extractor:
    - Find the first occurrence of any pattern for that section
    - Cut text from there until the next ALL-CAPS header or end
    """
    if not isinstance(note, str) or note.strip() == "":
        return None

    txt = normalize_text(note)
    patterns = SECTION_PATTERNS.get(section_name, [])
    if not patterns:
        return None

    start_idx = None
    for pat in patterns:
        m = re.search(pat, txt, flags=re.IGNORECASE)
        if m:
            start_idx = m.start()
            break

    if start_idx is None:
        return None

    # From start_idx forward, cut at next "header" line
    sub = txt[start_idx:]

    # Header heuristic: line with many uppercase + colons, etc.
    header_matches = list(
        re.finditer(r"\n[A-Z0-9 /,&\-]{5,}:\s*", sub)
    )
    if len(header_matches) > 0:
        end_idx = header_matches[0].start()
        section_text = sub[:end_idx]
    else:
        section_text = sub

    return section_text.strip()


def extract_all_sections(df: pd.DataFrame, text_col: str = "clean_text") -> pd.DataFrame:
    """Extract all configured sections and combine them"""
    print("\nExtracting sections from '{}'...".format(text_col))
    print("   Sections to extract:", config.SECTIONS)

    for sec in config.SECTIONS:
        df[sec] = df[text_col].apply(lambda x: extract_section(x, sec))

        n_non_null = df[sec].notnull().sum()
        print(f"   {sec}: {n_non_null}/{len(df)} notes ({100*n_non_null/len(df):.1f}%)")

    # Combine sections into a single text field
    def combine_sections(row):
        pieces = []
        for sec in config.SECTIONS:
            val = row.get(sec, None)
            if isinstance(val, str) and val.strip():
                pieces.append(f"{sec.upper()}:\n{val.strip()}")
        return "\n\n".join(pieces)

    df["combined_text"] = df.apply(combine_sections, axis=1)
    print("\nCreating combined text from extracted sections...")
    print("   Combined text created successfully")
    return df

# =============================================================================
# DATA LOADING
# =============================================================================

def load_and_prepare_data():
    """Load data and convert to binary labels"""
    print("\nLoading data...")
    df = pd.read_excel(config.DATA_PATH)
    
    print(f"Total samples: {len(df)}")
    
    # Drop missing
    df = df.dropna(subset=["predicted_ohca", "clean_text"]).copy()
    df["clean_text"] = df["clean_text"].astype(str)
    
    # Show original multi-class distribution
    print("\nOriginal Multi-class Distribution:")
    vc = df["predicted_ohca"].value_counts().sort_index()
    for k, v in vc.items():
        class_name = {0: "Non-OHCA", 1: "OHCA", 2: "Transfer", 3: "In-hospital"}.get(k, f"Class {k}")
        print(f"   {class_name}: {v} ({100*v/len(df):.1f}%)")
    
    # FIX: Convert to BINARY labels
    # 0 = Non-OHCA (class 0, 2, 3)  ← Include transfers and in-hospital as NON-OHCA
    # 1 = OHCA (class 1 ONLY)       ← Only out-of-hospital cardiac arrest
    df["label"] = np.where(df["predicted_ohca"] == 1, 1, 0)
    
    print("\nBinary Label Distribution (CORRECTED):")
    vc = df["label"].value_counts()
    for label, count in vc.items():
        label_name = "OHCA (out-of-hospital only)" if label == 1 else "Non-OHCA (includes transfers & in-hospital)"
        print(f"   {label_name}: {count} ({100*count/len(df):.1f}%)")
    
    return df


def create_splits(df: pd.DataFrame):
    """Patient-level splits"""
    df = df.copy()
    df["hadm_id"] = df["hadm_id"].astype(str)
    unique_ids = df["hadm_id"].unique()
    
    print(f"\nTotal unique patients: {len(unique_ids)}")
    
    # 70% train, 15% val, 15% test
    train_ids, temp_ids = train_test_split(
        unique_ids, test_size=0.30, random_state=config.SEED, stratify=None
    )
    val_ids, test_ids = train_test_split(
        temp_ids, test_size=0.50, random_state=config.SEED, stratify=None
    )
    
    train_df = df[df["hadm_id"].isin(train_ids)].reset_index(drop=True)
    val_df = df[df["hadm_id"].isin(val_ids)].reset_index(drop=True)
    test_df = df[df["hadm_id"].isin(test_ids)].reset_index(drop=True)
    
    print("\nSplit Statistics:")
    for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        ohca_count = (split_df["label"] == 1).sum()
        print(f"   {name}: {len(split_df)} samples, {ohca_count} OHCA ({100*ohca_count/len(split_df):.1f}%)")
    
    return train_df, val_df, test_df


# =============================================================================
# DATASET
# =============================================================================

class OHCADataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer):
        self.texts = df["combined_text"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=config.MAX_LEN,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long)
        }


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(eval_pred: EvalPrediction):
    """Compute metrics for evaluation"""
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    preds = np.argmax(logits, axis=1)
    
    # Get probabilities for OHCA class (index 1)
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    probs_ohca = probs[:, 1]
    
    # Basic metrics
    acc = (preds == labels).mean()
    
    # Per-class metrics
    tn = ((preds == 0) & (labels == 0)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    tp = ((preds == 1) & (labels == 1)).sum()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    
    # AUC
    auc = roc_auc_score(labels, probs_ohca) if len(np.unique(labels)) > 1 else 0
    
    return {
        "accuracy": acc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "auc": auc,
    }


# =============================================================================
# CLASS-BALANCED SAMPLER
# =============================================================================

def get_weighted_sampler(dataset):
    """Create weighted sampler for class imbalance"""
    labels = np.array([dataset[i]["labels"].item() for i in range(len(dataset))])
    
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    sample_weights = class_weights[labels]
    
    print(f"\nClass weights for sampler:")
    print(f"   Non-OHCA (0): {class_weights[0]:.3f}")
    print(f"   OHCA (1): {class_weights[1]:.3f}")
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )


# =============================================================================
# CUSTOM TRAINER WITH WEIGHTED SAMPLER
# =============================================================================

class WeightedTrainer(Trainer):
    """Trainer with weighted sampling"""
    def _get_train_sampler(self):
        return get_weighted_sampler(self.train_dataset)


# =============================================================================
# TRAINING
# =============================================================================

def train_model(train_df, val_df, tokenizer):
    print("\n" + "="*80)
    print("TRAINING MODEL")
    print("="*80)
    
    # Create datasets
    train_dataset = OHCADataset(train_df, tokenizer)
    val_dataset = OHCADataset(val_df, tokenizer)
    
    print(f"\nTrain dataset: {len(train_dataset)} samples")
    print(f"Val dataset: {len(val_dataset)} samples")
    
    # Load model - STANDARD ARCHITECTURE
    print(f"\nLoading model: {config.MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(
        config.MODEL_NAME,
        num_labels=config.NUM_LABELS,
        id2label={0: "Non-OHCA", 1: "OHCA"},
        label2id={"Non-OHCA": 0, "OHCA": 1}
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRAD_ACCUM,
        learning_rate=config.LR,
        weight_decay=config.WEIGHT_DECAY,
        warmup_ratio=config.WARMUP_RATIO,
        
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        
        logging_steps=20,
        save_total_limit=2,
        
        fp16=False,  # Set to True if using GPU
        disable_tqdm=True,
        report_to=[],
        seed=config.SEED,
        save_safetensors=False, 
        
    )
    
    # Trainer with weighted sampling
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    print("\nStarting training...")
    trainer.train()
    
    print("\n✓ Training complete!")
    return trainer, model


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_model(model, tokenizer, test_df):
    """Detailed evaluation on test set"""
    print("\n" + "="*80)
    print("TEST SET EVALUATION")
    print("="*80)
    
    test_dataset = OHCADataset(test_df, tokenizer)
    
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for i in tqdm(range(len(test_dataset)), desc="Evaluating"):
            item = test_dataset[i]
            inputs = {
                "input_ids": item["input_ids"].unsqueeze(0).to(model.device),
                "attention_mask": item["attention_mask"].unsqueeze(0).to(model.device)
            }
            
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[0]
            
            all_probs.append(probs[1].item())  # OHCA probability
            all_labels.append(item["labels"].item())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Find optimal threshold
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in np.linspace(0.1, 0.9, 81):
        preds = (all_probs >= threshold).astype(int)
        tn = ((preds == 0) & (all_labels == 0)).sum()
        fp = ((preds == 1) & (all_labels == 0)).sum()
        fn = ((preds == 0) & (all_labels == 1)).sum()
        tp = ((preds == 1) & (all_labels == 1)).sum()
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    # Final predictions with best threshold
    preds = (all_probs >= best_threshold).astype(int)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, preds)
    tn, fp, fn, tp = cm.ravel()
    
    # Metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    auc = roc_auc_score(all_labels, all_probs)
    
    print(f"\n{'='*60}")
    print(f"RESULTS AT OPTIMAL THRESHOLD: {best_threshold:.3f}")
    print(f"{'='*60}")
    
    print("\nConfusion Matrix:")
    print("                 Pred Non-OHCA  Pred OHCA")
    print(f"Actual Non-OHCA        {tn:>3}          {fp:>3}")
    print(f"Actual OHCA             {fn:>3}          {tp:>3}")
    
    print("\nPerformance Metrics:")
    print(f"   Sensitivity (Recall): {sensitivity:.3f}")
    print(f"   Specificity:          {specificity:.3f}")
    print(f"   Precision (PPV):      {precision:.3f}")
    print(f"   NPV:                  {npv:.3f}")
    print(f"   F1-Score:             {f1:.3f}")
    print(f"   AUC-ROC:              {auc:.3f}")
    
    return best_threshold, {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "npv": npv,
        "f1": f1,
        "auc": auc,
        "threshold": best_threshold
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*80)
    print("OHCA CLASSIFIER V9 - SIMPLIFIED BINARY TRAINING")
    print("="*80)
    print(f"Device: {config.DEVICE}")
    print(f"Model: {config.MODEL_NAME}")
    print(f"Output: {config.OUTPUT_DIR}")
    print("="*80)
    
    # 1. Load and prepare data
    df = load_and_prepare_data()
    df = extract_all_sections(df, text_col="clean_text")
    
    # 2. Create splits
    train_df, val_df, test_df = create_splits(df)
    
    # 3. Load tokenizer
    print(f"\nLoading tokenizer: {config.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    
    # 4. Train
    trainer, model = train_model(train_df, val_df, tokenizer)
    
    # 5. Evaluate on test set
    best_threshold, metrics = evaluate_model(model, tokenizer, test_df)
    
    # 6. Save model and metadata
    final_model_dir = os.path.join(config.OUTPUT_DIR, "final_model")
    os.makedirs(final_model_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("SAVING MODEL")
    print(f"{'='*60}")
    print(f"Saving to: {final_model_dir}")

    # Save model and tokenizer - DISABLE SAFETENSORS!
    model.save_pretrained(final_model_dir, safe_serialization=False)  # ADD safe_serialization=False
    tokenizer.save_pretrained(final_model_dir)

    # Save training info
    import json
    training_info = {
        "model_name": config.MODEL_NAME,
        "task": "binary_classification",
        "labels": {"0": "Non-OHCA", "1": "OHCA"},
        "max_length": config.MAX_LEN,
        "optimal_threshold": float(best_threshold),
        "test_metrics": {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                        for k, v in metrics.items()},
        "training_params": {
            "epochs": config.NUM_EPOCHS,
            "batch_size": config.BATCH_SIZE,
            "learning_rate": config.LR,
            "weight_decay": config.WEIGHT_DECAY,
            "sections_used": config.SECTIONS
        }
    }

    with open(os.path.join(final_model_dir, "training_info.json"), "w") as f:
        json.dump(training_info, f, indent=2)

    print("✓ Model saved successfully!")
    print("\nFiles created:")
    print("   ✓ config.json")
    print("   ✓ pytorch_model.bin")  
    print("   ✓ tokenizer files")
    print("   ✓ training_info.json")


if __name__ == "__main__":
    main()
