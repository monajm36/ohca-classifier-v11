# =============================================================================
# OHCA CLASSIFIER V10 - LOCATION-AWARE BINARY TRAINING
# =============================================================================

import os
import re
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from typing import Optional
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data.sampler import WeightedRandomSampler

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_fscore_support

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
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
    OUTPUT_DIR = "/Users/monamoukaddem/Desktop/ohca_classifier_v10_location_aware"

    MAX_LEN = 512
    NUM_LABELS = 2  # Binary: 0=Non-OHCA, 1=OHCA
    NUM_LOCATION_FEATURES = 2  # OHCA indicators count, IHCA indicators count
    
    BATCH_SIZE = 4
    GRAD_ACCUM = 2
    LR = 2e-5
    NUM_EPOCHS = 5
    SEED = 42
    
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1

    SECTIONS = ["chief_complaint", "hpi", "ems", "ed_course", "impression"]
    DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


config = Config()
set_seed(config.SEED)
os.makedirs(config.OUTPUT_DIR, exist_ok=True)

# =============================================================================
# LOCATION INDICATORS
# =============================================================================

OHCA_INDICATORS = [
    'home', 'found at home', 'at home', 'ems', 'brought by ems', 'ems brought',
    'scene', 'field', 'bystander', 'found unresponsive', 'outside hospital',
    'pre-hospital', 'prehospital', 'arrived by ems', 'transported by',
    'found down', 'collapsed at', 'discovered at', 'witnessed at home',
    'family called', 'called 911', 'ambulance', 'paramedics'
]

IHCA_INDICATORS = [
    'floor', 'on floor', 'on the floor', 'unit', 'in unit', 'in the unit',
    'icu', 'in icu', 'in the icu', 'ward', 'on ward', 'room', 'bed',
    'code blue called', 'code blue', 'rapid response', 'on telemetry',
    'while admitted', 'during admission', 'in hospital', 'inpatient',
    'telemetry', 'monitored bed', 'hospital bed', 'admitted to',
    'transferred to icu', 'on ventilator', 'in bed'
]

def count_location_indicators(text, indicators):
    """Count how many location indicators are present"""
    if pd.isna(text):
        return 0
    
    text_lower = str(text).lower()
    count = 0
    
    for indicator in indicators:
        # Use word boundaries for more precise matching
        pattern = r'\b' + re.escape(indicator) + r'\b'
        count += len(re.findall(pattern, text_lower))
    
    return count

def extract_location_features(df, text_col='combined_text'):
    """Extract location features from text"""
    print("\n" + "="*60)
    print("EXTRACTING LOCATION FEATURES")
    print("="*60)
    
    df['ohca_indicator_count'] = df[text_col].apply(
        lambda x: count_location_indicators(x, OHCA_INDICATORS)
    )
    df['ihca_indicator_count'] = df[text_col].apply(
        lambda x: count_location_indicators(x, IHCA_INDICATORS)
    )
    
    # Derived feature: net location score (positive = more OHCA-like)
    df['location_score'] = df['ohca_indicator_count'] - df['ihca_indicator_count']
    
    print(f"\nLocation Feature Statistics:")
    print(f"   OHCA indicators - Mean: {df['ohca_indicator_count'].mean():.2f}, Max: {df['ohca_indicator_count'].max()}")
    print(f"   IHCA indicators - Mean: {df['ihca_indicator_count'].mean():.2f}, Max: {df['ihca_indicator_count'].max()}")
    print(f"   Location score - Mean: {df['location_score'].mean():.2f}")
    
    # Show by class
    for label in [0, 1]:
        label_name = "OHCA" if label == 1 else "Non-OHCA"
        subset = df[df['label'] == label]
        print(f"\n   {label_name} (n={len(subset)}):")
        print(f"      OHCA indicators: {subset['ohca_indicator_count'].mean():.2f}")
        print(f"      IHCA indicators: {subset['ihca_indicator_count'].mean():.2f}")
        print(f"      Location score: {subset['location_score'].mean():.2f}")
    
    return df

# =============================================================================
# SECTION EXTRACTION
# =============================================================================

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
    txt = txt.replace("\r", "\n")
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n+", "\n", txt)
    return txt.strip()

def extract_section(note: str, section_name: str) -> Optional[str]:
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

    sub = txt[start_idx:]
    header_matches = list(re.finditer(r"\n[A-Z0-9 /,&\-]{5,}:\s*", sub))
    if len(header_matches) > 0:
        end_idx = header_matches[0].start()
        section_text = sub[:end_idx]
    else:
        section_text = sub

    return section_text.strip()

def extract_all_sections(df: pd.DataFrame, text_col: str = "clean_text") -> pd.DataFrame:
    print(f"\n" + "="*60)
    print(f"EXTRACTING SECTIONS")
    print("="*60)
    print(f"Extracting from '{text_col}'...")
    print(f"Sections: {config.SECTIONS}")

    for sec in config.SECTIONS:
        df[sec] = df[text_col].apply(lambda x: extract_section(x, sec))
        n_non_null = df[sec].notnull().sum()
        print(f"   {sec}: {n_non_null}/{len(df)} ({100*n_non_null/len(df):.1f}%)")

    def combine_sections(row):
        pieces = []
        for sec in config.SECTIONS:
            val = row.get(sec, None)
            if isinstance(val, str) and val.strip():
                pieces.append(f"{sec.upper()}:\n{val.strip()}")
        return "\n\n".join(pieces) if pieces else row[text_col][:1000]

    df["combined_text"] = df.apply(combine_sections, axis=1)
    print("   ✓ Combined text created")
    return df

# =============================================================================
# DATA LOADING
# =============================================================================

def load_and_prepare_data():
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    df = pd.read_excel(config.DATA_PATH)
    print(f"\nTotal samples: {len(df)}")
    
    df = df.dropna(subset=["predicted_ohca", "clean_text"]).copy()
    df["clean_text"] = df["clean_text"].astype(str)
    
    print("\nOriginal Multi-class Distribution:")
    vc = df["predicted_ohca"].value_counts().sort_index()
    for k, v in vc.items():
        class_name = {0: "Non-OHCA", 1: "OHCA", 2: "Transfer", 3: "In-hospital"}.get(k, f"Class {k}")
        print(f"   {class_name}: {v} ({100*v/len(df):.1f}%)")
    
    # Binary labels: only class 1 is OHCA
    df["label"] = np.where(df["predicted_ohca"] == 1, 1, 0)
    
    print("\nBinary Label Distribution:")
    vc = df["label"].value_counts()
    for label, count in vc.items():
        label_name = "OHCA" if label == 1 else "Non-OHCA"
        print(f"   {label_name}: {count} ({100*count/len(df):.1f}%)")
    
    return df

def create_splits(df: pd.DataFrame):
    df = df.copy()
    df["hadm_id"] = df["hadm_id"].astype(str)
    unique_ids = df["hadm_id"].unique()
    
    print(f"\n" + "="*60)
    print("CREATING SPLITS")
    print("="*60)
    print(f"Total unique patients: {len(unique_ids)}")
    
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
# LOCATION-AWARE MODEL
# =============================================================================

class LocationAwareOHCAModel(nn.Module):
    """BERT model with additional location features"""
    
    def __init__(self, model_name, num_labels=2, num_location_features=2):
        super().__init__()
        
        # Load base BERT model
        self.bert = AutoModel.from_pretrained(model_name)
        bert_hidden_size = self.bert.config.hidden_size
        
        # Classifier that combines BERT output + location features
        self.classifier = nn.Sequential(
            nn.Linear(bert_hidden_size + num_location_features, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_labels)
        )
        
        self.num_labels = num_labels
    
    def forward(self, input_ids, attention_mask, location_features, labels=None):
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Concatenate with location features
        combined_features = torch.cat([pooled_output, location_features], dim=1)
        
        # Pass through classifier
        logits = self.classifier(combined_features)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        return {"loss": loss, "logits": logits}

# =============================================================================
# DATASET
# =============================================================================

class LocationAwareDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer):
        self.texts = df["combined_text"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()
        
        # Location features
        self.ohca_counts = df["ohca_indicator_count"].astype(float).tolist()
        self.ihca_counts = df["ihca_indicator_count"].astype(float).tolist()
        
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
        
        # Create location feature vector
        location_features = torch.tensor([
            self.ohca_counts[idx],
            self.ihca_counts[idx]
        ], dtype=torch.float32)
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "location_features": location_features,
            "labels": torch.tensor(label, dtype=torch.long)
        }

# =============================================================================
# CUSTOM TRAINER
# =============================================================================

class LocationAwareTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs, labels=labels)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        labels = inputs.pop("labels")
        
        with torch.no_grad():
            outputs = model(**inputs, labels=labels)
            loss = outputs["loss"]
            logits = outputs["logits"]
        
        return (loss, logits, labels)
    
    def _get_train_sampler(self):
        labels = np.array([self.train_dataset[i]["labels"].item() for i in range(len(self.train_dataset))])
        class_counts = np.bincount(labels)
        class_weights = 1. / class_counts
        sample_weights = class_weights[labels]
        
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(eval_pred: EvalPrediction):
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    preds = np.argmax(logits, axis=1)
    
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    probs_ohca = probs[:, 1]
    
    acc = (preds == labels).mean()
    
    tn = ((preds == 0) & (labels == 0)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    tp = ((preds == 1) & (labels == 1)).sum()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
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
# TRAINING
# =============================================================================

def train_model(train_df, val_df, tokenizer):
    print("\n" + "="*80)
    print("TRAINING LOCATION-AWARE MODEL")
    print("="*80)
    
    train_dataset = LocationAwareDataset(train_df, tokenizer)
    val_dataset = LocationAwareDataset(val_df, tokenizer)
    
    print(f"\nDatasets:")
    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Val: {len(val_dataset)} samples")
    
    print(f"\nInitializing model...")
    print(f"   Base model: {config.MODEL_NAME}")
    print(f"   Location features: {config.NUM_LOCATION_FEATURES}")
    
    model = LocationAwareOHCAModel(
        model_name=config.MODEL_NAME,
        num_labels=config.NUM_LABELS,
        num_location_features=config.NUM_LOCATION_FEATURES
    )
    
    # Move to device
    model = model.to(config.DEVICE)
    
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
        
        fp16=False,
        report_to=[],
        seed=config.SEED,
        
        save_safetensors=False,
        disable_tqdm=False,
    )
    
    trainer = LocationAwareTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    print("\nStarting training...")
    print("="*60)
    trainer.train()
    
    print("\n✓ Training complete!")
    return trainer, model

# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_model(model, tokenizer, test_df):
    print("\n" + "="*80)
    print("TEST SET EVALUATION")
    print("="*80)
    
    test_dataset = LocationAwareDataset(test_df, tokenizer)
    
    model.eval()
    all_probs = []
    all_labels = []
    
    print("\nRunning inference...")
    with torch.no_grad():
        for i in tqdm(range(len(test_dataset)), desc="Evaluating"):
            item = test_dataset[i]
            inputs = {
                "input_ids": item["input_ids"].unsqueeze(0).to(model.device if hasattr(model, 'device') else config.DEVICE),
                "attention_mask": item["attention_mask"].unsqueeze(0).to(model.device if hasattr(model, 'device') else config.DEVICE),
                "location_features": item["location_features"].unsqueeze(0).to(model.device if hasattr(model, 'device') else config.DEVICE)
            }
            
            outputs = model(**inputs)
            logits = outputs["logits"]
            probs = torch.softmax(logits, dim=1)[0]
            
            all_probs.append(probs[1].item())
            all_labels.append(item["labels"].item())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Find optimal threshold
    print("\nFinding optimal threshold...")
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
    
    # Final predictions at optimal threshold
    preds = (all_probs >= best_threshold).astype(int)
    cm = confusion_matrix(all_labels, preds)
    tn, fp, fn, tp = cm.ravel()
    
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
# SAVE MODEL
# =============================================================================

def save_model(model, tokenizer, best_threshold, metrics):
    final_model_dir = os.path.join(config.OUTPUT_DIR, "final_model")
    os.makedirs(final_model_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("SAVING MODEL")
    print(f"{'='*60}")
    print(f"Saving to: {final_model_dir}")
    
    # Save model state dict
    torch.save(model.state_dict(), os.path.join(final_model_dir, "pytorch_model.bin"))
    
    # Save tokenizer
    tokenizer.save_pretrained(final_model_dir)
    
    # Save config
    model_config = {
        "model_type": "LocationAwareOHCAModel",
        "base_model": config.MODEL_NAME,
        "num_labels": config.NUM_LABELS,
        "num_location_features": config.NUM_LOCATION_FEATURES,
        "max_length": config.MAX_LEN,
    }
    
    with open(os.path.join(final_model_dir, "model_config.json"), "w") as f:
        json.dump(model_config, f, indent=2)
    
    # Save training info
    training_info = {
        "model_name": config.MODEL_NAME,
        "version": "v10_location_aware",
        "task": "binary_classification_with_location_features",
        "labels": {"0": "Non-OHCA", "1": "OHCA"},
        "location_features": ["ohca_indicator_count", "ihca_indicator_count"],
        "max_length": config.MAX_LEN,
        "optimal_threshold": float(best_threshold),
        "test_metrics": {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                        for k, v in metrics.items()},
        "training_params": {
            "epochs": config.NUM_EPOCHS,
            "batch_size": config.BATCH_SIZE,
            "learning_rate": config.LR,
            "weight_decay": config.WEIGHT_DECAY,
            "sections_used": config.SECTIONS,
        },
        "ohca_indicators": OHCA_INDICATORS,
        "ihca_indicators": IHCA_INDICATORS
    }
    
    with open(os.path.join(final_model_dir, "training_info.json"), "w") as f:
        json.dump(training_info, f, indent=2)
    
    print("\n✓ Model saved successfully!")
    print("\nFiles created:")
    print("   ✓ pytorch_model.bin")
    print("   ✓ model_config.json")
    print("   ✓ tokenizer files")
    print("   ✓ training_info.json")
    
    return final_model_dir

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*80)
    print("OHCA CLASSIFIER V10 - LOCATION-AWARE TRAINING")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"   Device: {config.DEVICE}")
    print(f"   Base Model: {config.MODEL_NAME}")
    print(f"   Output: {config.OUTPUT_DIR}")
    print(f"   Location Features: {config.NUM_LOCATION_FEATURES}")
    print("="*80)
    
    # 1. Load and prepare data
    df = load_and_prepare_data()
    
    # 2. Extract sections
    df = extract_all_sections(df)
    
    # 3. Extract location features
    df = extract_location_features(df)
    
    # 4. Create splits
    train_df, val_df, test_df = create_splits(df)
    
    # 5. Load tokenizer
    print(f"\n{'='*60}")
    print("LOADING TOKENIZER")
    print(f"{'='*60}")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    print("✓ Tokenizer loaded")
    
    # 6. Train
    trainer, model = train_model(train_df, val_df, tokenizer)
    
    # 7. Evaluate on test set
    best_threshold, metrics = evaluate_model(model, tokenizer, test_df)
    
    # 8. Save model
    final_model_dir = save_model(model, tokenizer, best_threshold, metrics)
    
    print("\n" + "="*80)
    print("✓ TRAINING COMPLETE!")
    print("="*80)
    print(f"\nModel saved to: {final_model_dir}")
    print(f"\nKey improvements in V10:")
    print("   ✓ Explicit location features (OHCA vs IHCA indicators)")
    print("   ✓ Custom classifier that combines BERT + location")
    print("   ✓ Should better distinguish IHCA from OHCA")


if __name__ == "__main__":
    main()
