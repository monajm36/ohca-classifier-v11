# =============================================================================
# OHCA CLASSIFIER V11 - TEMPORAL + LOCATION-AWARE BINARY TRAINING
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
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from transformers import (
    AutoTokenizer,
    AutoModel,
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
    DATA_PATH = "/Users/monamoukaddem/Desktop/mimic_labelled.csv"
    OUTPUT_DIR = "/Users/monamoukaddem/Desktop/ohca_classifier_v11_temporal"

    MAX_LEN = 512
    NUM_LABELS = 2  # Binary: 0=Non-OHCA, 1=OHCA
    NUM_LOCATION_FEATURES = 2  # OHCA indicators count, IHCA indicators count
    NUM_TEMPORAL_FEATURES = 7  # NEW: Temporal features
    
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
# LOCATION INDICATORS (from V10)
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

# =============================================================================
# TEMPORAL INDICATORS (NEW FOR V11)
# =============================================================================

ARREST_BEFORE_ADMISSION_PHRASES = [
    'presented with arrest', 'presented in arrest', 'arrived in arrest',
    'arrived with arrest', 'arrest prior to arrival', 'arrest en route',
    'arrest before arrival', 'found in arrest', 'cpr by ems',
    'cpr initiated by ems', 'ems found', 'ems initiated cpr',
    'arrest at home', 'arrest at scene', 'prehospital arrest',
    'pre-hospital arrest', 'outside hospital arrest', 'arrest outside',
    'on arrival in arrest', 'brought in arrest', 'transferred in arrest',
    'arrest before ed', 'arrest prior to ed'
]

ARREST_DURING_STAY_PHRASES = [
    'hospital day', 'while admitted', 'during admission', 'after admission',
    'following admission', 'subsequently arrested', 'later arrested',
    'days into admission', 'during hospitalization', 'while in hospital',
    'while on floor', 'while in icu', 'during stay', 'later coded',
    'subsequently coded', 'then arrested', 'then coded', 'later that',
    'following day', 'next day arrested', 'hours later arrested',
    'developed arrest', 'went into arrest', 'had arrest'
]

OUTSIDE_LOCATIONS = [
    'home', 'scene', 'street', 'work', 'restaurant', 'gym', 'store',
    'outside', 'residence', 'apartment', 'house', 'public', 'workplace'
]

INSIDE_LOCATIONS = [
    'ed', 'emergency department', 'floor', 'icu', 'ward', 'unit',
    'room', 'bed', 'telemetry', 'monitored', 'hospital'
]

TRANSPORT_PHRASES = [
    'brought by', 'transported by', 'ems brought', 'ems transported',
    'ambulance brought', 'arrived by', 'transferred from', 'came from'
]

# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def count_location_indicators(text, indicators):
    """Count how many location indicators are present"""
    if pd.isna(text):
        return 0
    
    text_lower = str(text).lower()
    count = 0
    
    for indicator in indicators:
        pattern = r'\b' + re.escape(indicator) + r'\b'
        count += len(re.findall(pattern, text_lower))
    
    return count

def extract_arrest_timing_score(text):
    """Extract temporal feature: arrest timing"""
    if pd.isna(text):
        return 0
    text_lower = str(text).lower()
    
    before_count = sum(len(re.findall(r'\b' + re.escape(phrase) + r'\b', text_lower)) 
                      for phrase in ARREST_BEFORE_ADMISSION_PHRASES)
    during_count = sum(len(re.findall(r'\b' + re.escape(phrase) + r'\b', text_lower)) 
                      for phrase in ARREST_DURING_STAY_PHRASES)
    
    return before_count - during_count

def extract_first_location_features(text):
    """Extract first mentioned location type"""
    if pd.isna(text):
        return 0, 0
    text_lower = str(text).lower()
    
    first_outside_pos = float('inf')
    for loc in OUTSIDE_LOCATIONS:
        match = re.search(r'\b' + re.escape(loc) + r'\b', text_lower)
        if match:
            first_outside_pos = min(first_outside_pos, match.start())
    
    first_inside_pos = float('inf')
    for loc in INSIDE_LOCATIONS:
        match = re.search(r'\b' + re.escape(loc) + r'\b', text_lower)
        if match:
            first_inside_pos = min(first_inside_pos, match.start())
    
    if first_outside_pos < first_inside_pos:
        return 1, 0
    elif first_inside_pos < first_outside_pos:
        return 0, 1
    else:
        return 0, 0

def extract_movement_direction(text):
    """Detect movement direction pattern"""
    if pd.isna(text):
        return 0, 0, 1
    text_lower = str(text).lower()
    
    has_outside_origin = False
    has_transport = False
    has_inside_destination = False
    
    for transport in TRANSPORT_PHRASES:
        match = re.search(r'(.{0,50})' + re.escape(transport), text_lower)
        if match:
            has_transport = True
            context_before = match.group(1)
            for outside_loc in OUTSIDE_LOCATIONS:
                if outside_loc in context_before:
                    has_outside_origin = True
                    break
    
    for inside_loc in INSIDE_LOCATIONS:
        if inside_loc in text_lower:
            has_inside_destination = True
            break
    
    if has_outside_origin and has_transport and has_inside_destination:
        return 1, 0, 0
    elif has_transport and has_inside_destination and not has_outside_origin:
        return 0, 1, 0
    else:
        return 0, 0, 1

def extract_hospital_day_mention(text):
    """Check for hospital day mentions"""
    if pd.isna(text):
        return 0
    text_lower = str(text).lower()
    pattern = r'hospital day \d+'
    return 1 if re.findall(pattern, text_lower) else 0

def extract_location_features(df, text_col='combined_text'):
    """Extract V10 location features from text"""
    print("\n" + "="*60)
    print("EXTRACTING V10 LOCATION FEATURES")
    print("="*60)
    
    df['ohca_indicator_count'] = df[text_col].apply(
        lambda x: count_location_indicators(x, OHCA_INDICATORS)
    )
    df['ihca_indicator_count'] = df[text_col].apply(
        lambda x: count_location_indicators(x, IHCA_INDICATORS)
    )
    
    df['location_score'] = df['ohca_indicator_count'] - df['ihca_indicator_count']
    
    print(f"\nLocation Feature Statistics:")
    print(f"   OHCA indicators - Mean: {df['ohca_indicator_count'].mean():.2f}, Max: {df['ohca_indicator_count'].max()}")
    print(f"   IHCA indicators - Mean: {df['ihca_indicator_count'].mean():.2f}, Max: {df['ihca_indicator_count'].max()}")
    print(f"   Location score - Mean: {df['location_score'].mean():.2f}")
    
    for label in [0, 1]:
        label_name = "OHCA" if label == 1 else "Non-OHCA"
        subset = df[df['label'] == label]
        if len(subset) > 0:
            print(f"\n   {label_name} (n={len(subset)}):")
            print(f"      OHCA indicators: {subset['ohca_indicator_count'].mean():.2f}")
            print(f"      IHCA indicators: {subset['ihca_indicator_count'].mean():.2f}")
            print(f"      Location score: {subset['location_score'].mean():.2f}")
    
    return df

def extract_temporal_features(df, text_col='combined_text'):
    """Extract V11 temporal features from text"""
    print("\n" + "="*60)
    print("EXTRACTING V11 TEMPORAL FEATURES")
    print("="*60)
    
    df['arrest_timing_score'] = df[text_col].apply(extract_arrest_timing_score)
    df['first_loc_outside'], df['first_loc_inside'] = zip(*df[text_col].apply(extract_first_location_features))
    df['move_out_to_in'], df['move_in_to_in'], df['move_unclear'] = zip(*df[text_col].apply(extract_movement_direction))
    df['hospital_day_mention'] = df[text_col].apply(extract_hospital_day_mention)
    
    print(f"\nTemporal Feature Statistics:")
    print(f"   Arrest timing score - Mean: {df['arrest_timing_score'].mean():.2f}")
    print(f"   First location outside: {df['first_loc_outside'].sum()} cases ({100*df['first_loc_outside'].sum()/len(df):.1f}%)")
    print(f"   First location inside: {df['first_loc_inside'].sum()} cases ({100*df['first_loc_inside'].sum()/len(df):.1f}%)")
    print(f"   Movement outside→inside: {df['move_out_to_in'].sum()} cases ({100*df['move_out_to_in'].sum()/len(df):.1f}%)")
    print(f"   Movement inside→inside: {df['move_in_to_in'].sum()} cases ({100*df['move_in_to_in'].sum()/len(df):.1f}%)")
    print(f"   Hospital day mentions: {df['hospital_day_mention'].sum()} cases ({100*df['hospital_day_mention'].sum()/len(df):.1f}%)")
    
    for label in [0, 1]:
        label_name = "OHCA" if label == 1 else "Non-OHCA"
        subset = df[df['label'] == label]
        if len(subset) > 0:
            print(f"\n   {label_name} (n={len(subset)}):")
            print(f"      Arrest timing score: {subset['arrest_timing_score'].mean():.2f}")
            print(f"      First loc outside: {subset['first_loc_outside'].sum()} ({100*subset['first_loc_outside'].sum()/len(subset):.1f}%)")
            print(f"      Movement out→in: {subset['move_out_to_in'].sum()} ({100*subset['move_out_to_in'].sum()/len(subset):.1f}%)")
            print(f"      Hospital day: {subset['hospital_day_mention'].sum()} ({100*subset['hospital_day_mention'].sum()/len(subset):.1f}%)")
    
    return df

# =============================================================================
# SECTION EXTRACTION (from V10)
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
    
    df = pd.read_csv(config.DATA_PATH)
    print(f"\nTotal samples: {len(df)}")
    
    # Check for required columns
    print(f"Columns: {list(df.columns)}")
    
    # Handle different possible column names
    if 'predicted_ohca' in df.columns:
        label_col = 'predicted_ohca'
    elif 'label' in df.columns:
        label_col = 'label'
    else:
        raise ValueError("No label column found! Expected 'predicted_ohca' or 'label'")
    
    if 'clean_text' in df.columns:
        text_col = 'clean_text'
    elif 'full_note' in df.columns:
        text_col = 'full_note'
    else:
        raise ValueError("No text column found! Expected 'clean_text' or 'full_note'")
    
    df = df.dropna(subset=[label_col, text_col]).copy()
    df["clean_text"] = df[text_col].astype(str)
    
    print("\nOriginal Multi-class Distribution:")
    vc = df[label_col].value_counts().sort_index()
    for k, v in vc.items():
        class_name = {0: "Non-OHCA", 1: "OHCA", 2: "Transfer", 3: "In-hospital"}.get(k, f"Class {k}")
        print(f"   {class_name}: {v} ({100*v/len(df):.1f}%)")
    
    # Binary labels: only class 1 is OHCA
    df["label"] = np.where(df[label_col] == 1, 1, 0)
    
    print("\nBinary Label Distribution:")
    vc = df["label"].value_counts()
    for label, count in vc.items():
        label_name = "OHCA" if label == 1 else "Non-OHCA"
        print(f"   {label_name}: {count} ({100*count/len(df):.1f}%)")
    
    return df

def create_splits(df: pd.DataFrame):
    df = df.copy()
    
    # Create patient ID if doesn't exist
    if 'hadm_id' not in df.columns:
        df['hadm_id'] = df.index.astype(str)
    
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
# TEMPORAL + LOCATION-AWARE MODEL (V11)
# =============================================================================

class TemporalOHCAModel(nn.Module):
    """BERT model with location + temporal features"""
    
    def __init__(self, model_name, num_labels=2, num_location_features=2, num_temporal_features=7):
        super().__init__()
        
        # Load base BERT model
        self.bert = AutoModel.from_pretrained(model_name)
        bert_hidden_size = self.bert.config.hidden_size
        
        # Classifier that combines BERT output + location + temporal features
        total_features = bert_hidden_size + num_location_features + num_temporal_features
        
        self.classifier = nn.Sequential(
            nn.Linear(total_features, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_labels)
        )
        
        self.num_labels = num_labels
        self.num_location_features = num_location_features
        self.num_temporal_features = num_temporal_features
    
    def forward(self, input_ids, attention_mask, location_features, temporal_features, labels=None):
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Concatenate with location + temporal features
        combined_features = torch.cat([pooled_output, location_features, temporal_features], dim=1)
        
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

class TemporalDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer):
        self.texts = df["combined_text"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()
        
        # Location features (V10)
        self.ohca_counts = df["ohca_indicator_count"].astype(float).tolist()
        self.ihca_counts = df["ihca_indicator_count"].astype(float).tolist()
        
        # Temporal features (V11)
        self.arrest_timing = df["arrest_timing_score"].astype(float).tolist()
        self.first_loc_out = df["first_loc_outside"].astype(float).tolist()
        self.first_loc_in = df["first_loc_inside"].astype(float).tolist()
        self.move_out_in = df["move_out_to_in"].astype(float).tolist()
        self.move_in_in = df["move_in_to_in"].astype(float).tolist()
        self.move_unclear = df["move_unclear"].astype(float).tolist()
        self.hosp_day = df["hospital_day_mention"].astype(float).tolist()
        
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
        
        # Create location feature vector (V10)
        location_features = torch.tensor([
            self.ohca_counts[idx],
            self.ihca_counts[idx]
        ], dtype=torch.float32)
        
        # Create temporal feature vector (V11)
        temporal_features = torch.tensor([
            self.arrest_timing[idx],
            self.first_loc_out[idx],
            self.first_loc_in[idx],
            self.move_out_in[idx],
            self.move_in_in[idx],
            self.move_unclear[idx],
            self.hosp_day[idx]
        ], dtype=torch.float32)
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "location_features": location_features,
            "temporal_features": temporal_features,
            "labels": torch.tensor(label, dtype=torch.long)
        }

# =============================================================================
# CUSTOM TRAINER
# =============================================================================

class TemporalTrainer(Trainer):
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
    print("TRAINING TEMPORAL + LOCATION-AWARE MODEL (V11)")
    print("="*80)
    
    train_dataset = TemporalDataset(train_df, tokenizer)
    val_dataset = TemporalDataset(val_df, tokenizer)
    
    print(f"\nDatasets:")
    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Val: {len(val_dataset)} samples")
    
    print(f"\nInitializing V11 model...")
    print(f"   Base model: {config.MODEL_NAME}")
    print(f"   Location features: {config.NUM_LOCATION_FEATURES}")
    print(f"   Temporal features: {config.NUM_TEMPORAL_FEATURES}")
    print(f"   Total input dim: 768 + {config.NUM_LOCATION_FEATURES} + {config.NUM_TEMPORAL_FEATURES} = {768 + config.NUM_LOCATION_FEATURES + config.NUM_TEMPORAL_FEATURES}")
    
    model = TemporalOHCAModel(
        model_name=config.MODEL_NAME,
        num_labels=config.NUM_LABELS,
        num_location_features=config.NUM_LOCATION_FEATURES,
        num_temporal_features=config.NUM_TEMPORAL_FEATURES
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
    
    trainer = TemporalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    print("\nStarting training...")
    print("="*60)
    trainer.train()
    
    return trainer, model

# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_test_set(trainer, test_df, tokenizer):
    print("\n" + "="*80)
    print("EVALUATING ON TEST SET")
    print("="*80)
    
    test_dataset = TemporalDataset(test_df, tokenizer)
    
    # Get predictions
    predictions = trainer.predict(test_dataset)
    logits = predictions.predictions
    labels = predictions.label_ids
    
    # Calculate probabilities
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    probs_ohca = probs[:, 1]
    
    # Find optimal threshold
    fpr, tpr, thresholds = roc_curve(labels, probs_ohca)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    print(f"\nOptimal threshold: {optimal_threshold:.3f}")
    
    # Apply optimal threshold
    preds_optimal = (probs_ohca >= optimal_threshold).astype(int)
    
    # Calculate metrics
    tn, fp, fn, tp = confusion_matrix(labels, preds_optimal).ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    auc = roc_auc_score(labels, probs_ohca)
    
    print("\n" + "="*60)
    print("TEST SET RESULTS")
    print("="*60)
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted Non-OHCA  Predicted OHCA")
    print(f"Actual Non-OHCA         {tn:4d}              {fp:4d}")
    print(f"Actual OHCA             {fn:4d}              {tp:4d}")
    print(f"\nMetrics:")
    print(f"  Sensitivity (Recall): {sensitivity:.3f} ({sensitivity*100:.1f}%)")
    print(f"  Specificity:          {specificity:.3f} ({specificity*100:.1f}%)")
    print(f"  Precision (PPV):      {precision:.3f} ({precision*100:.1f}%)")
    print(f"  NPV:                  {npv:.3f} ({npv*100:.1f}%)")
    print(f"  F1-Score:             {f1:.3f}")
    print(f"  AUC-ROC:              {auc:.3f}")
    
    return {
        "optimal_threshold": float(optimal_threshold),
        "test_metrics": {
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "precision": float(precision),
            "npv": float(npv),
            "f1": float(f1),
            "auc": float(auc)
        },
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        }
    }

# =============================================================================
# SAVE MODEL
# =============================================================================

def save_final_model(model, tokenizer, test_results):
    print("\n" + "="*60)
    print("SAVING FINAL MODEL")
    print("="*60)
    
    final_dir = os.path.join(config.OUTPUT_DIR, "final_model")
    os.makedirs(final_dir, exist_ok=True)
    
    # Save model weights
    torch.save(model.state_dict(), os.path.join(final_dir, "pytorch_model.bin"))
    
    # Save tokenizer
    tokenizer.save_pretrained(final_dir)
    
    # Save model config
    model_config = {
        'base_model': config.MODEL_NAME,
        'num_labels': config.NUM_LABELS,
        'num_location_features': config.NUM_LOCATION_FEATURES,
        'num_temporal_features': config.NUM_TEMPORAL_FEATURES,
        'max_length': config.MAX_LEN
    }
    
    with open(os.path.join(final_dir, "model_config.json"), 'w') as f:
        json.dump(model_config, f, indent=2)
    
    # Save training info
    training_info = {
        'optimal_threshold': test_results['optimal_threshold'],
        'test_metrics': test_results['test_metrics'],
        'confusion_matrix': test_results['confusion_matrix'],
        'training_params': {
            'batch_size': config.BATCH_SIZE,
            'learning_rate': config.LR,
            'num_epochs': config.NUM_EPOCHS,
            'seed': config.SEED
        }
    }
    
    with open(os.path.join(final_dir, "training_info.json"), 'w') as f:
        json.dump(training_info, f, indent=2)
    
    print(f"\n✓ Model saved to: {final_dir}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*80)
    print("OHCA CLASSIFIER V11 - TEMPORAL + LOCATION-AWARE TRAINING")
    print("="*80)
    print(f"Device: {config.DEVICE}")
    
    # Load data
    df = load_and_prepare_data()
    
    # Extract sections
    df = extract_all_sections(df, text_col="clean_text")
    
    # Extract V10 location features
    df = extract_location_features(df, text_col="combined_text")
    
    # Extract V11 temporal features
    df = extract_temporal_features(df, text_col="combined_text")
    
    # Create splits
    train_df, val_df, test_df = create_splits(df)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    
    # Train model
    trainer, model = train_model(train_df, val_df, tokenizer)
    
    # Evaluate on test set
    test_results = evaluate_test_set(trainer, test_df, tokenizer)
    
    # Save final model
    save_final_model(model, tokenizer, test_results)
    
    print("\n" + "="*80)
    print("✓ V11 TRAINING COMPLETE!")
    print("="*80)
    print(f"\nModel saved to: {config.OUTPUT_DIR}/final_model/")
    print("\nNext steps:")
    print("  1. Apply V11 to C19 validation set: python predict_v11_temporal.py")
    print("  2. Compare V10 vs V11 performance")
    print("="*80)

if __name__ == "__main__":
    main()
