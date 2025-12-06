# GitHub Repository Setup Guide

## Step-by-Step Instructions to Upload Your Project to GitHub

### Prerequisites

1. **GitHub Account**: Create one at https://github.com if you don't have one
2. **Git Installed**: Check with `git --version` in terminal
   - If not installed: https://git-scm.com/downloads

---

## Option 1: Using GitHub Website (Easiest)

### Step 1: Create Repository on GitHub

1. Go to https://github.com
2. Click the **"+"** icon (top right) → **"New repository"**
3. Fill in:
   - **Repository name**: `ohca-classifier`
   - **Description**: "Automated OHCA identification from clinical notes using deep learning"
   - **Public** or **Private** (your choice)
   - ✅ **Add README file** (uncheck this - we already have one)
   - License: MIT
4. Click **"Create repository"**

### Step 2: Upload Files

**Method A: Drag and Drop (for small files)**
1. On your new repository page, click **"uploading an existing file"**
2. Drag all files from `/mnt/user-data/outputs/github_repo/` folder
3. Write commit message: "Initial commit: OHCA Classifier V11"
4. Click **"Commit changes"**

**Method B: Use GitHub Desktop (recommended)**
1. Download GitHub Desktop: https://desktop.github.com
2. Clone your repository
3. Copy all files from `/mnt/user-data/outputs/github_repo/` to the cloned folder
4. Commit and push

---

## Option 2: Using Command Line (Advanced)

### Step 1: Initialize Git Repository Locally

Open Terminal and navigate to your project folder:

```bash
# Navigate to where you want the project
cd /Users/monamoukaddem/Desktop

# Create project directory
mkdir ohca-classifier
cd ohca-classifier

# Copy all files from the generated folder
cp -r /mnt/user-data/outputs/github_repo/* .

# Initialize git
git init

# Add all files
git add .

# Make first commit
git commit -m "Initial commit: OHCA Classifier V11"
```

### Step 2: Create Repository on GitHub

1. Go to https://github.com
2. Click **"+"** → **"New repository"**
3. Name: `ohca-classifier`
4. **Don't** initialize with README (we have one)
5. Click **"Create repository"**

### Step 3: Push to GitHub

GitHub will show you commands like this (copy from your GitHub page):

```bash
# Add remote repository
git remote add origin https://github.com/monajm36/ohca-classifier.git

# Rename branch to main
git branch -M main

# Push to GitHub
git push -u origin main
```

Enter your GitHub credentials when prompted.

---

## What to Upload

### ✅ Files Already Created for You

These files are in `/mnt/user-data/outputs/github_repo/`:

```
ohca-classifier/
├── README.md                     ✓ Main documentation
├── requirements.txt              ✓ Dependencies
├── LICENSE                       ✓ MIT license
├── CONTRIBUTING.md               ✓ Contribution guide
├── .gitignore                    ✓ Ignore rules
├── SETUP_GUIDE.md               ✓ This file
├── scripts/
│   └── download_model.py         ✓ Download from HF
└── examples/
    └── quick_start.py            ✓ Usage example
```

### 📝 Files You Need to Add

**Training Scripts** (from your Desktop):
```bash
# Copy your training scripts
cp /Users/monamoukaddem/Desktop/train_v11_temporal__1_.py training/train_v11.py
cp /Users/monamoukaddem/Desktop/train_v10_*.py training/train_v10.py
cp /Users/monamoukaddem/Desktop/train_v9_*.py training/train_v9.py
```

**Prediction Scripts**:
```bash
# Copy prediction scripts
cp /path/to/predict_v11_temporal_location_aware.py prediction/predict_v11.py
cp /path/to/predict_v10_location_aware.py prediction/predict_v10.py
```

**Documentation** (if you have it):
```bash
# Copy comprehensive report
cp /mnt/user-data/outputs/OHCA_Classifier_Comprehensive_Report.docx docs/
```

**Notebooks** (if you have any):
```bash
# Copy Jupyter notebooks
cp /path/to/*.ipynb notebooks/
```

---

## After Uploading

### 1. Verify Repository

Visit: `https://github.com/monajm36/ohca-classifier`

Check that you see:
- ✅ README displays correctly
- ✅ Files are organized
- ✅ License shows "MIT"

### 2. Add Topics/Tags

On GitHub repository page:
1. Click ⚙️ (settings icon) next to "About"
2. Add topics: `machine-learning`, `deep-learning`, `healthcare`, `cardiac-arrest`, `nlp`, `transformers`, `pytorch`, `medical-ai`
3. Add website: `https://huggingface.co/monajm36/ohca-classifier-v11`
4. Save

### 3. Create Releases (Optional)

1. Go to repository → **"Releases"** → **"Create a new release"**
2. Tag: `v1.0.0`
3. Title: "V11: Temporal + Location-Aware OHCA Classifier"
4. Description: 
   ```
   First release of OHCA Classifier V11
   
   Performance on C19 validation (647 notes):
   - Sensitivity: 92.1%
   - Specificity: 89.4%
   - F1-Score: 0.856
   - AUC: 0.956
   
   Pre-trained model: https://huggingface.co/monajm36/ohca-classifier-v11
   ```
5. Click **"Publish release"**

### 4. Update README Links

Edit README.md and update:
- Replace `[your-email@example.com]` with your actual email
- Add any blog post links
- Add demo links (if you create one)

---

## Making Updates Later

### Add New Files

```bash
# Add new files
git add filename.py

# Commit with message
git commit -m "Add: new feature for XYZ"

# Push to GitHub
git push
```

### Update Existing Files

```bash
# After editing files
git add .
git commit -m "Update: improved documentation"
git push
```

---

## Troubleshooting

### "Permission denied" Error

Generate SSH key:
```bash
ssh-keygen -t ed25519 -C "your-email@example.com"
cat ~/.ssh/id_ed25519.pub
```

Add the key to GitHub: Settings → SSH Keys → New SSH Key

### "Repository too large" Error

Large model files should NOT be in GitHub. Use `.gitignore` to exclude:
- `*.bin` (model weights)
- Large CSV files

Point users to Hugging Face instead.

### Can't Push

```bash
# Pull first
git pull origin main

# Then push
git push origin main
```

---

## Quick Reference

| Action | Command |
|--------|---------|
| Check status | `git status` |
| Add files | `git add .` |
| Commit | `git commit -m "message"` |
| Push | `git push` |
| Pull updates | `git pull` |
| Create branch | `git checkout -b feature-name` |

---

## Next Steps

1. ✅ Upload to GitHub
2. ✅ Link to Hugging Face model
3. ⭐ Share on social media
4. 📝 Write blog post
5. 🎥 Create demo video
6. 📊 Add to portfolio/CV

---

**Questions?** Open an issue on GitHub or see CONTRIBUTING.md

**Repository URL:** `https://github.com/monajm36/ohca-classifier`
