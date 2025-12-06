"""
Download OHCA Classifier V11 Model from Hugging Face
====================================================

Usage:
    python scripts/download_model.py
"""

import os
from huggingface_hub import snapshot_download

def download_model(output_dir="./models/v11_pretrained"):
    """
    Download the pre-trained V11 model from Hugging Face.
    
    Args:
        output_dir: Directory to save model files
    """
    
    print("="*70)
    print("DOWNLOADING OHCA CLASSIFIER V11")
    print("="*70)
    
    repo_id = "monajm36/ohca-classifier-v11"
    
    print(f"\nRepository: {repo_id}")
    print(f"Output directory: {output_dir}")
    
    try:
        print("\nDownloading model files...")
        
        # Download entire model repository
        model_path = snapshot_download(
            repo_id=repo_id,
            local_dir=output_dir,
            local_dir_use_symlinks=False
        )
        
        print(f"\n✓ Model downloaded successfully!")
        print(f"✓ Saved to: {model_path}")
        
        # List downloaded files
        print("\nDownloaded files:")
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                filepath = os.path.join(root, file)
                size_mb = os.path.getsize(filepath) / 1024 / 1024
                print(f"  - {file} ({size_mb:.1f} MB)")
        
        print("\n" + "="*70)
        print("Ready to use!")
        print("="*70)
        print("\nNext steps:")
        print("  1. See examples in notebooks/demo.ipynb")
        print("  2. Or run: python prediction/predict_v11.py")
        
    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        print("\nTroubleshooting:")
        print("  1. Check internet connection")
        print("  2. Install huggingface_hub: pip install huggingface_hub")
        print("  3. Visit: https://huggingface.co/monajm36/ohca-classifier-v11")

if __name__ == "__main__":
    download_model()
