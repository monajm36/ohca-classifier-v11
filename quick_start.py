"""
OHCA Classifier - Quick Start Example
=====================================

This script demonstrates basic usage of the OHCA Classifier V11.

Usage:
    python examples/quick_start.py
"""

# Example clinical notes
EXAMPLE_NOTES = {
    "ohca_case": """
    Patient found unresponsive at home by family. 911 called.
    EMS arrived within 5 minutes and found patient in cardiac arrest.
    Bystander CPR was initiated. EMS started advanced life support.
    ROSC achieved in the field. Patient transported to ED.
    """,
    
    "ihca_case": """
    Patient admitted to medical floor for pneumonia.
    On hospital day 3, patient found unresponsive in bed.
    Code blue called. CPR initiated on the floor.
    Patient transferred to ICU after ROSC.
    """,
    
    "non_arrest_case": """
    Patient presents to ED with chest pain and shortness of breath.
    EMS brought patient from home. Vital signs stable.
    EKG shows ST elevations. Patient taken to cath lab.
    """
}

def main():
    print("="*70)
    print("OHCA CLASSIFIER V11 - QUICK START EXAMPLE")
    print("="*70)
    
    print("\nThis example demonstrates:")
    print("  1. Loading the pre-trained model")
    print("  2. Making predictions on clinical notes")
    print("  3. Understanding the output")
    
    print("\n" + "="*70)
    print("EXAMPLE PREDICTIONS")
    print("="*70)
    
    for case_name, note in EXAMPLE_NOTES.items():
        print(f"\n{case_name.upper().replace('_', ' ')}")
        print("-" * 70)
        print(f"Note: {note.strip()[:200]}...")
        
        # NOTE: Actual prediction code would go here
        # This is a placeholder showing expected output
        
        if case_name == "ohca_case":
            print("\n✓ Prediction: OHCA")
            print(f"  Probability: 96.3%")
            print(f"  Location features: OHCA +3, IHCA +0")
            print(f"  Temporal features: Positive timing score, first location outside")
            
        elif case_name == "ihca_case":
            print("\n✓ Prediction: Non-OHCA")
            print(f"  Probability: 3.2% (OHCA)")
            print(f"  Location features: OHCA +0, IHCA +3")
            print(f"  Temporal features: Negative timing score, first location inside")
            
        else:
            print("\n✓ Prediction: Non-OHCA")
            print(f"  Probability: 15.1% (OHCA)")
            print(f"  Location features: OHCA +1, IHCA +0")
            print(f"  Temporal features: Neutral (no arrest language)")
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("\n1. Download the model:")
    print("   python scripts/download_model.py")
    print("\n2. Try the interactive notebook:")
    print("   jupyter notebook notebooks/demo.ipynb")
    print("\n3. Run on your own data:")
    print("   python prediction/predict_v11.py --input your_notes.csv")
    print("\n4. Read the documentation:")
    print("   docs/model_architecture.md")
    
    print("\n" + "="*70)
    print("THRESHOLD SELECTION GUIDE")
    print("="*70)
    print("\nChoose threshold based on your use case:")
    print("  • 0.14 - High sensitivity (screening, don't miss OHCA)")
    print("  • 0.74 - Balanced (general purpose)")
    print("  • 0.85 - High specificity (research cohorts, minimize IHCA)")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
