from pathlib import Path

def verify_structure():
    required = [
        "mitdb/mit-bih-arrhythmia-database-1.0.0/100.hea",
        "processed_data/ecg_segments.npy",
        "app.py",
        "config.py",
        "data_pipeline.py",
        "model.py",
        "utils.py"
    ]
    
    missing = []
    for path in required:
        if not Path(path).exists():
            missing.append(path)
    
    if missing:
        print("🚨 Missing files:")
        for m in missing:
            print(f"- {m}")
    else:
        print("✅ Folder structure is correct!")

if __name__ == "__main__":
    verify_structure()

