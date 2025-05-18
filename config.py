from pathlib import Path

# Update paths to match new structure
PROJECT_ROOT = Path(__file__).parent.resolve()
MITBIH_PATH = PROJECT_ROOT / "mitdb" / "mit-bih-arrhythmia-database-1.0.0"
SAVE_PATH = PROJECT_ROOT / "processed_data"
SAMPLE_RATE = 360
SEGMENT_LENGTH = 360