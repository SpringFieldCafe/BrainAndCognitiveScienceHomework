from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = ROOT_DIR.parent
ASSIGNMENT_DIR = PROJECT_DIR / "assignment"
OUTPUT_DIR = ROOT_DIR / "images"

INPUT_SIZE = 8
HIDDEN_SIZE = 3
OUTPUT_SIZE = 8
EPOCHS = 50000
LEARNING_RATE = 10.0
SEED = 42
THRESHOLD = 0.5

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
