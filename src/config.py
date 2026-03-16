from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"

OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
MODELS = OUTPUTS / "models"
TABLES = OUTPUTS / "tables"

for path in [DATA_RAW, DATA_INTERIM, DATA_PROCESSED, FIGURES, MODELS, TABLES]:
    path.mkdir(parents=True, exist_ok=True)