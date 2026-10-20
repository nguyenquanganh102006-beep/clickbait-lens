"""Default local paths, optionally isolated for automated application checks."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = Path(os.environ.get("KHDL_STATE_DIR", str(ROOT / "state")))
INPUT_DIR = Path(os.environ.get("KHDL_INPUT_DIR", str(ROOT / "data" / "input")))
OUTPUT_DIR = Path(os.environ.get("KHDL_OUTPUT_DIR", str(ROOT / "data" / "output")))
