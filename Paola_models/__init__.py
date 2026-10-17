"""Paola's isolated experimental models; run modules from the ApplePi root."""
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
# Reuse the existing challenge input checks, calendar history, and PCHIP baseline.
sys.path.insert(0, str(REPO / "AET-HACKATHON"))
