#!/usr/bin/env python3
from pathlib import Path

# Path to your CSV file
CSV_PATH = Path(r"C:\Users\QATCH\dev\VisQ.AI\ModelPackager\train_features.csv")


def remove_trailing_commas_in_place(path: Path) -> None:
    # Read all lines
    text = path.read_text(encoding="utf-8")
    # Process each line: strip CR/LF, then strip commas, then re-add '\n'
    cleaned = "\n".join(
        line.rstrip("\r\n").rstrip(",") for line in text.splitlines()
    ) + "\n"
    # Overwrite the original file
    path.write_text(cleaned, encoding="utf-8")


if __name__ == "__main__":
    remove_trailing_commas_in_place(CSV_PATH)
    print(f"Cleaned trailing commas in: {CSV_PATH}")
