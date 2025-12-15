#!/usr/bin/env python3
"""
Extract important columns from experiments_log.csv for quick review
"""
import csv
from pathlib import Path

# Define the important columns to extract
IMPORTANT_COLUMNS = [
    'experiment_id',
    'timestamp',
    'dataset_name',
    'train_dataset_name',
    'model_name_or_path',
    'num_train_epochs',
    'learning_rate',
    'base_model_accuracy',
    'base_model_f1_score',
    'base_model_f1_moving',
    'base_model_f1_stopped',
    'finetuned_model_accuracy',
    'finetuned_model_f1_score',
    'finetuned_model_f1_moving',
    'finetuned_model_f1_stopped',
]

def extract_summary(input_file, output_file):
    """Extract important columns from experiments log"""
    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_file}")
        return

    with open(input_path, 'r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)

        # Get the columns that exist in the file
        available_columns = [col for col in IMPORTANT_COLUMNS if col in reader.fieldnames]

        # Read all rows
        rows = list(reader)

        if not rows:
            print("No data found in the CSV file")
            return

    # Write the extracted data
    with open(output_path, 'w', encoding='utf-8', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=available_columns)
        writer.writeheader()

        for row in rows:
            # Extract only the important columns
            extracted_row = {col: row.get(col, '') for col in available_columns}
            writer.writerow(extracted_row)

    print(f"✓ Extracted {len(rows)} experiments")
    print(f"✓ Saved {len(available_columns)} columns to: {output_file}")
    print(f"\nColumns included:")
    for i, col in enumerate(available_columns, 1):
        print(f"  {i}. {col}")

if __name__ == "__main__":
    input_file = "data/experiments_log.csv"
    output_file = "data/experiments_summary.csv"

    print("Extracting experiment summary...")
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}\n")

    extract_summary(input_file, output_file)
