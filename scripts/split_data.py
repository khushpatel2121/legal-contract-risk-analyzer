"""
Split cuad_25_clauses.json into train/val/test by contract.
Ensures no contract appears in more than one split.
Saves to data/processed/train_raw.json, val_raw.json, test_raw.json
Run: python scripts/split_data.py
"""

import json
import random
from pathlib import Path
from collections import defaultdict

random.seed(42)

INPUT_PATH  = Path("data/processed/cuad_25_clauses.json")
OUTPUT_DIR  = Path("data/processed")

TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10


def main():
    # Load processed data
    print(f"Loading {INPUT_PATH}...")
    with open(INPUT_PATH) as f:
        data = json.load(f)

    records = data["data"]
    print(f"Total records: {len(records)}")

    # Group records by contract title
    contract_records = defaultdict(list)
    for record in records:
        contract_records[record["contract_title"]].append(record)

    print(f"Total unique contracts: {len(contract_records)}")

    # Shuffle contracts
    contracts = list(contract_records.keys())
    random.shuffle(contracts)

    # Split contracts
    total = len(contracts)
    train_end = int(total * TRAIN_RATIO)
    val_end   = int(total * (TRAIN_RATIO + VAL_RATIO))

    train_contracts = contracts[:train_end]
    val_contracts   = contracts[train_end:val_end]
    test_contracts  = contracts[val_end:]

    # Collect records per split
    train = [r for c in train_contracts for r in contract_records[c]]
    val   = [r for c in val_contracts   for r in contract_records[c]]
    test  = [r for c in test_contracts  for r in contract_records[c]]

    # Verify no overlap
    train_titles = set(r["contract_title"] for r in train)
    val_titles   = set(r["contract_title"] for r in val)
    test_titles  = set(r["contract_title"] for r in test)

    assert len(train_titles & val_titles)  == 0, "Train/Val overlap detected"
    assert len(train_titles & test_titles) == 0, "Train/Test overlap detected"
    assert len(val_titles   & test_titles) == 0, "Val/Test overlap detected"
    print("\n✓ No contract overlap between splits")

    # Print per split stats
    for split_name, split_records, split_contracts in [
        ("Train", train, train_contracts),
        ("Val",   val,   val_contracts),
        ("Test",  test,  test_contracts),
    ]:
        positives = sum(1 for r in split_records if r["is_positive"])
        negatives = sum(1 for r in split_records if not r["is_positive"])
        print(f"\n{split_name}:")
        print(f"  Contracts : {len(split_contracts)}")
        print(f"  Records   : {len(split_records)}")
        print(f"  Positives : {positives}")
        print(f"  Negatives : {negatives}")

    # Save splits
    for split_name, split_records in [
        ("train", train),
        ("val",   val),
        ("test",  test),
    ]:
        output = {
            "metadata": {
                "split": split_name,
                "total_records": len(split_records),
                "total_positives": sum(1 for r in split_records if r["is_positive"]),
                "total_negatives": sum(1 for r in split_records if not r["is_positive"]),
                "total_contracts": len(set(r["contract_title"] for r in split_records)),
            },
            "data": split_records,
        }
        path = OUTPUT_DIR / f"{split_name}_raw.json"
        with open(path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved {split_name}_raw.json → {len(split_records)} records")

    print("\n" + "=" * 50)
    print("  Split complete.")
    print(f"  Train : {len(train)} records ({len(train_contracts)} contracts)")
    print(f"  Val   : {len(val)} records ({len(val_contracts)} contracts)")
    print(f"  Test  : {len(test)} records ({len(test_contracts)} contracts)")
    print("=" * 50)


if __name__ == "__main__":
    main()