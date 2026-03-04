"""
Check per-clause distribution in train_raw.json
to inform augmentation strategy.
Run: python scripts/check_train_distribution.py
"""

import json
from pathlib import Path
from collections import defaultdict

TRAIN_PATH = Path("data/processed/train_raw.json")
MIN_THRESHOLD = 80  # flag clauses below this


def main():
    with open(TRAIN_PATH) as f:
        data = json.load(f)

    records = data["data"]

    # Count positives per clause in training split
    clause_counts = defaultdict(int)
    for record in records:
        if record["is_positive"]:
            clause_counts[record["clause_type"]] += 1

    # Rank by count
    ranked = sorted(clause_counts.items(), key=lambda x: x[1], reverse=True)

    print(f"\n{'Clause Type':<40} {'Train Positives':<18} {'Augmentation Needed'}")
    print(f"{'-'*39} {'-'*17} {'-'*20}")

    for clause, count in ranked:
        if count >= 200:
            action = "None"
        elif count >= 100:
            action = "Oversample x2"
        elif count >= 60:
            action = "Oversample x3"
        else:
            action = "Oversample x3 + back translation"

        flag = " ⚠️" if count < MIN_THRESHOLD else ""
        print(f"{clause:<40} {count:<18} {action}{flag}")

    print(f"\nTotal training positives: {sum(clause_counts.values())}")


if __name__ == "__main__":
    main()