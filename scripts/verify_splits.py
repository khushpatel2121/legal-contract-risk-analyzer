"""
Verify train/val/test splits for data leakage.
Checks contract-level isolation and data integrity.
Run: python3 scripts/verify_splits.py
"""

import json
from pathlib import Path
from collections import defaultdict

TRAIN_PATH = Path("data/processed/train_augmented.json")
VAL_PATH   = Path("data/processed/val_raw.json")
TEST_PATH  = Path("data/processed/test_raw.json")


def section(title: str):
    print("\\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def load_split(path: Path) -> list:
    with open(path) as f:
        data = json.load(f)
    return data["data"]


def get_contracts(records: list) -> set:
    return set(r["contract_title"] for r in records)


def get_answer_texts(records: list) -> set:
    return set(r["answer_text"] for r in records if r["is_positive"])


def main():
    print("Loading splits...")
    train = load_split(TRAIN_PATH)
    val   = load_split(VAL_PATH)
    test  = load_split(TEST_PATH)

    train_contracts = get_contracts(train)
    val_contracts   = get_contracts(val)
    test_contracts  = get_contracts(test)

    # ─────────────────────────────────────────
    section("1. CONTRACT-LEVEL ISOLATION")
    # ─────────────────────────────────────────

    train_val_overlap  = train_contracts & val_contracts
    train_test_overlap = train_contracts & test_contracts
    val_test_overlap   = val_contracts   & test_contracts

    if not train_val_overlap:
        print("  ✓ No contract overlap between Train and Val")
    else:
        print(f"  ✗ {len(train_val_overlap)} contracts overlap Train/Val:")
        for c in list(train_val_overlap)[:3]:
            print(f"    - {c}")

    if not train_test_overlap:
        print("  ✓ No contract overlap between Train and Test")
    else:
        print(f"  ✗ {len(train_test_overlap)} contracts overlap Train/Test:")
        for c in list(train_test_overlap)[:3]:
            print(f"    - {c}")

    if not val_test_overlap:
        print("  ✓ No contract overlap between Val and Test")
    else:
        print(f"  ✗ {len(val_test_overlap)} contracts overlap Val/Test:")
        for c in list(val_test_overlap)[:3]:
            print(f"    - {c}")

    # ─────────────────────────────────────────
    section("2. ANSWER TEXT LEAKAGE CHECK")
    # ─────────────────────────────────────────

    # Check if any exact answer texts appear in multiple splits
    # This would indicate data leakage at record level
    train_answers = get_answer_texts(train)
    val_answers   = get_answer_texts(val)
    test_answers  = get_answer_texts(test)

    train_val_answer_overlap  = train_answers & val_answers
    train_test_answer_overlap = train_answers & test_answers

    if not train_val_answer_overlap:
        print("  ✓ No answer text overlap between Train and Val")
    else:
        print(f"  ⚠️  {len(train_val_answer_overlap)} answer texts appear "
              f"in both Train and Val")
        print(f"     (expected if same clause appears in different contracts)")

    if not train_test_answer_overlap:
        print("  ✓ No answer text overlap between Train and Test")
    else:
        print(f"  ⚠️  {len(train_test_answer_overlap)} answer texts appear "
              f"in both Train and Test")
        print(f"     (expected if same clause appears in different contracts)")

    # ─────────────────────────────────────────
    section("3. SPLIT SIZE VERIFICATION")
    # ─────────────────────────────────────────

    total = len(train) + len(val) + len(test)
    print(f"  Train records    : {len(train)}")
    print(f"  Val records      : {len(val)}")
    print(f"  Test records     : {len(test)}")
    print(f"  Total            : {total}")
    print(f"\\n  Train contracts  : {len(train_contracts)}")
    print(f"  Val contracts    : {len(val_contracts)}")
    print(f"  Test contracts   : {len(test_contracts)}")
    print(f"  Total contracts  : "
          f"{len(train_contracts | val_contracts | test_contracts)}")

    # ─────────────────────────────────────────
    section("4. CLAUSE DISTRIBUTION ACROSS SPLITS")
    # ─────────────────────────────────────────

    def clause_counts(records):
        counts = defaultdict(int)
        for r in records:
            if r["is_positive"]:
                counts[r["clause_type"]] += 1
        return counts

    train_counts = clause_counts(train)
    val_counts   = clause_counts(val)
    test_counts  = clause_counts(test)

    all_clauses = sorted(train_counts.keys())

    print(f"\\n  {'Clause Type':<40} {'Train':<8} {'Val':<8} {'Test'}")
    print(f"  {'-'*39} {'-'*7} {'-'*7} {'-'*7}")

    for clause in all_clauses:
        t = train_counts[clause]
        v = val_counts.get(clause, 0)
        te = test_counts.get(clause, 0)
        # Flag if clause missing from val or test
        flag = " ⚠️" if v == 0 or te == 0 else ""
        print(f"  {clause:<40} {t:<8} {v:<8} {te}{flag}")

    # ─────────────────────────────────────────
    section("5. POSITIVE/NEGATIVE RATIO PER SPLIT")
    # ─────────────────────────────────────────

    for split_name, split_records in [
        ("Train", train),
        ("Val",   val),
        ("Test",  test)
    ]:
        pos = sum(1 for r in split_records if r["is_positive"])
        neg = sum(1 for r in split_records if not r["is_positive"])
        ratio = round(pos / neg, 2) if neg > 0 else "N/A"
        print(f"\\n  {split_name}:")
        print(f"    Positives : {pos}")
        print(f"    Negatives : {neg}")
        print(f"    Ratio     : {ratio}")

    print("\\n" + "=" * 60)
    print("  Verification complete.")
    print("=" * 60 + "\\n")


if __name__ == "__main__":
    main()