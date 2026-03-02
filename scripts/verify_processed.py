"""
Verify the quality of data/processed/cuad_25_clauses.json
Checks for missing text, short/long spans, garbled text, and anomalies.
Run: python scripts/verify_processed.py
"""

import json
import re
from pathlib import Path
from collections import defaultdict

DATA_PATH = Path("data/processed/cuad_25_clauses.json")

# Thresholds
MIN_ANSWER_LENGTH = 10       # chars — below this is suspiciously short
MAX_ANSWER_LENGTH = 3000     # chars — above this may cause token issues
TRUNCATION_PATTERN = re.compile(r'\b\w+$')  # ends mid-word


def section(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def load_data():
    print(f"Loading {DATA_PATH}...")
    with open(DATA_PATH) as f:
        data = json.load(f)
    print(f"Total records    : {data['metadata']['total_records']}")
    print(f"Total positives  : {data['metadata']['total_positives']}")
    print(f"Total negatives  : {data['metadata']['total_negatives']}")
    print(f"Overall ratio    : {data['metadata']['ratio']} : 1")
    return data["data"]


def check_missing_answer_text(records):
    section("1. MISSING ANSWER TEXT (positives with empty answer)")

    issues = []
    for r in records:
        if r["is_positive"] and not r["answer_text"].strip():
            issues.append(r)

    if not issues:
        print("  ✓ No missing answer text found")
    else:
        print(f"  ✗ Found {len(issues)} positives with empty answer text:")
        for r in issues[:5]:  # show first 5
            print(f"    - {r['clause_type']} | {r['contract_title'][:50]}")
        if len(issues) > 5:
            print(f"    ... and {len(issues) - 5} more")

    return issues


def check_short_answer_text(records):
    section(f"2. VERY SHORT ANSWER TEXT (positives under {MIN_ANSWER_LENGTH} chars)")

    issues = []
    for r in records:
        if r["is_positive"] and 0 < len(r["answer_text"].strip()) < MIN_ANSWER_LENGTH:
            issues.append(r)

    if not issues:
        print(f"  ✓ No answer text under {MIN_ANSWER_LENGTH} chars found")
    else:
        print(f"  ✗ Found {len(issues)} suspiciously short answers:")
        for r in issues[:5]:
            print(f"    - [{len(r['answer_text'])} chars] '{r['answer_text']}' | {r['clause_type']}")
        if len(issues) > 5:
            print(f"    ... and {len(issues) - 5} more")

    return issues


def check_long_answer_text(records):
    section(f"3. VERY LONG ANSWER TEXT (positives over {MAX_ANSWER_LENGTH} chars)")

    issues = []
    for r in records:
        if r["is_positive"] and len(r["answer_text"]) > MAX_ANSWER_LENGTH:
            issues.append(r)

    if not issues:
        print(f"  ✓ No answer text over {MAX_ANSWER_LENGTH} chars found")
    else:
        print(f"  ⚠️  Found {len(issues)} very long answers (may hit token limits):")
        for r in issues[:5]:
            print(f"    - [{len(r['answer_text'])} chars] {r['clause_type']} | {r['contract_title'][:50]}")
        if len(issues) > 5:
            print(f"    ... and {len(issues) - 5} more")

        # Show length distribution
        lengths = [len(r["answer_text"]) for r in issues]
        print(f"\n    Max length  : {max(lengths)} chars")
        print(f"    Avg length  : {sum(lengths) // len(lengths)} chars")

    return issues


def check_garbled_text(records):
    section("4. GARBLED TEXT (excessive whitespace or artifacts)")

    issues = []
    for r in records:
        if not r["is_positive"]:
            continue
        text = r["answer_text"]
        # Check for excessive whitespace (4+ consecutive spaces)
        if re.search(r' {4,}', text):
            issues.append(("excessive whitespace", r))
        # Check for excessive newlines
        elif text.count('\n') > 10:
            issues.append(("excessive newlines", r))
        # Check for non-ASCII characters
        elif re.search(r'[^\x00-\x7F]{3,}', text):
            issues.append(("non-ASCII chars", r))

    if not issues:
        print("  ✓ No garbled text found")
    else:
        print(f"  ⚠️  Found {len(issues)} records with text artifacts:")
        for issue_type, r in issues[:5]:
            print(f"    - [{issue_type}] {r['clause_type']} | {r['contract_title'][:50]}")
        if len(issues) > 5:
            print(f"    ... and {len(issues) - 5} more")

    return issues


def check_negative_with_text(records):
    section("5. NEGATIVES WITH NON-EMPTY ANSWER TEXT (should never happen)")

    issues = []
    for r in records:
        if not r["is_positive"] and r["answer_text"].strip():
            issues.append(r)

    if not issues:
        print("  ✓ All negatives have empty answer text")
    else:
        print(f"  ✗ Found {len(issues)} negatives with non-empty answer text:")
        for r in issues[:5]:
            print(f"    - {r['clause_type']} | '{r['answer_text'][:50]}'")

    return issues


def check_per_clause_health(records):
    section("6. PER CLAUSE HEALTH CHECK")

    stats = defaultdict(lambda: {"positives": 0, "negatives": 0})
    for r in records:
        clause = r["clause_type"]
        if r["is_positive"]:
            stats[clause]["positives"] += 1
        else:
            stats[clause]["negatives"] += 1

    print(f"\n  {'Clause Type':<40} {'Pos':<8} {'Neg':<8} {'Ratio':<8} {'Status'}")
    print(f"  {'-'*39} {'-'*7} {'-'*7} {'-'*7} {'-'*10}")

    for clause, s in sorted(stats.items(), key=lambda x: x[1]["positives"], reverse=True):
        p = s["positives"]
        n = s["negatives"]
        ratio = round(p / n, 1) if n > 0 else "N/A"

        # Flag clauses where negatives exceed positives
        if n == 0:
            status = "⚠️  no negatives"
        elif p == 0:
            status = "✗  no positives"
        elif n > p:
            status = "⚠️  neg > pos"
        else:
            status = "✓"

        print(f"  {clause:<40} {p:<8} {n:<8} {str(ratio):<8} {status}")


def check_answer_length_distribution(records):
    section("7. ANSWER LENGTH DISTRIBUTION (positives)")

    lengths = [len(r["answer_text"]) for r in records if r["is_positive"]]

    if not lengths:
        print("  No positive records found")
        return

    lengths.sort()
    n = len(lengths)

    print(f"  Total positives    : {n}")
    print(f"  Min length (chars) : {lengths[0]}")
    print(f"  Max length (chars) : {lengths[-1]}")
    print(f"  Avg length (chars) : {sum(lengths) // n}")
    print(f"  Median (chars)     : {lengths[n // 2]}")

    # Bucket distribution
    buckets = [
        (0, 100, "0-100"),
        (100, 300, "100-300"),
        (300, 600, "300-600"),
        (600, 1000, "600-1000"),
        (1000, 2000, "1000-2000"),
        (2000, float("inf"), "2000+"),
    ]

    print(f"\n  Length distribution:")
    for low, high, label in buckets:
        count = sum(1 for l in lengths if low <= l < high)
        pct = (count / n) * 100
        bar = "█" * int(pct / 2)
        print(f"    {label:<12} : {count:<6} ({pct:.1f}%) {bar}")


def main():
    records = load_data()

    check_missing_answer_text(records)
    check_short_answer_text(records)
    check_long_answer_text(records)
    check_garbled_text(records)
    check_negative_with_text(records)
    check_per_clause_health(records)
    check_answer_length_distribution(records)

    print("\n" + "=" * 60)
    print("  Verification complete.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()