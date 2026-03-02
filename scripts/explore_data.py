"""
Exploration script for CUAD_v1.json.
Analyzes the dataset across 6 metric categories before preprocessing.
Run: python scripts/explore_data.py
"""

import json
from pathlib import Path
from collections import defaultdict
import statistics

DATA_PATH = Path("/Users/khushpatel/Desktop/ML Projects/legal-contract-risk-analyzer/data/processed/cuad_25_clauses.json")
MIN_POSITIVE_THRESHOLD = 50  # flag clause types below this


def extract_clause_type(qa_id: str) -> str:
    """Extract clause type from QA id field (everything after __)"""
    return qa_id.split("__")[-1]


def load_data():
    print("Loading CUAD_v1.json...")
    with open(DATA_PATH) as f:
        data = json.load(f)
    return data["data"]


def section(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def analyze_dataset_level(contracts):
    section("1. DATASET-LEVEL STATS")

    total_contracts = len(contracts)
    total_qa = 0
    total_positives = 0
    total_negatives = 0

    for contract in contracts:
        for paragraph in contract["paragraphs"]:
            for qa in paragraph["qas"]:
                total_qa += 1
                if not qa["is_impossible"]:
                    total_positives += 1
                else:
                    total_negatives += 1

    ratio = total_positives / total_negatives if total_negatives > 0 else 0

    print(f"  Total contracts       : {total_contracts}")
    print(f"  Total QA pairs        : {total_qa}")
    print(f"  Total positives       : {total_positives}")
    print(f"  Total negatives       : {total_negatives}")
    print(f"  Positive/Negative ratio: {ratio:.2f}")

    return total_positives, total_negatives


def analyze_clause_level(contracts):
    section("2. CLAUSE-LEVEL STATS")

    clause_positives = defaultdict(int)
    clause_negatives = defaultdict(int)

    for contract in contracts:
        for paragraph in contract["paragraphs"]:
            for qa in paragraph["qas"]:
                clause = extract_clause_type(qa["id"])
                if not qa["is_impossible"]:
                    clause_positives[clause] += 1
                else:
                    clause_negatives[clause] += 1

    zero_positive_clauses = [c for c in clause_positives if clause_positives[c] == 0]
    below_threshold = [c for c, count in clause_positives.items() if count < MIN_POSITIVE_THRESHOLD]

    print(f"  Total clause types    : {len(clause_positives)}")
    print(f"  Clause types with 0 positives  : {len(zero_positive_clauses)}")
    if zero_positive_clauses:
        for c in zero_positive_clauses:
            print(f"    - {c}")

    print(f"\n  Clause types below {MIN_POSITIVE_THRESHOLD} positives:")
    if below_threshold:
        for c in below_threshold:
            print(f"    - {c} ({clause_positives[c]} positives)")
    else:
        print("    None")

    return clause_positives, clause_negatives


def analyze_answer_spans(contracts):
    section("3. ANSWER SPAN STATS")

    span_lengths = []
    multi_answer_counts = defaultdict(int)  # how many QAs have 1, 2, 3+ answers

    for contract in contracts:
        for paragraph in contract["paragraphs"]:
            for qa in paragraph["qas"]:
                if not qa["is_impossible"] and qa["answers"]:
                    num_answers = len(qa["answers"])
                    multi_answer_counts[num_answers] += 1

                    # join multiple spans
                    combined = " ".join(a["text"] for a in qa["answers"])
                    span_lengths.append(len(combined))

    if span_lengths:
        print(f"  Min span length (chars)   : {min(span_lengths)}")
        print(f"  Max span length (chars)   : {max(span_lengths)}")
        print(f"  Average span length       : {statistics.mean(span_lengths):.1f}")
        print(f"  Median span length        : {statistics.median(span_lengths):.1f}")

        print(f"\n  Answer span distribution:")
        for count, freq in sorted(multi_answer_counts.items()):
            print(f"    {count} answer span(s)  : {freq} QA pairs")


def analyze_contract_level(contracts):
    section("4. CONTRACT-LEVEL STATS")

    positives_per_contract = []
    zero_positive_contracts = 0

    for contract in contracts:
        count = 0
        for paragraph in contract["paragraphs"]:
            for qa in paragraph["qas"]:
                if not qa["is_impossible"]:
                    count += 1
        positives_per_contract.append(count)
        if count == 0:
            zero_positive_contracts += 1

    print(f"  Contracts with 0 positives : {zero_positive_contracts}")
    print(f"  Min positives per contract : {min(positives_per_contract)}")
    print(f"  Max positives per contract : {max(positives_per_contract)}")
    print(f"  Avg positives per contract : {statistics.mean(positives_per_contract):.1f}")
    print(f"  Median positives/contract  : {statistics.median(positives_per_contract):.1f}")


def analyze_negative_sampling(total_positives, total_negatives):
    section("5. NEGATIVE SAMPLING GUIDANCE")

    target_negatives = total_positives // 2  # 2:1 ratio
    print(f"  Total positives           : {total_positives}")
    print(f"  Target negatives (2:1)    : {target_negatives}")
    print(f"  Available negatives       : {total_negatives}")

    if total_negatives >= target_negatives:
        print(f"  Status                    : ✓ Enough negatives to sample from")
    else:
        print(f"  Status                    : ✗ Not enough negatives — adjust ratio")

    print(f"\n  Final estimated dataset size : {total_positives + target_negatives}")


def analyze_clause_frequency_ranking(contracts, clause_positives):
    section("6. CLAUSE FREQUENCY RANKING (most → least)")

    total_contracts = len(contracts)
    ranked = sorted(clause_positives.items(), key=lambda x: x[1], reverse=True)

    print(f"  {'Rank':<5} {'Clause Type':<40} {'Positives':<12} {'% Contracts'}")
    print(f"  {'-'*4} {'-'*39} {'-'*11} {'-'*12}")

    for rank, (clause, count) in enumerate(ranked, 1):
        pct = (count / total_contracts) * 100
        flag = " ⚠️ " if count < MIN_POSITIVE_THRESHOLD else ""
        print(f"  {rank:<5} {clause:<40} {count:<12} {pct:.1f}%{flag}")


def main():
    contracts = load_data()

    total_positives, total_negatives = analyze_dataset_level(contracts)
    clause_positives, clause_negatives = analyze_clause_level(contracts)
    analyze_answer_spans(contracts)
    analyze_contract_level(contracts)
    analyze_negative_sampling(total_positives, total_negatives)
    analyze_clause_frequency_ranking(contracts, clause_positives)

    print("\n" + "=" * 60)
    print("  Exploration complete.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()