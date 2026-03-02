"""
Filter CUAD dataset to only the 25 selected clause types.
Downsamples negatives per clause to maintain 2:1 positive/negative ratio.
Saves result to data/processed/cuad_25_clauses.json
Run: python scripts/filter_clauses.py
"""

import json
import random
from pathlib import Path
from collections import defaultdict
import re

random.seed(42)

DATA_PATH = Path("data/raw/CUAD_v1.json")
OUTPUT_PATH = Path("data/processed/cuad_25_clauses.json")

SELECTED_CLAUSES = [
    # High frequency
    "Governing Law",
    "Anti-Assignment",
    # Medium-High frequency
    "Cap On Liability",
    "License Grant",
    "Audit Rights",
    "Termination For Convenience",
    "Exclusivity",
    "Renewal Term",
    "Insurance",
    "Minimum Commitment",
    # Medium-Low frequency
    "Ip Ownership Assignment",
    "Change Of Control",
    "Non-Compete",
    "Uncapped Liability",
    "Notice Period To Terminate Renewal",
    "Covenant Not To Sue",
    "Rofr/Rofo/Rofn",
    "Warranty Duration",
    "Liquidated Damages",
    # Low frequency (sparse)
    "Irrevocable Or Perpetual License",
    "No-Solicit Of Employees",
    "Affiliate License-Licensee",
    "Joint Ip Ownership",
    "Non-Disparagement",
    "No-Solicit Of Customers",
]

# Normalize to lowercase for matching
SELECTED_CLAUSES_LOWER = {c.lower(): c for c in SELECTED_CLAUSES}


def extract_clause_type(qa_id: str) -> str:
    """Extract clause type from QA id (everything after __)"""
    return qa_id.split("__")[-1]


def join_answer_spans(answers: list) -> str:
    """Join multiple answer spans into one string."""
    return " [...] ".join(a["text"].strip() for a in answers)

def normalize_whitespace(text: str) -> str:
    """Collapse excessive whitespace and remove non-ASCII characters."""
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)   # remove non-ASCII
    text = re.sub(r'\s+', ' ', text)               # collapse whitespace
    return text.strip()


def truncate_at_sentence(text: str, max_chars: int = 1500) -> str:
    """Truncate text at nearest sentence boundary before max_chars."""
    if len(text) <= max_chars:
        return text
    # Find the last sentence boundary before max_chars
    truncated = text[:max_chars]
    last_period = truncated.rfind('.')
    last_semicolon = truncated.rfind(';')
    cut_point = max(last_period, last_semicolon)
    if cut_point == -1:
        # No sentence boundary found — cut at last space
        cut_point = truncated.rfind(' ')
    if cut_point == -1:
        # No space found either — hard cut
        return truncated
    return text[:cut_point + 1].strip()


def clean_answer_text(text: str) -> str:
    """Full cleaning pipeline for answer text."""
    text = normalize_whitespace(text)
    text = truncate_at_sentence(text)
    return text


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Loading CUAD_v1.json...")
    with open(DATA_PATH) as f:
        data = json.load(f)

    contracts = data["data"]
    print(f"Total contracts loaded: {len(contracts)}")

    # Separate positives and negatives per clause
    positives = defaultdict(list)
    negatives = defaultdict(list)

    for contract in contracts:
        title = contract["title"]
        for paragraph in contract["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                clause_type = extract_clause_type(qa["id"])

                if clause_type.lower() not in SELECTED_CLAUSES_LOWER:
                    continue

                canonical_clause = SELECTED_CLAUSES_LOWER[clause_type.lower()]

                record = {
                    "contract_title": title,
                    "clause_type": canonical_clause,
                    "is_positive": not qa["is_impossible"],
                    "answer_text": clean_answer_text(join_answer_spans(qa["answers"])) if not qa["is_impossible"] else "",
                    "context": context,
                }

                if not qa["is_impossible"]:
                    positives[canonical_clause].append(record)
                else:
                    negatives[canonical_clause].append(record)

    # Downsample negatives per clause
    filtered = []
    stats = defaultdict(lambda: {"positives": 0, "negatives": 0, "note": ""})

    for clause in SELECTED_CLAUSES:
        pos_records = positives[clause]
        neg_records = negatives[clause]

        target_negatives = len(pos_records) // 2

        # Keep all if not enough, otherwise downsample
        if len(neg_records) <= target_negatives:
            sampled_negatives = neg_records
            note = "kept all (below threshold)"
        else:
            sampled_negatives = random.sample(neg_records, target_negatives)
            note = "downsampled"

        filtered.extend(pos_records)
        filtered.extend(sampled_negatives)

        stats[clause]["positives"] = len(pos_records)
        stats[clause]["negatives"] = len(sampled_negatives)
        stats[clause]["note"] = note

    # Shuffle final dataset
    random.shuffle(filtered)

    total_positives = sum(s["positives"] for s in stats.values())
    total_negatives = sum(s["negatives"] for s in stats.values())

    output = {
        "metadata": {
            "total_records": len(filtered),
            "total_positives": total_positives,
            "total_negatives": total_negatives,
            "ratio": round(total_positives / total_negatives, 2),
            "selected_clauses": SELECTED_CLAUSES,
        },
        "data": filtered,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved {len(filtered)} records to {OUTPUT_PATH}")

    print(f"\n{'Clause Type':<40} {'Positives':<12} {'Negatives':<12} {'Ratio':<8} {'Note'}")
    print(f"{'-'*39} {'-'*11} {'-'*11} {'-'*7} {'-'*20}")
    for clause in SELECTED_CLAUSES:
        p = stats[clause]["positives"]
        n = stats[clause]["negatives"]
        ratio = round(p / n, 1) if n > 0 else "N/A"
        note = stats[clause]["note"]
        print(f"{clause:<40} {p:<12} {n:<12} {str(ratio):<8} {note}")

    print(f"\nTotal positives : {total_positives}")
    print(f"Total negatives : {total_negatives}")
    print(f"Overall ratio   : {output['metadata']['ratio']} : 1")


if __name__ == "__main__":
    main()