"""
Augment training data for sparse clause types.
Strategy:
  150-200 positives → oversample x1.5
  80-150  positives → oversample x2
  <80     positives → oversample x2.5
  <60     positives → back translation + oversample x2.5
Saves augmented training data to data/processed/train_augmented.json
Run: python scripts/augment_data.py
"""

import json
import random
from pathlib import Path
from collections import defaultdict
from transformers import pipeline

random.seed(42)

INPUT_PATH  = Path("data/processed/train_raw.json")
OUTPUT_PATH = Path("data/processed/train_augmented.json")

# Augmentation thresholds
TIER_1_MIN = 150   # 150-200 → x1.5
TIER_1_MAX = 200
TIER_2_MIN = 80    # 80-150  → x2
TIER_2_MAX = 150
TIER_3_MAX = 80    # <80     → x2.5
BT_THRESHOLD = 60  # <60     → back translation too


def load_translation_models():
    """Load Helsinki-NLP translation models."""
    print("Loading translation models...")
    en_es = pipeline("translation_en_to_es", model="Helsinki-NLP/opus-mt-en-es")
    es_en = pipeline("translation_es_to_en", model="Helsinki-NLP/opus-mt-es-en")
    print("✓ Translation models loaded")
    return en_es, es_en


def back_translate(text: str, en_es, es_en) -> str:
    """
    Translate English → Spanish → English.
    Returns paraphrased version of the input text.
    Falls back to original if translation fails.
    """
    try:
        # Truncate to 512 chars — translation models have token limits
        truncated = text[:512]
        spanish = en_es(truncated)[0]["translation_text"]
        back    = es_en(spanish)[0]["translation_text"]

        # Only return if meaningfully different
        if back.lower().strip() != text.lower().strip():
            return back
        return text

    except Exception as e:
        print(f"  ⚠️  Back translation failed: {e}")
        return text


def get_augmentation_tier(count: int) -> str:
    """Return augmentation tier based on positive count."""
    if count >= TIER_1_MAX:
        return "none"
    elif count >= TIER_1_MIN:
        return "tier1"   # x1.5
    elif count >= TIER_2_MIN:
        return "tier2"   # x2
    else:
        return "tier3"   # x2.5 (+ back translation if < 60)


def oversample(records: list, target_count: int) -> list:
    """
    Oversample records to reach target count.
    Samples without replacement when possible.
    """
    current = len(records)
    needed  = int(target_count) - current

    if needed <= 0:
        return records

    # Sample needed records from existing ones
    if needed <= current:
        sampled = random.sample(records, needed)
    else:
        # Need more than one full copy
        sampled = []
        while len(sampled) < needed:
            remaining = needed - len(sampled)
            batch = random.sample(records, min(remaining, current))
            sampled.extend(batch)

    # Mark as augmented
    for r in sampled:
        r = r.copy()
        r["augmented"] = True

    return records + sampled


def augment_with_back_translation(
    records: list, en_es, es_en
) -> list:
    """
    Generate one back-translated copy of each record.
    Returns original records + back-translated copies.
    """
    bt_records = []
    print(f"    Running back translation on {len(records)} records...")

    for i, record in enumerate(records):
        if i % 10 == 0:
            print(f"    Progress: {i}/{len(records)}")

        bt_text = back_translate(record["answer_text"], en_es, es_en)

        # Only add if back translation produced different text
        if bt_text != record["answer_text"]:
            bt_record = record.copy()
            bt_record["answer_text"] = bt_text
            bt_record["augmented"]   = True
            bt_record["aug_method"]  = "back_translation"
            bt_records.append(bt_record)

    print(f"    ✓ Generated {len(bt_records)} back-translated records")
    return records + bt_records


def main():
    # Load training data
    print(f"Loading {INPUT_PATH}...")
    with open(INPUT_PATH) as f:
        data = json.load(f)

    records = data["data"]
    print(f"Total training records: {len(records)}")

    # Separate positives and negatives
    # We only augment positives — negatives stay as is
    positives = defaultdict(list)
    negatives = []

    for record in records:
        if record["is_positive"]:
            positives[record["clause_type"]].append(record)
        else:
            negatives.append(record)

    print(f"Positives: {sum(len(v) for v in positives.values())}")
    print(f"Negatives: {len(negatives)}")

    # Identify which clauses need back translation
    bt_clauses = [
        clause for clause, recs in positives.items()
        if len(recs) < BT_THRESHOLD
    ]

    print(f"\nClauses needing back translation: {len(bt_clauses)}")
    for c in bt_clauses:
        print(f"  - {c} ({len(positives[c])} examples)")

    # Load translation models only if needed
    en_es, es_en = None, None
    if bt_clauses:
        en_es, es_en = load_translation_models()

    # Augment per clause
    print("\nAugmenting clauses...")
    augmented_positives = []

    for clause, clause_records in positives.items():
        count = len(clause_records)
        tier  = get_augmentation_tier(count)

        if tier == "none":
            augmented_positives.extend(clause_records)
            print(f"  {clause:<40} {count} → {count} (no augmentation)")
            continue

        # Apply back translation first for sparse clauses
        working_records = clause_records.copy()
        if count < BT_THRESHOLD and en_es is not None:
            print(f"  {clause:<40} applying back translation...")
            working_records = augment_with_back_translation(
                working_records, en_es, es_en
            )

        # Calculate target count based on tier
        if tier == "tier1":
            target = int(count * 1.5)
        elif tier == "tier2":
            target = count * 1.5
        else:  # tier3
            target = int(count * 2)

        # Oversample to target
        final_records = oversample(working_records, target)
        augmented_positives.extend(final_records)

        print(f"  {clause:<40} {count} → {len(final_records)} ({tier})")

    # Combine augmented positives with original negatives
    all_records = augmented_positives + negatives
    random.shuffle(all_records)

    total_positives = sum(1 for r in all_records if r["is_positive"])
    total_negatives = sum(1 for r in all_records if not r["is_positive"])

    # Save augmented training data
    output = {
        "metadata": {
            "total_records"  : len(all_records),
            "total_positives": total_positives,
            "total_negatives": total_negatives,
            "augmented"      : True,
        },
        "data": all_records,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*50}")
    print(f"  Augmentation complete.")
    print(f"  Original positives  : 2859")
    print(f"  Augmented positives : {total_positives}")
    print(f"  Negatives           : {total_negatives}")
    print(f"  Total records       : {len(all_records)}")
    print(f"  Saved → {OUTPUT_PATH}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()