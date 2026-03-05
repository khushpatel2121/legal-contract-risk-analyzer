"""
Build Phi-3 training instructions from processed CUAD data.
Reads train_augmented.json, val_raw.json, test_raw.json
Produces train.jsonl, val.jsonl, test.jsonl
Run: python3 scripts/build_training_data.py
"""

import json
import random
from pathlib import Path
import re 

random.seed(42)

# --- Paths ---
TRAIN_INPUT = Path("data/processed/train_augmented.json")
VAL_INPUT   = Path("data/processed/val_raw.json")
TEST_INPUT  = Path("data/processed/test_raw.json")
OUTPUT_DIR  = Path("data/processed")

# --- Constants ---
WINDOW_SIZE = 1500
PADDING     = 500

SYSTEM_PROMPT = (
    "You are a legal contract analyst. Analyze the contract text "
    "and detect if the specified clause is present. "
    "If present, extract the relevant text and explain it in plain English. "
    "If not present, state 'NOT PRESENT'."
)

def clean_text(text: str) -> str:
    """Normalize whitespace in extracted window."""
    text = re.sub(r'[^\\x00-\\x7F]+', ' ', text)  # remove non-ASCII
    text = re.sub(r'\\s+', ' ', text)              # collapse whitespace
    return text.strip()




# ============================================================
# SECTION 1 — Context Window Extraction
# ============================================================

def extract_window(context: str, answer_text: str, is_positive: bool) -> str:
    if not is_positive:
        if len(context) <= WINDOW_SIZE:
            return context
        max_start = len(context) - WINDOW_SIZE
        start = random.randint(0, max_start)
        # Snap to nearest sentence boundary
        boundary = context.find('.', start)
        if boundary != -1 and boundary < start + 200:
            start = boundary + 1
        return clean_text(context[start:start + WINDOW_SIZE])

    start = context.find(answer_text)
    if start == -1:
        return context[:WINDOW_SIZE]

    end          = start + len(answer_text)
    window_start = max(0, start - PADDING)
    window_end   = min(len(context), end + PADDING)

    # Snap window_start to nearest sentence boundary
    boundary = context.find('.', window_start)
    if boundary != -1 and boundary < start:
        window_start = boundary + 1

    # Snap window_end to nearest sentence boundary
    boundary = context.rfind('.', end, window_end)
    if boundary != -1:
        window_end = boundary + 1

    return clean_text(context[window_start:window_end])


# ============================================================
# SECTION 2 — Explanation Templates
# ============================================================

def get_explanation(clause_type: str, answer_text: str) -> str:
    templates = {
        "Governing Law": (
            f"This clause establishes which jurisdiction's laws govern "
            f"the contract. Specifically: \"{answer_text}\". This determines "
            f"which courts have authority over disputes and which legal system "
            f"applies. Pay close attention if the specified jurisdiction is "
            f"different from where you or your business operate, as it may "
            f"require hiring local legal counsel."
        ),
        "Anti-Assignment": (
            f"This clause restricts the ability to transfer the contract "
            f"to another party. Specifically: \"{answer_text}\". This means "
            f"one or both parties cannot assign their rights or obligations "
            f"under this agreement without prior consent. This is important "
            f"in scenarios like mergers, acquisitions, or business "
            f"restructuring where contracts may need to transfer."
        ),
        "Cap On Liability": (
            f"This clause limits the maximum financial exposure of one "
            f"or both parties in case of a breach. Specifically: "
            f"\"{answer_text}\". This sets a ceiling on damages that can "
            f"be claimed, protecting parties from unlimited financial "
            f"risk. Review whether the cap amount is reasonable relative "
            f"to the value of the contract."
        ),
        "License Grant": (
            f"This clause defines the rights granted by one party to "
            f"another to use intellectual property, technology, or other "
            f"assets. Specifically: \"{answer_text}\". Pay attention to "
            f"whether the license is exclusive or non-exclusive, whether "
            f"it is transferable, and any restrictions on how the licensed "
            f"material can be used."
        ),
        "Audit Rights": (
            f"This clause grants one party the right to inspect the books, "
            f"records, or operations of the other party to verify compliance "
            f"with the contract. Specifically: \"{answer_text}\". Consider "
            f"how frequently audits can occur, who bears the cost, and what "
            f"records are subject to inspection."
        ),
        "Termination For Convenience": (
            f"This clause allows one or both parties to terminate the "
            f"contract without needing a specific cause or breach. "
            f"Specifically: \"{answer_text}\". Either party can exit the "
            f"agreement simply by providing notice. This means there is no "
            f"guaranteed contract duration — the other party could walk "
            f"away at any time with proper notice."
        ),
        "Exclusivity": (
            f"This clause creates an exclusive dealing arrangement between "
            f"the parties. Specifically: \"{answer_text}\". This may prevent "
            f"one party from working with competitors or require purchases "
            f"exclusively from the other party. Pay close attention to the "
            f"scope, territory, and duration of the exclusivity obligation."
        ),
        "Renewal Term": (
            f"This clause governs what happens when the initial contract "
            f"term expires. Specifically: \"{answer_text}\". The contract "
            f"may renew automatically or require active steps to extend. "
            f"Pay attention to whether renewal is automatic, the notice "
            f"period required to prevent renewal, and whether terms change "
            f"upon renewal."
        ),
        "Insurance": (
            f"This clause requires one or both parties to maintain specific "
            f"insurance coverage. Specifically: \"{answer_text}\". This "
            f"protects both parties in case of damages, liability claims, "
            f"or unforeseen events. Review the type of insurance required, "
            f"the minimum coverage amounts, and whether you must be named "
            f"as an additional insured."
        ),
        "Minimum Commitment": (
            f"This clause establishes a minimum purchase, usage, or "
            f"performance obligation. Specifically: \"{answer_text}\". "
            f"One party is required to meet a baseline threshold — whether "
            f"in units, revenue, or activity — during a given period. "
            f"Failure to meet minimums may result in penalties or loss of "
            f"contract rights such as exclusivity."
        ),
        "Ip Ownership Assignment": (
            f"This clause determines who owns intellectual property created "
            f"during the course of the agreement. Specifically: "
            f"\"{answer_text}\". IP created by one party may automatically "
            f"transfer to the other. This is critical for contractors and "
            f"developers — any work product you create may legally belong "
            f"to the other party under this clause."
        ),
        "Change Of Control": (
            f"This clause addresses what happens if one party undergoes a "
            f"significant ownership change such as a merger or acquisition. "
            f"Specifically: \"{answer_text}\". The other party may have the "
            f"right to terminate, renegotiate, or require consent before "
            f"the agreement continues under new ownership. This is "
            f"particularly important in fast-growing companies that may "
            f"be acquisition targets."
        ),
        "Non-Compete": (
            f"This clause restricts a party from engaging in competitive "
            f"activities. Specifically: \"{answer_text}\". It limits your "
            f"ability to work with competitors, start a competing business, "
            f"or operate in certain markets or geographies. Pay close "
            f"attention to the duration, geographic scope, and definition "
            f"of competing activity — overly broad non-competes may be "
            f"unenforceable in some jurisdictions."
        ),
        "Uncapped Liability": (
            f"This clause leaves one or both parties exposed to unlimited "
            f"financial liability in case of a breach. Specifically: "
            f"\"{answer_text}\". Unlike a cap on liability clause, there "
            f"is no ceiling on damages that can be claimed. This represents "
            f"significant financial exposure and should be carefully "
            f"reviewed before signing."
        ),
        "Notice Period To Terminate Renewal": (
            f"This clause specifies the advance notice required to prevent "
            f"automatic contract renewal. Specifically: \"{answer_text}\". "
            f"If you fail to provide notice within this window the contract "
            f"will automatically renew, potentially locking you in for "
            f"another full term. Calendar this date well in advance."
        ),
        "Covenant Not To Sue": (
            f"This clause restricts a party from bringing legal action "
            f"against the other party in certain circumstances. "
            f"Specifically: \"{answer_text}\". This may limit your ability "
            f"to challenge the validity of intellectual property rights or "
            f"bring claims unrelated to the contract. Review what specific "
            f"legal actions are restricted and for how long."
        ),
        "Rofr/Rofo/Rofn": (
            f"This clause grants one party a preferential right before the "
            f"other party can deal with third parties. Specifically: "
            f"\"{answer_text}\". A right of first refusal means you must be "
            f"offered the deal before it goes to others. A right of first "
            f"offer means you get to make the first bid. A right of first "
            f"negotiation means you get priority in negotiations. These "
            f"rights can significantly affect business flexibility."
        ),
        "Warranty Duration": (
            f"This clause defines how long product or service warranties "
            f"remain in effect. Specifically: \"{answer_text}\". During the "
            f"warranty period the provider must repair or replace defective "
            f"products or services. Review the duration, what is covered, "
            f"and the process for making warranty claims."
        ),
        "Liquidated Damages": (
            f"This clause pre-determines the amount of damages payable in "
            f"case of a specific breach or early termination. Specifically: "
            f"\"{answer_text}\". Rather than litigating actual damages, a "
            f"fixed amount is agreed upon upfront. Review whether the amount "
            f"is reasonable and proportionate to the potential harm — courts "
            f"may refuse to enforce penalties that are grossly "
            f"disproportionate."
        ),
        "Irrevocable Or Perpetual License": (
            f"This clause grants a license that cannot be revoked or that "
            f"has no expiration date. Specifically: \"{answer_text}\". Once "
            f"granted the licensor cannot take back these rights regardless "
            f"of circumstances. This provides strong protection for the "
            f"licensee but significantly limits the licensor's future "
            f"flexibility over their own intellectual property."
        ),
        "No-Solicit Of Employees": (
            f"This clause restricts one party from recruiting or hiring "
            f"employees of the other party. Specifically: \"{answer_text}\". "
            f"During and sometimes after the contract term you cannot "
            f"directly or indirectly solicit the other party's staff. Pay "
            f"attention to the duration and whether it covers indirect "
            f"solicitation through third parties or recruiters."
        ),
        "Affiliate License-Licensee": (
            f"This clause extends the license grant to cover affiliates of "
            f"the licensee. Specifically: \"{answer_text}\". This means "
            f"subsidiaries and related companies of the licensee can also "
            f"use the licensed material under the same terms. Review which "
            f"entities qualify as affiliates and whether any restrictions "
            f"apply to how affiliates may use the license."
        ),
        "Joint Ip Ownership": (
            f"This clause establishes shared ownership of intellectual "
            f"property created jointly by both parties. Specifically: "
            f"\"{answer_text}\". Joint ownership can be complex — in many "
            f"jurisdictions either owner can exploit the IP independently "
            f"without accounting to the other. Review how decisions about "
            f"the jointly owned IP will be made and how commercialization "
            f"rights are divided."
        ),
        "Non-Disparagement": (
            f"This clause prohibits one or both parties from making negative "
            f"statements about the other. Specifically: \"{answer_text}\". "
            f"This restricts public criticism, negative reviews, or damaging "
            f"statements about the other party, their products, or their "
            f"employees. Consider how this may affect your ability to share "
            f"honest feedback or participate in legal proceedings."
        ),
        "No-Solicit Of Customers": (
            f"This clause restricts a party from approaching or soliciting "
            f"the other party's customers. Specifically: \"{answer_text}\". "
            f"During and potentially after the contract term you cannot "
            f"target the other party's existing or prospective customers. "
            f"Pay close attention to how broadly customers are defined and "
            f"whether the restriction extends beyond the contract period."
        ),
    }
    return templates.get(
        clause_type,
        f"This clause relates to {clause_type}. "
        f"Specifically: \"{answer_text}\". "
        f"Review this clause carefully with your legal counsel."
    )


# ============================================================
# SECTION 3 — Instruction Builder
# ============================================================

def build_instruction(
    window: str,
    clause_type: str,
    is_positive: bool,
    answer_text: str
) -> str:
    """Build complete Phi-3 instruction for one training example."""

    prompt = (
        f"<|system|>\n{SYSTEM_PROMPT}\n<|end|>\n"
        f"<|user|>\n"
        f"Contract text:\n\"{window}\"\n\n"
        f"Detect: {clause_type} clause\n"
        f"<|end|>\n"
        f"<|assistant|>\n"
    )

    if is_positive:
        explanation = get_explanation(clause_type, answer_text)
        response = (
            f"CLAUSE FOUND: {clause_type}\n\n"
            f"EXTRACTED TEXT:\n\"{answer_text}\"\n\n"
            f"EXPLANATION:\n{explanation}\n"
            f"<|end|>"
        )
    else:
        response = (
            f"NOT PRESENT: {clause_type}\n\n"
            f"This contract does not contain a {clause_type} clause. "
            f"No relevant provisions were found in this section.\n"
            f"<|end|>"
        )

    return prompt + response


# ============================================================
# SECTION 4 — JSONL Writer
# ============================================================

def save_jsonl(instructions: list, path: Path):
    """Save list of instruction strings as JSONL."""
    with open(path, "w") as f:
        for instruction in instructions:
            f.write(json.dumps({"text": instruction}) + "\n")
    print(f"  Saved {len(instructions)} examples → {path}")


# ============================================================
# SECTION 5 — Process One Split
# ============================================================

def process_split(input_path: Path) -> list:
    """
    Load a split file and convert all records
    to Phi-3 instruction strings.
    """
    with open(input_path) as f:
        data = json.load(f)

    records      = data["data"]
    instructions = []
    skipped      = 0

    for record in records:
        context     = record["context"]
        clause_type = record["clause_type"]
        is_positive = record["is_positive"]
        answer_text = record["answer_text"]

        # Extract context window
        window = extract_window(context, answer_text, is_positive)

        # Skip if window is too short
        if len(window.strip()) < 50:
            skipped += 1
            continue

        # Build instruction
        instruction = build_instruction(
            window, clause_type, is_positive, answer_text
        )
        instructions.append(instruction)

    print(f"  Processed {len(instructions)} instructions "
          f"({skipped} skipped) from {input_path.name}")

    return instructions


# ============================================================
# SECTION 6 — Main
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Building training instructions...\n")

    # Process each split
    print("Train split:")
    train = process_split(TRAIN_INPUT)

    print("\nVal split:")
    val = process_split(VAL_INPUT)

    print("\nTest split:")
    test = process_split(TEST_INPUT)

    # Save JSONL files
    print("\nSaving JSONL files...")
    save_jsonl(train, OUTPUT_DIR / "train.jsonl")
    save_jsonl(val,   OUTPUT_DIR / "val.jsonl")
    save_jsonl(test,  OUTPUT_DIR / "test.jsonl")

    # Final summary
    print(f"\n{'='*50}")
    print(f"  Build complete.")
    print(f"  Train : {len(train)} examples")
    print(f"  Val   : {len(val)} examples")
    print(f"  Test  : {len(test)} examples")
    print(f"  Total : {len(train) + len(val) + len(test)} examples")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()