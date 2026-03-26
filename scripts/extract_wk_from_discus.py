#!/usr/bin/env python3
"""Extract Waasmaier-Kirfel X-ray form factor coefficients from DISCUS source.

Reads the Fortran arrays per_name, per_wa1..per_wa5, per_wb1..per_wb5, per_wc
from element_data_mod.f90 and writes scattersim/data/waasmaier_kirfel.json.

The DISCUS source contains the intended full-precision values (before any
float32 truncation at compile time). We extract these source-text values
directly.
"""

import json
import re
from pathlib import Path

DISCUS_SOURCE = Path(
    "/Users/bjm42/source/DiffuseCode/lib_f90/element_data_mod.f90"
)
OUTPUT = Path(__file__).resolve().parent.parent / "scattersim" / "data" / "waasmaier_kirfel.json"


def extract_fortran_array(text: str, array_name: str) -> list[str]:
    """Extract values from a Fortran array parameter declaration.

    Returns raw string values to preserve full precision from source text.
    """
    # Match: parameter :: array_name = (/ ... /)
    pattern = (
        rf"parameter\s*::\s*{re.escape(array_name)}\s*=\s*\(/\s*&?\s*(.*?)\s*/\)"
    )
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not find array {array_name}")

    body = match.group(1)
    # Remove continuation markers, line breaks, and whitespace
    body = body.replace("&", " ").replace("\n", " ")
    # Split on commas and strip
    values = [v.strip() for v in body.split(",") if v.strip()]
    return values


def extract_character_array(text: str, array_name: str) -> list[str]:
    """Extract values from a Fortran CHARACTER array parameter declaration."""
    pattern = (
        rf"parameter\s*::\s*{re.escape(array_name)}\s*=\s*\(/\s*&?\s*(.*?)\s*/\)"
    )
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not find array {array_name}")

    body = match.group(1)
    # Extract quoted strings
    names = re.findall(r"'([^']*)'", body)
    # Strip whitespace from names
    return [n.strip() for n in names]


def main():
    text = DISCUS_SOURCE.read_text()

    names = extract_character_array(text, "per_name")
    print(f"Found {len(names)} element/ion names")

    # Extract all coefficient arrays
    array_names = {
        "wa": [f"per_wa{i}" for i in range(1, 6)],
        "wb": [f"per_wb{i}" for i in range(1, 6)],
        "wc": ["per_wc"],
    }

    arrays = {}
    for group_name, arr_list in array_names.items():
        for arr_name in arr_list:
            raw_values = extract_fortran_array(text, arr_name)
            if len(raw_values) != len(names):
                raise ValueError(
                    f"Array {arr_name} has {len(raw_values)} values, "
                    f"expected {len(names)}"
                )
            arrays[arr_name] = [float(v) for v in raw_values]

    # Build the elements dictionary
    # Skip VOID (index 0) and special entries at the end (E1-, POSI, axes, etc.)
    special_names = {"VOID", "E1-", "POSI", "XAXI", "YAXI", "ZAXI", "CENT", "XDIM", "YDIM", "ZDIM"}

    elements = {}
    for i, name in enumerate(names):
        if name in special_names:
            continue

        # Skip entries where all coefficients are zero (no data)
        a_vals = [arrays[f"per_wa{j}"][i] for j in range(1, 6)]
        b_vals = [arrays[f"per_wb{j}"][i] for j in range(1, 6)]
        c_val = arrays["per_wc"][i]

        if all(v == 0.0 for v in a_vals + b_vals + [c_val]):
            continue

        elements[name] = {
            "a": a_vals,
            "b": b_vals,
            "c": c_val,
        }

    print(f"Extracted {len(elements)} species (excluding VOID and special entries)")

    # Build output JSON
    data = {
        "reference": "Waasmaier & Kirfel, Acta Cryst. A51 (1995) 416-431",
        "note": "Coefficients from DISCUS source (full precision, pre-float32 bug)",
        "formula": "f(s) = sum_i a_i * exp(-b_i * s^2) + c, where s = |Q| / (4 * pi)",
        "elements": elements,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Written to {OUTPUT}")

    # Spot-check: sum(a) + c should approximate Z for neutral atoms
    spot_checks = [
        ("NB", 41),  # Niobium
        ("O", 8),    # Oxygen
        ("F", 9),    # Fluorine
    ]
    print("\nSpot-checks (sum(a) + c vs Z):")
    for symbol, z in spot_checks:
        entry = elements[symbol]
        total = sum(entry["a"]) + entry["c"]
        print(f"  {symbol:4s}: sum(a)+c = {total:.6f}, Z = {z}, diff = {total - z:+.6f}")


if __name__ == "__main__":
    main()
