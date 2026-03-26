#!/usr/bin/env python3
"""Extract electron scattering factor coefficients from DISCUS source.

Reads the Fortran arrays per_ea1..per_ea5, per_eb1..per_eb5 from
element_data_mod.f90 and writes scattersim/data/peng.json.

NOTE: These coefficients are stored in DISCUS without attribution. They
appear to use the 5-Gaussian form f(s) = sum_i a_i * exp(-b_i * s^2).
The provenance (Peng vs Doyle-Turner vs other) needs verification.
"""

import json
import re
from pathlib import Path

DISCUS_SOURCE = Path(
    "/Users/bjm42/source/DiffuseCode/lib_f90/element_data_mod.f90"
)
OUTPUT = Path(__file__).resolve().parent.parent / "scattersim" / "data" / "peng.json"


def extract_fortran_array(text: str, array_name: str) -> list[str]:
    """Extract values from a Fortran array parameter declaration."""
    pattern = (
        rf"parameter\s*::\s*{re.escape(array_name)}\s*=\s*\(/\s*&?\s*(.*?)\s*/\)"
    )
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not find array {array_name}")

    body = match.group(1)
    body = body.replace("&", " ").replace("\n", " ")
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
    names = re.findall(r"'([^']*)'", body)
    return [n.strip() for n in names]


def main():
    text = DISCUS_SOURCE.read_text()

    names = extract_character_array(text, "per_name")

    # Extract electron scattering arrays
    arrays = {}
    for prefix in ("per_ea", "per_eb"):
        for i in range(1, 6):
            arr_name = f"{prefix}{i}"
            raw_values = extract_fortran_array(text, arr_name)
            if len(raw_values) != len(names):
                raise ValueError(
                    f"Array {arr_name} has {len(raw_values)} values, "
                    f"expected {len(names)}"
                )
            arrays[arr_name] = [float(v) for v in raw_values]

    # Build elements dictionary -- only include entries with non-zero data
    special_names = {"VOID", "E1-", "POSI", "XAXI", "YAXI", "ZAXI", "CENT", "XDIM", "YDIM", "ZDIM"}

    elements = {}
    for i, name in enumerate(names):
        if name in special_names:
            continue

        a_vals = [arrays[f"per_ea{j}"][i] for j in range(1, 6)]
        b_vals = [arrays[f"per_eb{j}"][i] for j in range(1, 6)]

        # Skip entries where all coefficients are zero
        if all(v == 0.0 for v in a_vals + b_vals):
            continue

        elements[name] = {
            "a": a_vals,
            "b": b_vals,
        }

    print(f"Extracted {len(elements)} species with electron scattering data")

    data = {
        "reference": "Peng, Ren, Dudarev & Whelan, Acta Cryst. A52 (1996) 257-276",
        "note": "TODO: verify provenance -- coefficients extracted from DISCUS source (per_ea/per_eb arrays), attribution unconfirmed",
        "formula": "f(s) = sum_i a_i * exp(-b_i * s^2), where s = |Q| / (4 * pi)",
        "elements": elements,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Written to {OUTPUT}")

    # Spot-check: print Nb, O, F entries
    for symbol in ("NB", "O", "F"):
        if symbol in elements:
            entry = elements[symbol]
            print(f"  {symbol}: a={entry['a']}, b={entry['b']}")
        else:
            print(f"  {symbol}: NOT FOUND in electron data")


if __name__ == "__main__":
    main()
