from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def objective(x1: np.ndarray, x2: np.ndarray, x3: np.ndarray) -> np.ndarray:
    return (
        65
        + 18 * np.sin(x1 / 2.2)
        - 0.065 * (x2 - 55) ** 2
        + 7 * np.cos(x3 * 1.4)
        + 0.45 * x1 * x3
    )


def main() -> None:
    rng = np.random.default_rng(42)
    n = 36
    x1 = rng.uniform(0, 10, n)
    x2 = rng.uniform(20, 90, n)
    x3 = rng.uniform(0, 5, n)
    y = objective(x1, x2, x3) + rng.normal(0, 2.5, n)

    data = pd.DataFrame(
        {
            "temperature": x1,
            "time": x2,
            "catalyst": x3,
            "yield": y,
        }
    )
    limit_data = pd.DataFrame(
        {
            "min": [0, 20, 0],
            "max": [10, 90, 5],
            "step": [0.5, 5, 0.25],
        },
        index=["temperature", "time", "catalyst"],
    )

    output_path = Path("sample_data.xlsx")
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        data.to_excel(writer, sheet_name="Sheet1", index=False)
        limit_data.to_excel(writer, sheet_name="Sheet2")

    data.to_csv("sample_data.csv", index=False, encoding="utf-8-sig")
    limit_data.to_csv("sample_limits.csv", encoding="utf-8-sig")
    data.to_csv("sample_data.tsv", index=False, sep="\t", encoding="utf-8-sig")
    limit_data.to_csv("sample_limits.tsv", sep="\t", encoding="utf-8-sig")

    material = rng.choice(["A", "B", "C"], n)
    bonus = np.array([{"A": 0, "B": 5, "C": -4}[value] for value in material])
    mixed_data = data[["temperature", "time"]].copy()
    mixed_data["material"] = material
    mixed_data["yield"] = data["yield"] + bonus
    mixed_limits = pd.DataFrame(
        {
            "min": [0, 20, None],
            "max": [10, 90, None],
            "step": [0.5, 5, None],
            "values": [None, None, "A,B,C"],
        },
        index=["temperature", "time", "material"],
    )
    with pd.ExcelWriter("sample_mixed.xlsx", engine="openpyxl") as writer:
        mixed_data.to_excel(writer, sheet_name="Sheet1", index=False)
        mixed_limits.to_excel(writer, sheet_name="Sheet2")
    mixed_data.to_csv("sample_mixed_data.csv", index=False, encoding="utf-8-sig")
    mixed_limits.to_csv("sample_mixed_limits.csv", encoding="utf-8-sig")


if __name__ == "__main__":
    main()
