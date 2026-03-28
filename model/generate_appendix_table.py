#!/usr/bin/env python3
"""Generate a full appendix LaTeX table covering all experimental setups."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from model.result_paths import main_result_dir


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULT_CSV = main_result_dir() / "comprehensive_results.csv"
OUTPUT_TEX = REPO_ROOT / "paper" / "data" / "appendix_all_setups_table.tex"

def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "$": r"\$",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def parse_source_filename(filename: str) -> dict[str, str]:
    stem = filename.replace(".csv", "")
    if stem.startswith("baseline_"):
        prefix = "baseline_"
        tau_label = "---"
        embedding_label_map = {
            "naive": "Naive-Baseline",
            "rasch": "Rasch-1PL",
            "irt_2pl": "IRT-2PL-Baseline",
            "mirt": "MIRT-Baseline",
            "knn": "kNN-Baseline",
        }
    else:
        prefix = "amortized_irt_"
        tau_label = "off" if "notau" in stem else "on"
        embedding_label_map = {
            "sae": "SAE",
            "pca": "PCA",
            "raw": "RAW",
            "ones": "ONES",
            "rasch_2pl": "Rasch-2PL",
            "nonamortised_mirt": "NonAmortised-MIRT",
        }

    core = stem[len(prefix) :] if stem.startswith(prefix) else stem
    tokens = core.split("_")

    model_idx = None
    for idx, token in enumerate(tokens):
        if token in {"bernoulli", "beta"}:
            model_idx = idx
            break

    embedding = "_".join(tokens[:model_idx]) if model_idx is not None else tokens[0]
    model = tokens[model_idx] if model_idx is not None else "unknown"

    n_samples = "unknown"
    if "n" in tokens:
        n_idx = tokens.index("n")
        if n_idx + 1 < len(tokens):
            n_samples = tokens[n_idx + 1]

    pre_revision = "none"
    if "pre" in tokens:
        p_idx = tokens.index("pre")
        if p_idx + 1 < len(tokens):
            pre_revision = tokens[p_idx + 1]

    j_val = "1.0"
    for token in tokens:
        if token.startswith("j") and len(token) > 1:
            j_val = token[1:]
            break

    return {
        "embedding": embedding_label_map.get(embedding, embedding.upper()),
        "model": model,
        "n": n_samples,
        "pre": pre_revision,
        "tau": tau_label,
        "j": j_val,
    }


def sem_to_math(value: object) -> str:
    text = str(value)
    if text.strip().lower() in {"n/a", "na", "unavailable", "---"}:
        return "---"
    # convert 0.700±0.009 -> $0.700 \pm 0.009$
    if "±" in text:
        left, right = text.split("±", 1)
        return f"${left.strip()} \\pm {right.strip()}$"
    if re.fullmatch(r"-?\d+(\.\d+)?", text):
        return f"${text}$"
    return latex_escape(text)


def numeric_to_math(value: object) -> str:
    text = str(value).strip()
    if text.lower() in {"n/a", "na", "baseline", "unavailable", "---"}:
        return "---"
    if text == "max":
        return r"$\max$"
    if text == "none":
        return r"$\varnothing$"
    if re.fullmatch(r"-?\d+(\.\d+)?", text):
        return f"${text}$"
    return latex_escape(text)


def normalize_n_mode(value: object) -> str:
    text = str(value).strip().lower()
    if text == "1":
        return "1"
    return "full"


def normalize_test_takers(value: object) -> str:
    text = str(value).strip().lower()
    if text in {"full", "max"}:
        return "143"
    return str(value).strip()


def build_table(df: pd.DataFrame) -> str:
    records = []
    for _, row in df.iterrows():
        src = str(row.get("Source File", ""))
        meta = parse_source_filename(src)
        records.append(
            {
                "Embedding": meta["embedding"],
                "Model": meta["model"],
                "N": normalize_n_mode(meta["n"]),
                "Pre": meta["pre"],
                "Tau": meta["tau"],
                "j": meta["j"],
                "BestTau": row.get("Best Tau", "---"),
                "AUC": row.get("AUC", "---"),
                "RMSE": row.get("RMSE", "---"),
                "Seeds": row.get("Seeds @ Best Tau", "---"),
            }
        )

    out = pd.DataFrame(records)

    # Explicitly separate revision stage from test-taker count shown in the table:
    # - Stage: Pre or Post
    # - TestTakers: pre-revision level for Pre runs, else N for Post runs
    out["Stage"] = out["Pre"].apply(lambda v: "Pre" if str(v) != "none" else "Post")
    out["TestTakers"] = out.apply(
        lambda r: r["Pre"] if str(r["Pre"]) != "none" else r["N"],
        axis=1,
    )
    out["TestTakers"] = out["TestTakers"].apply(normalize_test_takers)

    # Sort by setup first, then method, so baselines and amortized methods
    # appear together for each comparable condition.
    method_order = {
        "Naive-Baseline": 0,
        "Rasch-1PL": 1,
        "IRT-2PL-Baseline": 2,
        "MIRT-Baseline": 3,
        "kNN-Baseline": 4,
        "ONES": 5,
        "SAE": 6,
        "PCA": 7,
        "RAW": 8,
        "Rasch-2PL": 9,
        "NonAmortised-MIRT": 10,
    }
    model_order = {"bernoulli": 0, "beta": 1}
    stage_order = {"Pre": 0, "Post": 1}
    t_order = {"4": 1, "8": 2, "16": 3, "32": 4, "54": 5, "64": 6, "143": 7, "max": 7}

    def n_key(value: str) -> float:
        if value == "1":
            return 1.0
        if value == "full":
            return 2.0
        return 9.0

    out["_model_order"] = out["Model"].map(model_order).fillna(99)
    out["_n_order"] = out["N"].map(n_key)
    out["_stage_order"] = out["Stage"].map(stage_order).fillna(99)
    out["_t_order"] = out["TestTakers"].map(t_order).fillna(99)
    out["_j_order"] = pd.to_numeric(out["j"], errors="coerce").fillna(999.0)
    out["_method_order"] = out["Embedding"].map(method_order).fillna(99)
    out["_tau_order"] = out["Tau"].map({"---": 0, "on": 1, "off": 2}).fillna(9)

    out = out.sort_values(
        by=[
            "_model_order",
            "_stage_order",
            "_t_order",
            "_j_order",
            "_method_order",
            "_tau_order",
            "_n_order",
        ],
        kind="stable",
    )

    lines = []
    lines.append(r"% Auto-generated by model/generate_appendix_table.py")
    lines.append(r"% Required packages: longtable, booktabs")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\begin{longtable}{llccccccccc}")
    lines.append(r"\caption{Appendix summary of all experimental setups. Each row reports one setup from the full sweep, evaluated at the best $\tau$ (selected by highest mean $\mathrm{AUC}$). Baseline rows are also included for Naive, Rasch-1PL, IRT-2PL, MIRT, and kNN, preserving variation over test takers and item subsets. Notation: \emph{Revision} indicates whether the run is pre-revision (Pre) or post-revision (Post). \emph{Test Takers} is the effective test-taker count used for that run (for Pre rows this comes from the pre-revision subset level; for Post rows this follows the run setting), with any legacy \texttt{max}/\texttt{full} test-taker setting shown as $143$. $N\in\{1,\mathrm{full}\}$ denotes repeated matrix-sampling mode, where \emph{full} uses all available repetitions for that setup. $\tau\in\{\mathrm{on},\mathrm{off},\texttt{---}\}$ indicates whether regularization is enabled (or not applicable for baseline-only rows); $j$ is the item-fraction control used in scaling-law runs; \emph{Best $\tau$} is the selected regularization value for amortized setups and \texttt{---} for baseline rows; $\mathrm{AUC}$ and $\mathrm{RMSE}$ are reported as mean $\pm$ standard error over repetitions; \emph{Seeds} is the number of random seeds contributing to that summary.}\\")
    lines.append(r"\label{tab:appendix_all_setups}\\")
    lines.append(r"\toprule")
    lines.append(r"Embedding & Likelihood & Revision & Test Takers & $N$ & $\tau$ & $j$ & Best $\tau$ & $\mathrm{AUC}$ & $\mathrm{RMSE}$ & Seeds \\")
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(r"\multicolumn{11}{c}{\tablename\ \thetable{} -- continued from previous page}\\")
    lines.append(r"\toprule")
    lines.append(r"Embedding & Likelihood & Revision & Test Takers & $N$ & $\tau$ & $j$ & Best $\tau$ & $\mathrm{AUC}$ & $\mathrm{RMSE}$ & Seeds \\")
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{11}{r}{continued on next page}\\")
    lines.append(r"\endfoot")
    lines.append(r"\bottomrule")
    lines.append(r"\endlastfoot")

    prev_group = None
    for _, row in out.iterrows():
        group_key = (row["Model"], row["Stage"], row["TestTakers"], row["j"])
        if prev_group is not None and group_key != prev_group:
            lines.append(r"\cmidrule(lr){1-11}")

        line = (
            f"{latex_escape(row['Embedding'])} & "
            f"{latex_escape(row['Model'])} & "
            f"{latex_escape(row['Stage'])} & "
            f"{numeric_to_math(row['TestTakers'])} & "
            f"{numeric_to_math(row['N'])} & "
            f"{latex_escape(row['Tau'])} & "
            f"{numeric_to_math(row['j'])} & "
            f"{sem_to_math(row['BestTau'])} & "
            f"{sem_to_math(row['AUC'])} & "
            f"{sem_to_math(row['RMSE'])} & "
            f"{numeric_to_math(row['Seeds'])} \\\\" 
        )
        lines.append(line)
        prev_group = group_key

    lines.append(r"\end{longtable}")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    if not RESULT_CSV.exists():
        raise FileNotFoundError(f"Missing input CSV: {RESULT_CSV}")

    df = pd.read_csv(RESULT_CSV)
    required_cols = {"AUC", "RMSE", "Best Tau", "Seeds @ Best Tau", "Source File"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing required columns: {sorted(missing)}")

    latex = build_table(df)
    OUTPUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_TEX.write_text(latex, encoding="utf-8")

    print(f"Wrote full appendix table with {len(df)} setups to {OUTPUT_TEX}")


if __name__ == "__main__":
    main()
