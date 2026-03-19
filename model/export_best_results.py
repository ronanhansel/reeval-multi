import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


RESULT_DIR = Path('/Users/ronan/Developer/agent-eval/model/result')
OUTPUT_CSV = RESULT_DIR / 'comprehensive_results.csv'
OUTPUT_MD = RESULT_DIR / 'comprehensive_results.md'
APPENDIX_GENERATOR = RESULT_DIR.parent / 'generate_appendix_table.py'


def format_res(mean: float, sem: float) -> str:
    if pd.isna(mean):
        return 'N/A'
    if sem == 0 or pd.isna(sem):
        return f"{mean:.3f}"
    return f"{mean:.3f}±{sem:.3f}"


def discover_result_csvs() -> list[str]:
    files = []
    for path in RESULT_DIR.glob('amortized_irt_*.csv'):
        files.append(path.name)
    return sorted(files)


def parse_setup_from_filename(filename: str) -> dict[str, str]:
    stem = filename.replace('.csv', '')
    prefix = 'amortized_irt_'
    core = stem[len(prefix):] if stem.startswith(prefix) else stem
    tokens = core.split('_')

    model_idx = None
    for i, tok in enumerate(tokens):
        if tok in {'bernoulli', 'beta'}:
            model_idx = i
            break

    embedding = '_'.join(tokens[:model_idx]) if model_idx is not None else tokens[0]
    model = tokens[model_idx] if model_idx is not None else 'unknown'

    n_samples = 'unknown'
    if 'n' in tokens:
        n_idx = tokens.index('n')
        if n_idx + 1 < len(tokens):
            n_samples = tokens[n_idx + 1]

    pre_revision = 'none'
    if 'pre' in tokens:
        p_idx = tokens.index('pre')
        if p_idx + 1 < len(tokens):
            pre_revision = tokens[p_idx + 1]

    no_tau = 'notau' in tokens
    j_token = next((tok for tok in tokens if tok.startswith('j')), None)
    j_percentage = j_token[1:] if j_token is not None and len(j_token) > 1 else '1.0'

    emb_label_map = {
        'sae': 'SAE',
        'pca': 'PCA',
        'raw': 'RAW',
        'ones': 'ONES',
        'rasch_2pl': 'Rasch-2PL',
        'nonamortised_mirt': 'NonAmortised-MIRT',
    }
    emb_label = emb_label_map.get(embedding, embedding.upper())

    setup_label = (
        f"{emb_label} | model={model} | n={n_samples} | pre={pre_revision} | "
        f"tau={'off' if no_tau else 'on'} | j={j_percentage}"
    )

    return {
        'embedding': emb_label,
        'model': model,
        'n_samples': n_samples,
        'pre_revision': pre_revision,
        'tau_mode': 'notau' if no_tau else 'tau',
        'j_percentage': j_percentage,
        'setup_label': setup_label,
    }


def summarize_result_file(filename: str) -> dict[str, str] | None:
    path = RESULT_DIR / filename
    try:
        df = pd.read_csv(path, on_bad_lines='skip')
    except Exception as exc:
        print(f"Warning: failed to read {filename}: {exc}")
        return None

    df.columns = df.columns.str.strip()
    for col in ['lambda_tau', 'auc_amortized', 'rmse_amortized', 'seed']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    required = {'lambda_tau', 'auc_amortized', 'rmse_amortized'}
    if not required.issubset(df.columns):
        print(f"Warning: skipping {filename}, missing required columns: {required - set(df.columns)}")
        return None

    df = df.dropna(subset=['lambda_tau', 'auc_amortized', 'rmse_amortized'])
    if df.empty:
        print(f"Warning: skipping {filename}, no valid rows after cleanup")
        return None

    tau_stats = df.groupby('lambda_tau', dropna=True)['auc_amortized'].mean()
    if tau_stats.empty:
        print(f"Warning: skipping {filename}, no valid tau statistics")
        return None

    best_tau = float(tau_stats.idxmax())
    best_df = df[np.isclose(df['lambda_tau'].astype(float), best_tau, atol=1e-12)]

    auc_mean = float(best_df['auc_amortized'].mean())
    auc_sem = float(best_df['auc_amortized'].sem()) if len(best_df) > 1 else 0.0
    rmse_mean = float(best_df['rmse_amortized'].mean())
    rmse_sem = float(best_df['rmse_amortized'].sem()) if len(best_df) > 1 else 0.0

    seed_count = int(best_df['seed'].dropna().nunique()) if 'seed' in best_df.columns else len(best_df)
    setup = parse_setup_from_filename(filename)

    row = {
        'Model Configuration': setup['setup_label'],
        'AUC': format_res(auc_mean, auc_sem),
        'RMSE': format_res(rmse_mean, rmse_sem),
        'Best Tau': f"{best_tau:.6g}",
        'Seeds @ Best Tau': seed_count,
        'Source File': filename,
    }
    return row


def sort_key(row: dict[str, str]) -> tuple:
    setup = parse_setup_from_filename(row['Source File'])
    emb_order = {
        'SAE': 0,
        'PCA': 1,
        'RAW': 2,
        'ONES': 3,
        'Rasch-2PL': 4,
        'NonAmortised-MIRT': 5,
    }

    pre_order = {'none': 0, '4': 1, '8': 2, '16': 3, '32': 4, '64': 5, 'max': 6}

    n_val = setup['n_samples']
    n_order = {'1': 1, '32': 2, 'max': 3}

    try:
        j_val = float(setup['j_percentage'])
    except ValueError:
        j_val = 1.0

    return (
        emb_order.get(setup['embedding'], 99),
        setup['model'],
        pre_order.get(setup['pre_revision'], 99),
        n_order.get(n_val, 98 if n_val.isdigit() else 99),
        1 if setup['tau_mode'] == 'notau' else 0,
        j_val,
        row['Source File'],
    )


def main() -> None:
    csv_files = discover_result_csvs()
    if not csv_files:
        print(f"No amortized_irt CSV files found in {RESULT_DIR}")
        return

    rows = []
    for filename in csv_files:
        row = summarize_result_file(filename)
        if row is not None:
            rows.append(row)

    if not rows:
        print('No valid result rows produced.')
        return

    rows = sorted(rows, key=sort_key)
    final_df = pd.DataFrame(rows)

    final_df.to_csv(OUTPUT_CSV, index=False)
    with open(OUTPUT_MD, 'w') as f:
        f.write(final_df.to_markdown(index=False))

    print(f"Exported {len(final_df)} setups to {OUTPUT_CSV}")

    if APPENDIX_GENERATOR.exists():
        try:
            subprocess.run([sys.executable, str(APPENDIX_GENERATOR)], check=True)
            print("Regenerated appendix table via generate_appendix_table.py")
        except subprocess.CalledProcessError as exc:
            print(f"Warning: failed to regenerate appendix table ({exc})")
    else:
        print(f"Warning: appendix generator not found at {APPENDIX_GENERATOR}")


if __name__ == '__main__':
    main()
