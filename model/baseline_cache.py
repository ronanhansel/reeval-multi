#!/usr/bin/env python3
"""
Shared helpers for setup-level baseline CSV caches.
"""

import os
import re

import numpy as np
import pandas as pd

BASELINE_METRIC_COLS = [
    'rmse_naive', 'rmse_rasch', 'rmse_2pl', 'rmse_mirt',
    'auc_naive', 'auc_rasch', 'auc_2pl', 'auc_mirt',
    'rmse_knn', 'auc_knn',
]
NON_MIRT_METRIC_COLS = [c for c in BASELINE_METRIC_COLS if c not in {'rmse_mirt', 'auc_mirt'}]
BASELINE_KEY_COLS = ['seed', 'model_type', 'n_samples', 'pre_revision', 'j_percentage', 'train_retention', 'baseline_embedding_type']
BASELINE_AUX_COLS = ['agent_batch_size', 'selected_mirt_dim', 'mirt_sweep_min', 'mirt_sweep_max', 'mirt_selection_version']
MIRT_SUMMARY_COLS = ['rmse_mirt', 'auc_mirt', 'selected_mirt_dim', 'mirt_sweep_min', 'mirt_sweep_max', 'mirt_selection_version']
MIRT_SWEEP_KEY_COLS = BASELINE_KEY_COLS + ['mirt_dim']
MIRT_SWEEP_METRIC_COLS = ['rmse_mirt', 'auc_mirt', 'val_rmse_mirt', 'val_auc_mirt']
BASELINE_METHOD_SPECS = {
    'naive': ('auc_naive', 'rmse_naive'),
    'rasch': ('auc_rasch', 'rmse_rasch'),
    'irt_2pl': ('auc_2pl', 'rmse_2pl'),
    'mirt': ('auc_mirt', 'rmse_mirt'),
    'knn': ('auc_knn', 'rmse_knn'),
}
GROUPED_BASELINE_RE = re.compile(
    r"^baseline_(naive|rasch|irt_2pl|mirt|knn)_([^_]+)_(beta|bernoulli)_pre_([^_]+)_n_([^_]+)_j(.+?)(?:_ret_[^_]+)?\.csv$"
)


def _last_non_null(series):
    non_null = series.dropna()
    if non_null.empty:
        return np.nan
    return non_null.iloc[-1]


def normalize_pre_revision(value):
    if value is None:
        return 'none'
    v = str(value).strip().lower()
    return v if v else 'none'


def normalize_j_percentage(value):
    return float(f"{float(value):.6f}")


def normalize_train_retention(value):
    return float(f"{float(value):.6f}")


def normalize_baseline_embedding_type(value):
    if value is None or pd.isna(value):
        return 'pca'
    v = str(value).strip().lower()
    return v if v and v != 'nan' else 'pca'


def compute_agent_batch_size(pre_revision, n_samples):
    pre = normalize_pre_revision(pre_revision)
    if pre == 'none':
        return str(int(n_samples))
    if pre == 'max':
        return 'max'
    try:
        return str(int(pre))
    except Exception:
        return pre


def _normalize_key_payload(payload):
    out = dict(payload)
    out['seed'] = int(out['seed'])
    out['model_type'] = str(out['model_type']).strip().lower()
    out['n_samples'] = int(out['n_samples'])
    out['pre_revision'] = normalize_pre_revision(out['pre_revision'])
    out['j_percentage'] = normalize_j_percentage(out['j_percentage'])
    out['train_retention'] = normalize_train_retention(out.get('train_retention', 1.0))
    out['baseline_embedding_type'] = normalize_baseline_embedding_type(out['baseline_embedding_type'])
    return out


def _format_numeric_token(value):
    return f"{float(value):.6f}".rstrip('0').rstrip('.')


def grouped_baseline_file(path, method_key, key):
    key = _normalize_key_payload(key)
    directory = os.path.dirname(path)
    j_token = _format_numeric_token(key['j_percentage'])
    retention_token = _format_numeric_token(key['train_retention'])
    return os.path.join(
        directory,
        (
            f"baseline_{method_key}_{key['baseline_embedding_type']}_{key['model_type']}"
            f"_pre_{key['pre_revision']}_n_{key['n_samples']}_j{j_token}_ret_{retention_token}.csv"
        ),
    )


def grouped_mirt_sweep_file(path, key):
    key = _normalize_key_payload(key)
    directory = os.path.dirname(path)
    j_token = _format_numeric_token(key['j_percentage'])
    retention_token = _format_numeric_token(key['train_retention'])
    return os.path.join(
        directory,
        (
            f"baseline_mirt_sweep_{key['baseline_embedding_type']}_{key['model_type']}"
            f"_pre_{key['pre_revision']}_n_{key['n_samples']}_j{j_token}_ret_{retention_token}.csv"
        ),
    )


def _append_grouped_row(path, row, key_cols):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df_new = pd.DataFrame([row])
    if os.path.exists(path):
        try:
            df_old = pd.read_csv(path, on_bad_lines='skip')
        except Exception:
            df_old = pd.DataFrame(columns=df_new.columns)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    df = df.drop_duplicates(subset=key_cols, keep='last')
    df.to_csv(path, index=False)


def write_grouped_baseline_files(path, row):
    key = _normalize_key_payload({k: row[k] for k in BASELINE_KEY_COLS})
    base_payload = {
        **key,
        'agent_batch_size': row.get('agent_batch_size', compute_agent_batch_size(key['pre_revision'], key['n_samples'])),
    }
    for method_key, (auc_col, rmse_col) in BASELINE_METHOD_SPECS.items():
        if method_key == 'mirt':
            payload = dict(base_payload)
            if auc_col in row and not pd.isna(row[auc_col]):
                payload[auc_col] = float(row[auc_col])
            if rmse_col in row and not pd.isna(row[rmse_col]):
                payload[rmse_col] = float(row[rmse_col])
            for col in ['selected_mirt_dim', 'mirt_sweep_min', 'mirt_sweep_max', 'mirt_selection_version']:
                if col in row and not pd.isna(row[col]):
                    payload[col] = int(row[col])
            if auc_col in payload or rmse_col in payload:
                _append_grouped_row(grouped_baseline_file(path, method_key, key), payload, ['seed'])
            continue

        payload = dict(base_payload)
        if auc_col in row and not pd.isna(row[auc_col]):
            payload[auc_col] = float(row[auc_col])
        if rmse_col in row and not pd.isna(row[rmse_col]):
            payload[rmse_col] = float(row[rmse_col])
        if auc_col in payload or rmse_col in payload:
            _append_grouped_row(grouped_baseline_file(path, method_key, key), payload, ['seed'])


def write_grouped_mirt_sweep_file(path, row):
    key = _normalize_key_payload({k: row[k] for k in BASELINE_KEY_COLS})
    payload = {**key, 'mirt_dim': int(row['mirt_dim'])}
    for col in MIRT_SWEEP_METRIC_COLS:
        if col in row and not pd.isna(row[col]):
            payload[col] = float(row[col])
    _append_grouped_row(grouped_mirt_sweep_file(path, key), payload, ['seed', 'mirt_dim'])


def _load_grouped_baseline_files(path):
    directory = os.path.dirname(path)
    if not os.path.isdir(directory):
        return None

    grouped_files = []
    for filename in os.listdir(directory):
        match = GROUPED_BASELINE_RE.match(filename)
        if match:
            grouped_files.append(os.path.join(directory, filename))
    if not grouped_files:
        return None

    frames = []
    for file_path in sorted(grouped_files):
        try:
            df = pd.read_csv(file_path, on_bad_lines='skip')
        except Exception:
            continue
        if df.empty:
            continue
        for col in BASELINE_KEY_COLS + BASELINE_METRIC_COLS + BASELINE_AUX_COLS:
            if col not in df.columns:
                df[col] = np.nan
        frames.append(df[BASELINE_KEY_COLS + BASELINE_METRIC_COLS + BASELINE_AUX_COLS])

    if not frames:
        return pd.DataFrame(columns=BASELINE_KEY_COLS + BASELINE_METRIC_COLS + BASELINE_AUX_COLS)

    combined = pd.concat(frames, ignore_index=True)
    agg_map = {}
    for col in BASELINE_METRIC_COLS + BASELINE_AUX_COLS:
        agg_map[col] = _last_non_null
    combined = combined.groupby(BASELINE_KEY_COLS, dropna=False, as_index=False).agg(agg_map)
    combined['baseline_embedding_type'] = combined['baseline_embedding_type'].map(normalize_baseline_embedding_type)
    combined['agent_batch_size'] = [
        compute_agent_batch_size(pr, ns)
        for pr, ns in zip(combined['pre_revision'], combined['n_samples'])
    ]
    return combined[BASELINE_KEY_COLS + BASELINE_METRIC_COLS + BASELINE_AUX_COLS]


def load_baseline_store(path):
    grouped = _load_grouped_baseline_files(path)
    if grouped is not None:
        return grouped

    if os.path.exists(path):
        try:
            df = pd.read_csv(path, on_bad_lines='skip')
            for col in BASELINE_KEY_COLS + BASELINE_METRIC_COLS + BASELINE_AUX_COLS:
                if col not in df.columns:
                    df[col] = np.nan
            df['baseline_embedding_type'] = df['baseline_embedding_type'].map(normalize_baseline_embedding_type)
            df['agent_batch_size'] = [
                compute_agent_batch_size(pr, ns)
                for pr, ns in zip(df['pre_revision'], df['n_samples'])
            ]
            return df[BASELINE_KEY_COLS + BASELINE_METRIC_COLS + BASELINE_AUX_COLS]
        except Exception:
            pass

    return pd.DataFrame(columns=BASELINE_KEY_COLS + BASELINE_METRIC_COLS + BASELINE_AUX_COLS)


def load_mirt_sweep_store(path):
    directory = os.path.dirname(path)
    if os.path.isdir(directory):
        frames = []
        for filename in sorted(os.listdir(directory)):
            if not filename.startswith('baseline_mirt_sweep_') or not filename.endswith('.csv'):
                continue
            try:
                df = pd.read_csv(os.path.join(directory, filename), on_bad_lines='skip')
            except Exception:
                continue
            for col in MIRT_SWEEP_KEY_COLS + MIRT_SWEEP_METRIC_COLS:
                if col not in df.columns:
                    df[col] = np.nan
            frames.append(df[MIRT_SWEEP_KEY_COLS + MIRT_SWEEP_METRIC_COLS])
        if frames:
            return pd.concat(frames, ignore_index=True)

    if os.path.exists(path):
        try:
            df = pd.read_csv(path, on_bad_lines='skip')
            for col in MIRT_SWEEP_KEY_COLS + MIRT_SWEEP_METRIC_COLS:
                if col not in df.columns:
                    df[col] = np.nan
            return df[MIRT_SWEEP_KEY_COLS + MIRT_SWEEP_METRIC_COLS]
        except Exception:
            pass

    return pd.DataFrame(columns=MIRT_SWEEP_KEY_COLS + MIRT_SWEEP_METRIC_COLS)


def write_baseline_manifest(path):
    df = load_baseline_store(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def write_mirt_sweep_manifest(path):
    df = load_mirt_sweep_store(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def migrate_baseline_csv_to_grouped_files(baseline_output, mirt_sweep_output=None, write_manifest=True):
    baseline_df = pd.read_csv(baseline_output, on_bad_lines='skip') if os.path.exists(baseline_output) else pd.DataFrame()
    for _, row in baseline_df.iterrows():
        write_grouped_baseline_files(baseline_output, row.to_dict())

    if mirt_sweep_output is not None and os.path.exists(mirt_sweep_output):
        sweep_df = pd.read_csv(mirt_sweep_output, on_bad_lines='skip', low_memory=False)
        for _, row in sweep_df.iterrows():
            write_grouped_mirt_sweep_file(mirt_sweep_output, row.to_dict())

    if write_manifest:
        write_baseline_manifest(baseline_output)
        if mirt_sweep_output is not None:
            write_mirt_sweep_manifest(mirt_sweep_output)
