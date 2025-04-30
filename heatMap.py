#!/usr/bin/env python3
"""
Keyword Frequency Heatmap Generator

Instructions:
  1. Set INPUT_FILE, DELIMITER, COMPANY_COL, and MAX_FREQ below.
  2. MAX_FREQ controls the upper bound of the color scale.
  3. If your true counts exceed 200, anything above 200 will be shown at the top end of the palette.
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt

# --------------- User Configuration --------------- #
INPUT_FILE  = 'compVsFreqList.csv'   # <-- path to your data file
DELIMITER   = ','             # <-- '\t' for tabs, ',' for commas, etc.
COMPANY_COL = 'Company'        # <-- name of your company column

MAX_FREQ    = 100              # <-- colorbar will cap at this value
# -------------------------------------------------- #

def load_data(filename, delimiter):
    """Load the dataset into a pandas DataFrame."""
    try:
        df = pd.read_csv(filename, sep=delimiter, dtype=str)
    except Exception as e:
        print(f"Error loading {filename!r} with delimiter {delimiter!r}:\n  {e}")
        sys.exit(1)
    return df

def clean_and_prepare(df):
    """
    1. Strip whitespace from all column names.
    2. Verify COMPANY_COL exists.
    3. Drop rows where all keyword columns are empty/NaN.
    4. Convert keyword columns to ints.
    5. Aggregate by company.
    """
    df.columns = df.columns.str.strip()

    if COMPANY_COL not in df.columns:
        print(f"Error: column {COMPANY_COL!r} not found.")
        print("  Pandas read these columns:", df.columns.tolist())
        print("→ Check your DELIMITER or header row.")
        sys.exit(1)

    keywords = [c for c in df.columns if c != COMPANY_COL]
    df = df.dropna(subset=keywords, how='all')
    for k in keywords:
        df[k] = pd.to_numeric(df[k], errors='coerce').fillna(0).astype(int)

    heatmap_df = df.groupby(COMPANY_COL)[keywords].sum()
    return heatmap_df

def plot_heatmap(data, normalize=False, annotate=False):
    """
    Plot a heatmap of 'data' (companies × keywords).
    - normalize: each row sums to 1
    - annotate: overlay cell values
    - color scale caps at [0, MAX_FREQ]
    """
    if normalize:
        data = data.div(data.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(12, 6))
    # vmin=0, vmax=MAX_FREQ to fix the color scale
    im = ax.imshow(data, aspect='auto', interpolation='nearest',
                   vmin=0, vmax=MAX_FREQ)

    if annotate:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, data.iat[i, j],
                        ha='center', va='center', fontsize=6)

    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels(data.columns, rotation=90)
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels(data.index)
    ax.set_xlabel('Keyword')
    ax.set_ylabel('Company')
    ax.set_title('Keyword Frequency Heatmap by Company (capped at {})'.format(MAX_FREQ))

    cbar = fig.colorbar(im, ax=ax)
    label = 'Frequency (0–{})'.format(MAX_FREQ)
    if normalize:
        label += ' / row'
    cbar.set_label(label)

    plt.tight_layout()
    plt.show()

def main():
    df = load_data(INPUT_FILE, DELIMITER)
    heatmap_df = clean_and_prepare(df)

    # Optional: sort by total mentions
    # heatmap_df = heatmap_df.loc[heatmap_df.sum(axis=1).sort_values(ascending=False).index]

    normalize = False   # True to see percentages per row
    annotate  = False   # True to overlay raw values

    plot_heatmap(heatmap_df, normalize=normalize, annotate=annotate)

if __name__ == '__main__':
    main()
