#!/usr/bin/env python3
"""
eda_cooccurrence.py

Compute pairwise Pearson correlations between keyword counts
and display a heatmap of keyword co-occurrence, with the color
scale clamped between 0.125 and 1.

Usage:
    python eda_cooccurrence.py path/to/your/keyword_counts.csv

If no path is given, it defaults to 'polarityVkeyword.csv'.
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    # 1. Get CSV path
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'polarityVkeyword.csv'
    
    # 2. Load data
    df = pd.read_csv(csv_path)
    
    # 3. Clean up
    df.columns = df.columns.str.strip()
    # Fill NaN in keyword columns with 0
    kw_cols = [c for c in df.columns if c.lower() != 'company']
    df[kw_cols] = df[kw_cols].fillna(0)
    
    # 4. Compute correlation matrix
    corr_mat = df[kw_cols].corr()
    
    # 5. Plot heatmap with scale from 0.125 to 1
    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.matshow(
        corr_mat,
        vmin=0.125,    # minimum of color scale
        vmax=1.0,      # maximum of color scale
        cmap='RdBu_r'
    )
    fig.colorbar(cax, fraction=0.046, pad=0.04, label='Pearson r')
    
    # 6. Label ticks
    ax.set_xticks(np.arange(len(kw_cols)))
    ax.set_yticks(np.arange(len(kw_cols)))
    ax.set_xticklabels(kw_cols, rotation=90)
    ax.set_yticklabels(kw_cols)
    
    plt.title('Keyword–Keyword Pearson Correlation Heatmap\n(Clamped to 0.125–1)')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
