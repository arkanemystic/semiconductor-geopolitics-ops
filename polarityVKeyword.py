#!/usr/bin/env python3
"""
Analyze keyword frequencies vs. sentiment polarity.

This version auto-strips column headers and finds any column
with “polarity” in its name (case-insensitive).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import sys

# --- Configuration ---
CSV_PATH = 'polarityVKeyword.csv'

KEYWORDS = [
    'taiwan','china','semiconductor','supply chain','risk','threat',
    'growth','chip','factory','production','shortage','investment',
    'manufacturing','innovation','disruption','global','market',
    'demand','silicon','foundry'
]

def main():
    # 1. Load
    df = pd.read_csv(CSV_PATH)
    # 1a. Clean up column names
    df.columns = df.columns.str.strip()
    
    # 1b. Auto-detect polarity column
    candidates = [c for c in df.columns if 'polarity' in c.lower()]
    if not candidates:
        print("Error: no column matching '*polarity*' found.")
        print("Available columns:\n", "\n".join(df.columns))
        sys.exit(1)
    POLARITY_COL = candidates[0]
    print(f"Using polarity column: '{POLARITY_COL}'\n")
    
    # 2. Convert polarity to numeric, drop rows where it fails
    df[POLARITY_COL] = pd.to_numeric(df[POLARITY_COL], errors='coerce')
    df = df.dropna(subset=[POLARITY_COL])
    
    # 3. Ensure keyword columns exist
    missing = [kw for kw in KEYWORDS if kw not in df.columns]
    if missing:
        print("Warning: these keywords not in your CSV columns:", missing)
        # you may choose to remove them or double-check spelling
    # Fill any NaNs in keyword counts with 0
    present = [kw for kw in KEYWORDS if kw in df.columns]
    df[present] = df[present].fillna(0)
    
    # 4. Compute correlations
    corrs = df[present].corrwith(df[POLARITY_COL]).sort_values(ascending=False)
    print("Keyword correlations with polarity:\n")
    print(corrs.to_string(), "\n")
    
    # 5. Bar‐plot correlations
    plt.figure(figsize=(10, 6))
    corrs.plot.bar()
    plt.axhline(0, color='grey', linestyle='--', linewidth=1)
    plt.ylabel('Pearson r')
    plt.title('Keyword Frequency vs. Sentiment Polarity')
    plt.tight_layout()
    plt.show()
    
    # 6. Drill into top 3 absolute correlations
    top3 = corrs.abs().sort_values(ascending=False).head(3).index.tolist()
    for kw in top3:
        x = df[kw]
        y = df[POLARITY_COL]
        m, b = np.polyfit(x, y, 1)
        line_x = np.linspace(x.min(), x.max(), 100)
        line_y = m * line_x + b
        
        plt.figure(figsize=(6, 4))
        plt.scatter(x, y, alpha=0.6)
        plt.plot(line_x, line_y, linewidth=2)
        plt.xlabel(f'Count of "{kw}"')
        plt.ylabel('Polarity Score')
        plt.title(f'"{kw}" vs. {POLARITY_COL} (r={corrs[kw]:.2f})')
        plt.tight_layout()
        plt.show()
    
    # 7. Multiple regression
    X = sm.add_constant(df[present])
    y = df[POLARITY_COL]
    model = sm.OLS(y, X).fit()
    print("Multiple Regression Summary:\n")
    print(model.summary())


if __name__ == '__main__':
    main()
