#!/usr/bin/env python3
"""
avg_polarity_by_company.py

Compute and plot the average sentiment polarity for each company as a bar chart,
with value labels on top of each bar—and automatically expand the y-axis
so labels don’t run outside the plot.

Usage:
    python avg_polarity_by_company.py path/to/your/sentiment.csv

If no path is given, it defaults to 'sentiment_articles_output - histogram.csv'.
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # 1. Determine CSV path
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'sentiment_articles_output - histogram.csv'
    
    # 2. Load and clean
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    
    # 3. Drop rows without a valid polarity score
    df = df.dropna(subset=['Polarity Score'])
    
    # 4. Normalize company names
    df['Company_norm'] = df['Company'].astype(str).str.strip()
    
    # 5. Compute the average polarity per company
    mean_polarity = (
        df.groupby('Company_norm')['Polarity Score']
          .mean()
          .sort_values()
    )
    
    # 6. Plot bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(mean_polarity.index, mean_polarity.values, edgecolor='gray')
    
    # 7. Expand y-axis to make room for labels
    top = mean_polarity.max()
    ax.set_ylim(0, top * 1.15)  # 15% headroom
    
    # 8. Annotate each bar with its value
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + (top * 0.02),  # a small offset above the bar
            f"{height:.2f}",
            ha='center',
            va='bottom',
            fontsize=9
        )
    
    # 9. Formatting
    ax.set_xlabel('Company')
    ax.set_ylabel('Average Polarity Score')
    ax.set_title('Average Sentiment Polarity by Company')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
