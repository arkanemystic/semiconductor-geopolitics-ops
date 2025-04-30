#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load and clean
df = pd.read_csv('sentiment_articles_output - histogram.csv')
df.columns = df.columns.str.strip()

# 2. Normalize company names
df['_company_norm'] = df['Company'].astype(str).str.strip().str.lower()

# 3. Drop rows without a valid Polarity Score
df = df.dropna(subset=['Polarity Score'])

# 4. Split external vs. others
ext_mask = df['_company_norm'] == 'external'
external_scores = df.loc[ext_mask, 'Polarity Score']
other_scores    = df.loc[~ext_mask, 'Polarity Score']

# 5. Take a random sample of 24 from the "other companies"
other_sample = other_scores.sample(n=24, random_state=42)

# 6. Plot side-by-side boxplots of the sampled others vs. external
plt.figure(figsize=(8, 5))
plt.boxplot(
    [other_sample, external_scores],
    labels=['Other Companies (n=24)', 'External'],
    patch_artist=True,
    medianprops={'color':'black'}
)
plt.ylabel('Polarity Score')
plt.title('Polarity Scores: Sampled Other Companies vs. External')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
