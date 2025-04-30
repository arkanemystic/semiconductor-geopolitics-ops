# semiconductor-geopolitics-ops

> **Understanding complexities of business operations in the semiconductor industry amid China–Taiwan tensions**

---

## 📖 Project Overview
This repository explores how third‑party media outlets frame the semiconductor industry in light of geopolitical tensions between China and Taiwan. By scraping, cleaning, and analyzing article text, we apply sentiment scoring, keyword extraction, and network analyses to uncover narrative clusters, sentiment drivers, and strategic insights.

Key objectives:
- Quantify positive vs. negative sentiment trends in industry coverage
- Identify which terms co‑occur and how they cluster
- Reveal which keywords most strongly influence sentiment
- Compare framing across different companies or outlets

## 🔍 Data Sources
- **`sentiment_articles_output - Sheet5.csv`**: Core dataset containing:
  - `Company`, `URL`, `Sentiment` label, `Polarity Score`
  - Precomputed keyword mention counts (e.g. `taiwan`, `risk`, `innovation`, …)

You can regenerate or expand this file by running the scraping pipeline (not included here).

## 🛠️ Installation & Dependencies
```bash
# Clone this repo
git clone https://github.com/<your-org>/semiconductor-geopolitics-ops.git
cd semiconductor-geopolitics-ops

# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate    # Windows

# Install required libraries
pip install -r requirements.txt
```

**`requirements.txt`** should include:
```
pandas
networkx
matplotlib
```

## 🚀 Usage
Ensure `sentiment_articles_output - Sheet5.csv` is in the project root.

### 1) Polarity Distribution
Generate and save a histogram of article polarity scores:
```bash
python polarityDist.py
# Outputs: polarity_distribution.png
```

### 2) Keyword Co‑Occurrence Network
Build and visualize a co‑mention network, with node size/color reflecting normalized sentiment:
```bash
python map.py
# Outputs: keyword_cooccurrence_network.png
```

### 3) Additional Analyses (Optional)
- **Keyword–Polarity Correlation Heatmap**: Compute correlations between keyword frequencies and polarity scores.  
- **Sentiment Boxplots**: Compare polarity distributions for articles mentioning specific terms.  
- **Time Series Analysis**: Plot average sentiment over time (requires `date` column).  

Scripts for these can be added in `analysis/`.

## 📈 Results & Insights
- **Histogram** reveals overall skew toward neutral or polarized coverage.  
- **Network Graph** highlights clusters around “tech supply‑chain” vs. “risk/threat” narratives.  
- **Normalized node shading** (0 → light yellow, 1 → deep purple) shows which keywords drive the strongest sentiment.

*(See `docs/` for embedded figures and slide decks.)*

## 🔮 Future Work
- Integrate **time‑series overlays** of major geopolitical events.
- Expand scraping to include **social media** or **forum discussions**.
- Use **topic modeling** (LDA) to uncover latent themes beyond predefined keywords.
- Incorporate **entity resolution** to distinguish between different semiconductor firms (TSMC vs. Samsung vs. Intel).

## 🤝 Contributing
Feel free to open issues or submit pull requests to improve analysis, add new visualizations, or refine data processing.

## 📜 License
This project is released under the MIT License. See [LICENSE](LICENSE.md) for details.

