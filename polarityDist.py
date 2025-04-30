import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    # Determine path of this script and locate the CSV in the same folder
    base_dir = os.path.dirname(os.path.realpath(__file__))
    csv_filename = 'sentiment_articles_output - Sheet5.csv'
    csv_path = os.path.join(base_dir, csv_filename)

    # Load the data file
    df = pd.read_csv(csv_path)

    # Ensure polarity scores are numeric and drop invalid entries
    df['Polarity Score'] = pd.to_numeric(df['Polarity Score'], errors='coerce')
    scores = df['Polarity Score'].dropna()

    # Plot histogram
    plt.figure(figsize=(8, 6))
    plt.hist(scores, bins=30)
    plt.title('Polarity Score Distribution')
    plt.xlabel('Polarity Score')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()

    # Save or show the plot
    output_path = os.path.join(base_dir, 'polarity_distribution.png')
    plt.savefig(output_path, dpi=300)
    plt.show()

if __name__ == '__main__':
    main()
