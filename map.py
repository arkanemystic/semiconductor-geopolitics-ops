import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

# === Keyword Co-occurrence Network Script with Reversed Color Scale ===
def main():
    # Locate CSV in the same folder
    base = os.path.dirname(os.path.realpath(__file__))
    csv_path = os.path.join(base, 'sentiment_articles_output - Sheet5.csv')

    # Load data
    df = pd.read_csv(csv_path)
    df['Polarity Score'] = pd.to_numeric(df['Polarity Score'], errors='coerce')

    # Multiply polarity by 2 for moderate exaggeration
    df['Polarity Score'] *= 2

    # Keywords list
    keywords = [
        'taiwan', 'china', 'semiconductor', 'supply chain', 'risk', 'threat', 'growth',
        'chip', 'factory', 'production', 'shortage', 'investment', 'manufacturing',
        'innovation', 'disruption', 'global', 'market', 'demand', 'silicon', 'foundry'
    ]

    # Build presence and co-occurrence matrix
    presence = df[keywords] > 0
    cooc = presence.T.dot(presence)

    # Compute average polarity per keyword
    avg_pol = {kw: df.loc[presence[kw], 'Polarity Score'].mean() or 0 for kw in keywords}

    # Determine max absolute polarity for normalization
    max_abs_pol = max(abs(pol) for pol in avg_pol.values()) or 1

    # Normalized polarity magnitude (0 to 1)
    norm_pol = {kw: abs(pol) / max_abs_pol for kw, pol in avg_pol.items()}

    # Build graph
    G = nx.Graph()
    for kw in keywords:
        G.add_node(kw, avg_pol=avg_pol[kw], norm_pol=norm_pol[kw])
    for i, k1 in enumerate(keywords):
        for k2 in keywords[i+1:]:
            w = int(cooc.loc[k1, k2])
            if w > 0:
                G.add_edge(k1, k2, weight=w)

    # Layout
    pos = nx.spring_layout(G, seed=42)

    # Styling parameters
    node_vals = [G.nodes[n]['norm_pol'] for n in G.nodes()]
    node_sizes = [600 + G.nodes[n]['norm_pol'] * 20000 for n in G.nodes()]
    edge_widths = [d['weight'] * 1.5 for _, _, d in G.edges(data=True)]

    # Use a reversed Viridis colormap: yellow at 0, purple at 1
    cmap = plt.cm.viridis_r
    vmin, vmax = 0, 1

    # Draw network
    plt.figure(figsize=(12, 10))
    nodes = nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color=node_vals,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolors='white',
        linewidths=1,
        alpha=0.9
    )
    nx.draw_networkx_edges(
        G, pos,
        width=edge_widths,
        alpha=0.6,
        edge_color='#CCCCCC'
    )
    nx.draw_networkx_labels(
        G, pos,
        font_size=10,
        font_color='white'
    )

    # Add colorbar
    ax = plt.gca()
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Normalized Polarity Magnitude (0 to 1)')

    # Final touches
    plt.title('Keyword Co-occurrence Network (Color Scale: Yellow→Purple)')
    plt.axis('off')
    plt.tight_layout()

    # Save and show
    out_png = os.path.join(base, 'keyword_cooccurrence_network.png')
    plt.savefig(out_png, dpi=300)
    plt.show()

if __name__ == '__main__':
    main()
