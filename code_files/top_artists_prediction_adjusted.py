import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np

def plot_top_artists_prediction_adjusted(artist_metrics, output_dir='output'):
    """
    Create a refined visualization of predicted artist success with adjusted weights.
    """
    # Set style for clean look
    plt.style.use('seaborn-v0_8')

    # Create figure with better spacing
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='white', dpi=100)

    # Calculate prediction score with adjusted weights
    artist_metrics['prediction_score'] = (
        artist_metrics['max_weeks_on_chart'] * 0.4 +  # Reduced to 40%
        artist_metrics['avg_weeks_on_chart'] * 0.3 +
        artist_metrics['avg_spotify_popularity'] * 0.3  # Increased to 30%
    ).round(1)

    # Drop "The Weeknd & Ariana Grande" if it exists
    if "The Weeknd & Ariana Grande" in artist_metrics.index:
        artist_metrics = artist_metrics.drop("The Weeknd & Ariana Grande")

    # Get top 10 artists and sort in descending order
    top_10 = artist_metrics.nlargest(10, 'prediction_score').copy()
    top_10 = top_10.iloc[::-1]  # Reverse the order to have highest at top

    # Professional color scheme
    colors = {
        'bar': '#e67e22',       # Orange
        'text': '#2c3e50',      # Dark gray
        'grid': '#ecf0f1',      # Light gray
        'highlight': '#e74c3c'  # Accent red
    }

    # Create horizontal bar plot
    bars = ax.barh(y=range(len(top_10)),
                   width=top_10['prediction_score'],
                   color=colors['bar'],
                   alpha=0.8,
                   height=0.5)

    # Customize appearance
    plt.title('Top 10 Artists: Predicted Billboard Hot 100 Success (Adjusted Weights)',
             fontsize=14,
             color=colors['text'],
             pad=20,
             fontweight='bold')

    # Add subtitle with more margin
    ax.text(0.5, 1.08,
            'Based on Adjusted Historical Performance and Current Popularity',
            transform=ax.transAxes,
            ha='center',
            fontsize=11,
            color=colors['text'],
            style='italic')

    # Axis labels
    plt.xlabel('Predicted Weeks on Chart',
              fontsize=11,
              color=colors['text'],
              labelpad=10)
    plt.ylabel('Artist Name',
              fontsize=11,
              color=colors['text'],
              labelpad=10)

    # Set y-axis labels
    ax.set_yticks(range(len(top_10)))
    ax.set_yticklabels(top_10.index,
                       fontsize=10,
                       color=colors['text'])

    # Clean grid
    ax.grid(True,
            axis='x',
            color=colors['grid'],
            linestyle='-',
            linewidth=0.5,
            alpha=0.5)
    ax.set_axisbelow(True)

    # Add value labels
    for i, row in enumerate(top_10.iterrows()):
        # Predicted weeks
        ax.text(row[1]['prediction_score'] + 0.5,
                i,
                f"{row[1]['prediction_score']:.0f} weeks",
                va='center',
                ha='left',
                fontsize=10,
                color=colors['text'])

    # Add methodology note
    methodology = (
        "Adjusted Prediction Model Weights:\n"
        "• Historical Peak (40%)\n"
        "• Consistency (30%)\n"
        "• Spotify Popularity (30%)"
    )
    plt.figtext(0.02,
                0.02,
                methodology,
                fontsize=9,
                color=colors['text'],
                bbox=dict(facecolor='white',
                         edgecolor=colors['bar'],
                         alpha=0.9,
                         pad=10,
                         boxstyle='round,pad=0.5'))

    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(colors['grid'])
    ax.spines['bottom'].set_color(colors['grid'])

    # Set axis limits with padding
    ax.set_xlim(-2, max(top_10['prediction_score']) * 1.15)

    # Adjust layout
    plt.tight_layout()

    # Save with high quality to a new file
    plt.savefig(os.path.join(output_dir, 'artist_success_prediction_adjusted.png'),
                dpi=300,
                bbox_inches='tight',
                facecolor='white')
    plt.close()
