import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_spotify_chart_correlation(artist_metrics, output_dir='output'):
    """Create a clean correlation plot between Spotify popularity and chart success."""
    plt.style.use('seaborn-v0_8')

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')

    # Color scheme
    colors = {
        'points': '#3498db',
        'line': '#e74c3c',
        'text': '#2c3e50',
        'grid': '#ecf0f1'
    }

    # Create scatter plot
    sns.scatterplot(data=artist_metrics,
                   x='avg_spotify_popularity',
                   y='max_weeks_on_chart',
                   alpha=0.5,
                   color=colors['points'])

    # Add trend line
    sns.regplot(data=artist_metrics,
                x='avg_spotify_popularity',
                y='max_weeks_on_chart',
                scatter=False,
                color=colors['line'])

    # Customize appearance
    plt.title('Spotify Popularity vs Chart Performance',
             fontsize=14,
             color=colors['text'],
             pad=20)

    plt.xlabel('Spotify Popularity Score',
              fontsize=11,
              color=colors['text'])
    plt.ylabel('Maximum Weeks on Chart',
              fontsize=11,
              color=colors['text'])

    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Save plot
    plt.savefig(os.path.join(output_dir, 'spotify_chart_correlation.png'),
                dpi=300,
                bbox_inches='tight',
                facecolor='white')
    plt.close()
