import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_success_score_distribution(artist_metrics, output_dir='output'):
    """Create a clean distribution plot of artist success scores."""
    plt.style.use('seaborn-v0_8')

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')

    # Calculate success score
    success_score = (artist_metrics['max_weeks_on_chart'] *
                    (100 - artist_metrics['best_peak_position']) / 100)

    # Color scheme
    colors = {
        'main': '#3498db',
        'text': '#2c3e50',
        'grid': '#ecf0f1'
    }

    # Create distribution plot
    sns.histplot(data=success_score,
                bins=50,
                color=colors['main'],
                alpha=0.7)

    # Add KDE line
    sns.kdeplot(data=success_score,
                color=colors['text'],
                linewidth=1.5)

    # Customize appearance
    plt.title('Distribution of Artist Success Scores',
             fontsize=14,
             color=colors['text'],
             pad=20)

    plt.xlabel('Success Score',
              fontsize=11,
              color=colors['text'])
    plt.ylabel('Number of Artists',
              fontsize=11,
              color=colors['text'])

    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Save plot
    plt.savefig(os.path.join(output_dir, 'success_score_distribution.png'),
                dpi=300,
                bbox_inches='tight',
                facecolor='white')
    plt.close()
