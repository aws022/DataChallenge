import os
import sys
from success_score_distribution import plot_success_score_distribution
from top_artists_prediction import plot_top_artists_prediction
from top_artists_prediction_adjusted import plot_top_artists_prediction_adjusted
from spotify_chart_correlation import plot_spotify_chart_correlation
from hailmary import ArtistSuccessPredictor

def main():
    try:
        # Setup paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(script_dir)
        output_dir = os.path.join(project_dir, 'output')

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Initialize predictor
        hot_100_path = os.path.join(project_dir, "data_files", "hot_100_current.csv")
        rolling_stone_path = os.path.join(project_dir, "data_files", "rolling_stone.csv")

        predictor = ArtistSuccessPredictor(hot_100_path, rolling_stone_path)

        # Load and process data
        predictor.load_and_validate_data()
        artist_metrics = predictor.generate_artist_metrics()

        # Generate individual visualizations
        plot_success_score_distribution(artist_metrics, output_dir)
        plot_top_artists_prediction(artist_metrics, output_dir)
        plot_top_artists_prediction_adjusted(artist_metrics, output_dir)
        plot_spotify_chart_correlation(artist_metrics, output_dir)

        print("Visualizations generated successfully in the output directory!")

    except Exception as e:
        print(f"Error generating visualizations: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
