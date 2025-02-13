"""
Artist Success Trajectory Predictor

This module implements a data science model to predict artist success trajectories
based on historical performance data, Spotify popularity, and other metrics.

Key components:
- Data preprocessing and cleaning
- Feature engineering
- Model development
- Success prediction visualization
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict
import logging

# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

class ArtistSuccessPredictor:
    def __init__(self, hot_100_path: str, rolling_stone_path: str):
        """
        Initialize the predictor with paths to required datasets.

        Args:
            hot_100_path: Path to Billboard Hot 100 dataset
            rolling_stone_path: Path to Rolling Stone dataset
        """
        self.hot_100_path = os.path.abspath(hot_100_path)
        self.rolling_stone_path = os.path.abspath(rolling_stone_path)
        self.hot_100_df = None
        self.rolling_stone_df = None
        self.artist_metrics = None

        # Verify file existence
        if not os.path.exists(self.hot_100_path):
            raise FileNotFoundError(f"Hot 100 file not found at: {self.hot_100_path}")
        if not os.path.exists(self.rolling_stone_path):
            raise FileNotFoundError(f"Rolling Stone file not found at: {self.rolling_stone_path}")

    def load_and_validate_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load datasets and perform initial validation checks.
        """
        try:
            # Load datasets
            self.hot_100_df = pd.read_csv(self.hot_100_path)
            self.rolling_stone_df = pd.read_csv(self.rolling_stone_path)

            logging.info("\nAvailable columns in Hot 100:")
            logging.info(self.hot_100_df.columns.tolist())
            logging.info("\nAvailable columns in Rolling Stone:")
            logging.info(self.rolling_stone_df.columns.tolist())

            # Map common variations of column names - do this BEFORE normalization
            column_mappings_hot100 = {
                'performer': 'artist',
                'title': 'song',
                'wks_on_chart': 'wks_on_chart',
                'peak_pos': 'peak_position'
            }

            column_mappings_rs = {
                'Clean Name': 'artist',
                'Spotify Popularity': 'spotify_popularity'
            }

            # Rename columns - do this BEFORE normalization
            self.hot_100_df = self.hot_100_df.rename(columns=column_mappings_hot100)
            self.rolling_stone_df = self.rolling_stone_df.rename(columns=column_mappings_rs)

            # Convert numeric columns to proper type
            self.hot_100_df['wks_on_chart'] = pd.to_numeric(self.hot_100_df['wks_on_chart'], errors='coerce')
            self.hot_100_df['peak_position'] = pd.to_numeric(self.hot_100_df['peak_position'], errors='coerce')
            self.rolling_stone_df['spotify_popularity'] = pd.to_numeric(self.rolling_stone_df['spotify_popularity'], errors='coerce')

            # Now normalize column names
            self.hot_100_df.columns = self.hot_100_df.columns.str.lower().str.strip()
            self.rolling_stone_df.columns = self.rolling_stone_df.columns.str.lower().str.strip()

            # Validate required columns
            required_hot100_cols = ['artist', 'song', 'wks_on_chart', 'peak_position']
            required_rs_cols = ['artist', 'spotify_popularity']

            # Debug column names after transformation
            logging.info("\nTransformed Hot 100 columns:")
            logging.info(self.hot_100_df.columns.tolist())
            logging.info("\nTransformed Rolling Stone columns:")
            logging.info(self.rolling_stone_df.columns.tolist())

            missing_hot100 = [col for col in required_hot100_cols if col not in self.hot_100_df.columns]
            missing_rs = [col for col in required_rs_cols if col not in self.rolling_stone_df.columns]

            if missing_hot100:
                raise KeyError(f"Missing columns in Hot 100 dataset: {missing_hot100}")
            if missing_rs:
                raise KeyError(f"Missing columns in Rolling Stone dataset: {missing_rs}")

            # Basic data cleaning
            # Remove duplicates
            self.hot_100_df = self.hot_100_df.drop_duplicates()
            self.rolling_stone_df = self.rolling_stone_df.drop_duplicates(subset=['artist'])

            # Handle missing values
            self.hot_100_df['wks_on_chart'] = self.hot_100_df['wks_on_chart'].fillna(0)
            self.hot_100_df['peak_position'] = self.hot_100_df['peak_position'].fillna(101)  # Beyond chart position
            self.rolling_stone_df['spotify_popularity'] = self.rolling_stone_df['spotify_popularity'].fillna(0)

            # Add data quality checks
            logging.info("\nData Quality Metrics:")
            logging.info(f"Hot 100 Records: {len(self.hot_100_df)}")
            logging.info(f"Rolling Stone Records: {len(self.rolling_stone_df)}")
            logging.info(f"Unique Artists in Hot 100: {self.hot_100_df['artist'].nunique()}")
            logging.info(f"Unique Artists in Rolling Stone: {self.rolling_stone_df['artist'].nunique()}")

            logging.info("Data cleaning completed successfully")
            return self.hot_100_df, self.rolling_stone_df

        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            logging.error("\nPlease ensure your CSV files have these columns:")
            logging.error(f"Hot 100: {required_hot100_cols}")
            logging.error(f"Rolling Stone: {required_rs_cols}")
            raise

    def generate_artist_metrics(self) -> pd.DataFrame:
        """
        Calculate key metrics per artist from both datasets.

        Returns:
            DataFrame containing aggregated artist metrics
        """
        try:
            # Ensure numeric types
            numeric_cols = {
                'wks_on_chart': 'float64',
                'peak_position': 'float64'
            }

            for col, dtype in numeric_cols.items():
                if col in self.hot_100_df.columns:
                    self.hot_100_df[col] = self.hot_100_df[col].astype(dtype)

            # Hot 100 metrics with error handling
            hot100_metrics = self.hot_100_df.groupby('artist').agg({
                'wks_on_chart': ['max', 'mean'],
                'peak_position': 'min',
                'song': 'count'
            }).round(2)

            # Flatten column names
            hot100_metrics.columns = [
                'max_weeks_on_chart',
                'avg_weeks_on_chart',
                'best_peak_position',
                'total_songs'
            ]

            # Ensure spotify_popularity is numeric
            self.rolling_stone_df['spotify_popularity'] = pd.to_numeric(
                self.rolling_stone_df['spotify_popularity'],
                errors='coerce'
            ).fillna(0)

            # Rolling Stone metrics with error handling
            rs_metrics = self.rolling_stone_df.groupby('artist').agg({
                'spotify_popularity': 'mean'
            }).round(2)

            rs_metrics.columns = ['avg_spotify_popularity']

            # Merge metrics with outer join to keep all artists
            self.artist_metrics = hot100_metrics.join(
                rs_metrics,
                how='outer'
            ).fillna(0)  # Fill missing values with 0

            logging.info(f"Generated metrics for {len(self.artist_metrics)} artists")

            # Log some statistics about the generated metrics
            logging.info("\nMetrics Summary:")
            logging.info(self.artist_metrics.describe().round(2))

            return self.artist_metrics

        except Exception as e:
            logging.error(f"Error generating metrics: {str(e)}")
            logging.error("Full error details:", exc_info=True)
            raise

    def analyze_data_distribution(self) -> None:
        """
        Create and display distribution plots for key metrics.
        """
        metrics_to_plot = [
            'max_weeks_on_chart',
            'avg_spotify_popularity',
            'best_peak_position'
        ]

        fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(15, 5))

        for i, metric in enumerate(metrics_to_plot):
            sns.histplot(data=self.artist_metrics, x=metric, ax=axes[i])
            axes[i].set_title(f'Distribution of {metric}')
            axes[i].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()

    def engineer_features(self) -> pd.DataFrame:
        """
        Engineer advanced features for artist success prediction.

        Features include:
        - Time-based metrics (career longevity, consistency)
        - Performance trends
        - Chart position patterns
        - Success rate indicators

        Returns:
            DataFrame with engineered features
        """
        # Ensure we have base metrics
        if self.artist_metrics is None:
            self.generate_artist_metrics()

        # Convert chart_week to datetime
        self.hot_100_df['chart_week'] = pd.to_datetime(self.hot_100_df['chart_week'])

        # Calculate career span features
        career_metrics = self.hot_100_df.groupby('artist').agg({
            'chart_week': lambda x: (x.max() - x.min()).days / 365  # Career span in years
        }).round(2)
        career_metrics.columns = ['career_span_years']

        # Calculate consistency metrics
        artist_appearances = self.hot_100_df.groupby(['artist', pd.Grouper(key='chart_week', freq='Y')]).size()
        consistency_metrics = artist_appearances.groupby('artist').agg(['mean', 'std']).round(2)
        consistency_metrics.columns = ['yearly_chart_frequency', 'chart_frequency_std']

        # Calculate performance trends
        def calculate_trend(group):
            if len(group) < 2:
                return 0
            x = np.arange(len(group))
            y = group.values
            slope, _ = np.polyfit(x, y, 1)
            return slope

        recent_performance = self.hot_100_df.sort_values('chart_week').groupby('artist').agg({
            'peak_position': lambda x: calculate_trend(x.rolling(window=10, min_periods=1).mean())
        })
        recent_performance.columns = ['recent_trend']

        # Calculate success rate features
        success_metrics = self.hot_100_df.groupby('artist').agg({
            'peak_position': lambda x: (x <= 10).mean()  # Ratio of top 10 hits
        }).round(2)
        success_metrics.columns = ['top_10_hit_ratio']

        # Combine all engineered features
        engineered_features = pd.concat([
            self.artist_metrics,
            career_metrics,
            consistency_metrics,
            recent_performance,
            success_metrics
        ], axis=1).fillna(0)

        # Add genre diversity if available
        if 'genre' in self.hot_100_df.columns:
            genre_diversity = self.hot_100_df.groupby('artist')['genre'].nunique()
            engineered_features['genre_diversity'] = genre_diversity

        return engineered_features

    def analyze_feature_importance(self, features_df: pd.DataFrame) -> Dict[str, float]:
        """
        Analyze and visualize the importance of engineered features.

        Args:
            features_df: DataFrame containing engineered features

        Returns:
            Dictionary of feature importance scores
        """
        # Calculate correlation matrix
        correlation_matrix = features_df.corr()

        # Create correlation heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix,
                    annot=True,
                    cmap='coolwarm',
                    center=0,
                    fmt='.2f')
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()

        # Calculate feature importance relative to success metrics
        target_cols = ['max_weeks_on_chart', 'best_peak_position']
        importance_scores = {}

        for target in target_cols:
            feature_cols = [col for col in features_df.columns if col not in target_cols]
            correlations = features_df[feature_cols].corrwith(features_df[target]).abs()
            importance_scores[target] = correlations.to_dict()

        return importance_scores

    def create_success_visualization(self) -> None:
        """Create a comprehensive and visually appealing success visualization dashboard."""
        if self.artist_metrics is None:
            self.generate_artist_metrics()

            # Set modern style
            plt.style.use('dark_background')
            sns.set_theme(style="darkgrid", palette="husl")

            # Create figure with custom layout
            fig = plt.figure(figsize=(20, 24), facecolor='#1a1a1a')
            gs = fig.add_gridspec(4, 2, hspace=0.4, wspace=0.3)

            # Custom color palette
            colors = {
                'text': '#ffffff',
                'accent': '#00ff99',
                'background': '#1a1a1a',
                'grid': '#333333'
            }

            # 1. Success Score Distribution with KDE
            ax1 = fig.add_subplot(gs[0, 0])
            success_score = (self.artist_metrics['max_weeks_on_chart'] *
                            (100 - self.artist_metrics['best_peak_position']) / 100)
            sns.histplot(data=success_score, bins=50, kde=True, color=colors['accent'],
                        alpha=0.6, ax=ax1)
            ax1.set_title('Artist Success Score Distribution', fontsize=14, pad=20, color=colors['text'])
            ax1.set_xlabel('Success Score', fontsize=12, color=colors['text'])
            ax1.set_ylabel('Count', fontsize=12, color=colors['text'])
            ax1.tick_params(colors=colors['text'])

            # 2. Weeks on Chart vs Peak Position (Hexbin plot)
            ax2 = fig.add_subplot(gs[0, 1])
            plt.hexbin(self.artist_metrics['max_weeks_on_chart'],
                       self.artist_metrics['best_peak_position'],
                       gridsize=30, cmap='magma')
            ax2.set_title('Weeks on Chart vs Peak Position\n(Density Hexbin)',
                         fontsize=14, pad=20, color=colors['text'])
            ax2.set_xlabel('Maximum Weeks on Chart', fontsize=12, color=colors['text'])
            ax2.set_ylabel('Best Peak Position', fontsize=12, color=colors['text'])
            ax2.tick_params(colors=colors['text'])
            plt.colorbar(label='Count')

            # 3. Top 20 Artists (Horizontal bar with color gradient)
            ax3 = fig.add_subplot(gs[1, :])
            top_20 = self.artist_metrics.nlargest(20, 'max_weeks_on_chart')
            colors_gradient = sns.color_palette("viridis", n_colors=20)
            bars = sns.barplot(data=top_20, y=top_20.index, x='max_weeks_on_chart',
                              palette=colors_gradient, ax=ax3)
            ax3.set_title('Top 20 Artists by Weeks on Chart', fontsize=14, pad=20, color=colors['text'])
            ax3.set_xlabel('Maximum Weeks on Chart', fontsize=12, color=colors['text'])
            ax3.set_ylabel('Artist', fontsize=12, color=colors['text'])
            ax3.tick_params(colors=colors['text'])

            # Add value labels to bars
            for i, v in enumerate(top_20['max_weeks_on_chart']):
                ax3.text(v + 1, i, f'{v:.0f}', va='center', fontsize=10, color=colors['text'])

            # 4. Spotify Popularity vs Chart Performance (Scatter with trend)
            ax4 = fig.add_subplot(gs[2, 0])
            sns.scatterplot(data=self.artist_metrics,
                           x='avg_spotify_popularity',
                           y='max_weeks_on_chart',
                           alpha=0.5, color=colors['accent'], ax=ax4)
            sns.regplot(data=self.artist_metrics,
                        x='avg_spotify_popularity',
                        y='max_weeks_on_chart',
                        scatter=False,
                        color='red',
                        ax=ax4)
            ax4.set_title('Spotify Popularity vs Chart Performance',
                         fontsize=14, pad=20, color=colors['text'])
            ax4.set_xlabel('Average Spotify Popularity', fontsize=12, color=colors['text'])
            ax4.set_ylabel('Maximum Weeks on Chart', fontsize=12, color=colors['text'])
            ax4.tick_params(colors=colors['text'])

            # 5. Success Metrics Correlation Heatmap
            ax5 = fig.add_subplot(gs[2, 1])
            correlation_matrix = self.artist_metrics.corr()
            sns.heatmap(correlation_matrix,
                        annot=True,
                        cmap='coolwarm',
                        center=0,
                        fmt='.2f',
                        ax=ax5)
            ax5.set_title('Success Metrics Correlation', fontsize=14, pad=20, color=colors['text'])
            ax5.tick_params(colors=colors['text'])

            # 6. Career Longevity Analysis
            ax6 = fig.add_subplot(gs[3, :])
            career_data = self.artist_metrics.nlargest(10, 'total_songs')
            sns.barplot(data=career_data,
                        x=career_data.index,
                        y='total_songs',
                        palette='viridis',
                        ax=ax6)
            ax6.set_title('Top 10 Artists by Total Songs', fontsize=14, pad=20, color=colors['text'])
            ax6.set_xlabel('Artist', fontsize=12, color=colors['text'])
            ax6.set_ylabel('Total Songs', fontsize=12, color=colors['text'])
            ax6.tick_params(axis='x', rotation=45, colors=colors['text'])
            ax6.tick_params(axis='y', colors=colors['text'])

            # Add title and subtitle
            fig.suptitle('Artist Success Analysis Dashboard',
                        fontsize=24, y=0.95, color=colors['text'])
            plt.figtext(0.5, 0.92,
                        'Analysis of Billboard Hot 100 and Rolling Stone Data',
                        ha='center', fontsize=16, style='italic', color=colors['text'])

            # Add summary statistics as text
            summary_stats = f"""
            Total Artists Analyzed: {len(self.artist_metrics):,}
            Average Weeks on Chart: {self.artist_metrics['max_weeks_on_chart'].mean():.1f}
            Most Successful Artist: {top_20.index[0]}
            Total Songs Analyzed: {self.artist_metrics['total_songs'].sum():,.0f}
            """
            plt.figtext(0.02, 0.02, summary_stats, fontsize=12,
                        bbox=dict(facecolor=colors['background'],
                                 edgecolor=colors['accent'],
                                 alpha=0.8),
                        color=colors['text'])

            # Adjust layout and save
            plt.tight_layout(rect=[0, 0.03, 1, 0.90])

            # Create output directory if it doesn't exist
            output_dir = 'output'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Save with high DPI for better quality
            plt.savefig(os.path.join(output_dir, 'artist_success_analysis.png'),
                        dpi=300,
                        bbox_inches='tight',
                        facecolor=colors['background'])
            plt.close()

            logging.info("Enhanced visualization saved to output/artist_success_analysis.png")


if __name__ == "__main__":
    try:
        # Setup file paths relative to script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(script_dir)

        hot_100_path = os.path.join(project_dir, "data_files", "hot_100_current.csv")
        rolling_stone_path = os.path.join(project_dir, "data_files", "rolling_stone.csv")

        logging.info(f"Loading data from:\nHot 100: {hot_100_path}\nRolling Stone: {rolling_stone_path}")

        # Initialize predictor
        predictor = ArtistSuccessPredictor(hot_100_path, rolling_stone_path)

        # Load and validate data
        hot_100_df, rolling_stone_df = predictor.load_and_validate_data()
        logging.info("Data loaded successfully!")

        # Generate artist metrics
        artist_metrics = predictor.generate_artist_metrics()
        logging.info("\nTop 5 artists by max weeks on chart:")
        print(artist_metrics.nlargest(5, 'max_weeks_on_chart'))

        # Create visualization
        predictor.create_success_visualization()
        logging.info("Analysis complete! Check the output directory for visualizations.")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        logging.error("Stack trace:", exc_info=True)
        sys.exit(1)



