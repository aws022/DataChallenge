# Run Instructions for Artist Success Trajectory Predictor

## Prerequisites
1. Ensure you have Python installed on your system.
2. Install the required packages using the following command:
   ```
   pip install -r requirements.txt
   ```
   (Make sure `requirements.txt` includes all necessary libraries like `matplotlib`, `seaborn`, `pandas`, etc.)

## Running the Model
1. Navigate to the project directory in your terminal.
2. Activate your virtual environment if you are using one:
   ```
   source .venv/bin/activate  # On macOS/Linux
   .venv\Scripts\activate     # On Windows
   ```
3. Run the visualization script:
   ```
   python code_files/generate_visualizations.py
   ```

## Output
- The visualizations will be generated in the `output` directory.
- Two key visualizations will be created:
  - `artist_success_prediction.png`: Original weights.
  - `artist_success_prediction_adjusted.png`: Adjusted weights with higher Spotify popularity.

## Recommendations
- Use the `artist_success_prediction_adjusted.png` for presentations as it reflects current trends better.
- Review the `summary.txt` for a detailed explanation of the project and model.

## Troubleshooting
- Ensure all data files are in the `data_files` directory.
- Check for any error messages in the terminal and ensure all dependencies are installed.
