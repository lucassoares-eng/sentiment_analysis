import os
import zipfile
import nltk

# Define the location to download and store the data
nltk_data_dir = "./nltk_data"
nltk.data.path.append(nltk_data_dir)
nltk.download('vader_lexicon', download_dir=nltk_data_dir)

# Path to the downloaded ZIP file
zip_path = os.path.join(nltk_data_dir, "sentiment", "vader_lexicon.zip")
extract_to = os.path.join(nltk_data_dir, "sentiment")

# Manually extract if the ZIP file exists
if os.path.exists(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        print(f"Files extracted to {extract_to}")
else:
    print(f"Error: {zip_path} not found!")

from app.routes import app
from app.model import analyze
from app.utils import load_results, save_results

if __name__ == "__main__":
    results = load_results()
    # Run the analysis on the default dataset when the app starts
    if not results:
        results = analyze()
        save_results(results)
        print("Default analysis completed")

    # Start the Flask server
    app.run(debug=True)