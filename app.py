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