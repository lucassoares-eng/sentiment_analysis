from flask import render_template, request, redirect, url_for, send_from_directory
from app.model import analyze
from app.utils import convert_to_serializable, delete_file, generate_wordcloud, load_results, save_file
from app import app  # Imports the pre-initialized app from the main package

@app.route('/static/<filename>')
def serve_wordcloud(filename):
    return send_from_directory('static', filename)

@app.route("/")
def home():
    """
    Renders the home page with data from the results.json file.
    """
    results = load_results()
    generate_wordcloud(results['positive_common_words'], 'positive_wordcloud.png')
    generate_wordcloud(results['negative_common_words'], 'negative_wordcloud.png')
    if results:
        return render_template("index.html", **results)
    else:
        return "Results file not found!", 404

@app.route("/upload", methods=["POST"])
def upload_and_analyze():
    """
    Route to receive the user's file, analyze it, and render the analysis page.
    """
    # Checks if a file was sent in the request
    if "file" not in request.files:
        return "No file part in the request!", 400

    uploaded_file = request.files["file"]

    # Checks if the file has a valid name
    if uploaded_file.filename == "":
        return "No file selected!", 400

    if uploaded_file:

        file_path = save_file(uploaded_file)
        # Loads and analyzes the file's data
        try:
            results = analyze(file_path)
        finally:
            # Ensures the temporary file is deleted after analysis
            delete_file(file_path)

        # Renders the analysis page with the processed data
        if results:
            results = convert_to_serializable(results)
            generate_wordcloud(results['positive_common_words'], 'positive_wordcloud.png')
            generate_wordcloud(results['negative_common_words'], 'negative_wordcloud.png')
            return render_template("index.html", **results)

    return redirect(url_for("home"))