from flask import Blueprint, make_response, render_template, request, redirect, url_for
from .model import analyze, analyze_review
from .utils import convert_to_serializable, delete_file, generate_wordcloud, load_results, save_file, save_results

# Configure o Blueprint para usar uma pasta de templates específica
routes_bp = Blueprint(
    "sentiment_analysis",
    __name__,
    template_folder="templates",  # Caminho relativo para a pasta de templates do módulo
    static_folder="static",        # Caminho relativo para a pasta de arquivos estáticos do módulo     
    static_url_path="/static/sentiment_analysis"
)

@routes_bp.route('/css/<filename>')
def serve_css(filename):
    response = make_response(render_template(filename))
    response.headers['Content-Type'] = 'text/css'
    return response

@routes_bp.route("/")
def home():
    """
    Renders the home page with data from the results.json file.
    """
    results = load_results()
    # Run the analysis on the default dataset when the app starts
    if not results:
        results = analyze()
        save_results(results)
        print("Default analysis completed")
    generate_wordcloud(results['positive_common_words'], 'positive_wordcloud.png')
    generate_wordcloud(results['negative_common_words'], 'negative_wordcloud.png')
    if results:
        return render_template("index.html", **results)
    else:
        return "Results file not found!", 404

@routes_bp.route("/analyze-text", methods=["POST"])
def analyze_text():
    """
    Route to analyze a single text review and return the sentiment and star ratings.
    """
    data = request.json
    review = data.get("review", "")

    if not review:
        return {"error": "Review text is required!"}, 400

    try:
        result = analyze_review(review)  # Call to the function in model.py
        return result  # Return the data as JSON
    except Exception as e:
        return {"error": str(e)}, 500

@routes_bp.route("/upload", methods=["POST"])
def upload():
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

    return redirect(url_for("routes.home"))