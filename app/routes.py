from flask import render_template, request, redirect, url_for
from app.model import analyze
from app.utils import delete_file, load_file, load_results, save_file
from app import app  # Importa o app já inicializado no pacote principal

@app.route("/")
def home():
    """
    Renderiza a página inicial com os dados do arquivo results.json.
    """
    results = load_results()
    if results:
        return render_template("index.html", **results)
    else:
        return "Results file not found!", 404
    
@app.route("/upload", methods=["POST"])
def upload_and_analyze():
    """
    Rota para receber o arquivo do usuário, analisá-lo e renderizar a página de análise.
    """
    # Verifica se o arquivo foi enviado na requisição
    if "file" not in request.files:
        return "No file part in the request!", 400

    uploaded_file = request.files["file"]

    # Verifica se o arquivo possui um nome válido
    if uploaded_file.filename == "":
        return "No file selected!", 400

    if uploaded_file:

        file_path = save_file(uploaded_file)
        # Carrega e analisa os dados do arquivo
        try:
            results = analyze(file_path)
        finally:
            # Garante que o arquivo temporário será excluído após a análise
            delete_file(file_path)

        # Renderiza a página de análise com os dados processados
        if results:
            return render_template("index.html", **results)

    return redirect(url_for("home"))