from flask import render_template
from app.utils import load_results
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