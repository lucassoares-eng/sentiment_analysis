from app import app  # Importa o app inicializado no __init__.py
from app.utils import import_nltk, load_results, save_results
from app.model import analyze

import_nltk()

# Analisa os resultados ao iniciar o servidor
results = load_results()
if not results:
    results = analyze()
    save_results(results)
    print("Default analysis completed")