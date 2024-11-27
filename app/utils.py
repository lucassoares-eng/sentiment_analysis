import json
import os
import numpy as np

DATA_FOLDER = "data"
DEFAULT_DATASET = os.path.join(DATA_FOLDER, "tripadvisor_hotel_reviews.csv")
STATIC_FOLDER = "static"
RESULTS_FILE = os.path.join(STATIC_FOLDER, "results.json")

# Ensure data folder exists
os.makedirs(DATA_FOLDER, exist_ok=True)

def convert_to_serializable(obj):
    """
    Recursively convert non-serializable objects (e.g., np.int64, np.float64) into serializable Python types.
    """
    if isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    return obj

def save_results(results):
    """
    Save the analysis results as a JSON file in the static folder.
    
    :param results: dict, Analysis results to save
    """
    # Cria a pasta static se não existir
    if not os.path.exists(STATIC_FOLDER):
        os.makedirs(STATIC_FOLDER)
    
    # Converte o dicionário inteiro para tipos serializáveis
    serializable_results = convert_to_serializable(results)

    # Salva os resultados no arquivo JSON
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, indent=4, ensure_ascii=False)

def load_results():
    """
    Load results from the JSON file in the static folder.
    """
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return None