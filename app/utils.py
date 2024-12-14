import json
import os
import numpy as np
import pandas as pd
from wordcloud import WordCloud
import matplotlib
matplotlib.use('Agg')  # Usa o backend 'Agg', que não exige uma GUI
import matplotlib.pyplot as plt
import zipfile
import nltk

DATA_FOLDER = "data"
DEFAULT_DATASET = os.path.join(DATA_FOLDER, "tripadvisor_hotel_reviews.csv")
STATIC_FOLDER = "app/static"
ALLOWED_EXTENSIONS = {'csv'}
UPLOAD_FOLDER = "uploads"

# Ensure folders exist
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Define the location to download and store the data
nltk_data_dir = "./nltk_data"
nltk.data.path.append(nltk_data_dir)

# Path to the specific file
vader_file_path = os.path.join(nltk_data_dir, "sentiment", "vader_lexicon", "vader_lexicon.txt")

# Check if the file already exists
if not os.path.exists(vader_file_path):
    print("vader_lexicon.txt not found. Proceeding with download and extraction...")

    # Download the dataset
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
else:
    print("vader_lexicon.txt already exists. Skipping download.")

def convert_to_serializable(obj):
    """
    Recursively converts non-serializable objects (e.g., np.int64, np.float64) into serializable Python types.
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

def save_results(results, file_name="results.json"):
    """
    Save the analysis results as a JSON file in the static folder.
    
    :param results: dict, Analysis results to save.
    """
    # Create the static folder if it does not exist
    if not os.path.exists(STATIC_FOLDER):
        os.makedirs(STATIC_FOLDER)
    
    # Convert the entire dictionary to serializable types
    serializable_results = convert_to_serializable(results)

    # Save the results to a JSON file
    RESULTS_FILE = os.path.join(STATIC_FOLDER, file_name)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, indent=4, ensure_ascii=False)

def load_results(file_name="results.json"):
    """
    Load results from the JSON file in the static folder.
    """
    RESULTS_FILE = os.path.join(STATIC_FOLDER, file_name)
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

# Function to verify allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to load data from a file
def load_file(dataset_path):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"The dataset at {dataset_path} does not exist.")
    
    try:
        # Attempt to load the dataset with UTF-8 encoding
        data = pd.read_csv(dataset_path, encoding='utf-8')
    except UnicodeDecodeError:
        # Fallback to 'latin1' encoding in case of a decoding error
        data = pd.read_csv(dataset_path, encoding='latin1')

    # Rename the first column to "Review"
    data.rename(columns={data.columns[0]: "Review"}, inplace=True)
    
    return data

# Function to delete a file after processing
def delete_file(file_path):
    """Deletes the specified file."""
    if os.path.exists(file_path):
        os.remove(file_path)

def save_file(uploaded_file):
    # Save the file temporarily
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.filename)
    uploaded_file.save(file_path)
    return file_path

def generate_wordcloud(word_data, file_name):
    word_freq = convert_to_wordcloud_format(word_data)
    
    # Condicional para escolher o colormap com base no nome do arquivo
    if "positive" in file_name.lower():
        colormap = 'Greens'  # Usar tons de verde para palavras positivas
    else:
        colormap = 'Reds'  # Usar tons de vermelho para palavras negativas

    # Gerando a nuvem de palavras com a cor especificada
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap=colormap).generate_from_frequencies(word_freq)
    
    output_file = os.path.join(STATIC_FOLDER, file_name)
    
    # Gera a nuvem de palavras e salva a imagem no arquivo
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # Remove o eixo
    plt.savefig(output_file, format='png')  # Salva a imagem
    plt.close()

# Converte a lista de dicionários para o formato esperado
def convert_to_wordcloud_format(data):
    return {item["word"]: item["count"] for item in data}