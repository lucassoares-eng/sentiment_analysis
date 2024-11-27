# Sentiment Analysis Application

This project is a Flask-based web application that performs sentiment analysis on hotel reviews. It calculates metrics such as the Net Promoter Score (NPS), identifies common words, and extracts the most relevant positive and negative comments. Results are visualized on a modern homepage.

## Features

- Sentiment analysis of reviews with NPS calculation.
- Identification of common words from reviews with word frequency.
- Extraction of most relevant positive and negative comments.
- Dynamic support for uploading custom review datasets.
- Persistent storage of results for quick access.

## Project Structure

```plaintext
project/
├── app/
│   ├── __init__.py        # Package initialization
│   ├── routes.py          # Application routes and endpoints
│   ├── model.py           # Sentiment analysis and data processing logic
│   ├── utils.py           # Utility functions for loading and saving results
│
├── static/
│   ├── results.json       # Saved analysis results
│
├── templates/
│   ├── index.html         # Homepage template for displaying analysis
│
├── app.py                 # Application entry point
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
```

## Requirements

The project is built with the following tools and libraries:

- **Python 3.8+**
- **Flask** for web development
- **scikit-learn** for NLP
- **NumPy** and **pandas** for data manipulation
- **Jinja2** for templating

### Python Dependencies

Ensure you install the following Python dependencies:

```plaintext
Flask==3.1.0
nltk==3.9.1
spacy==3.8.2
pandas==2.2.3
langdetect==1.0.9
googletrans==4.0.0-rc1
matplotlib==3.9.2
wordcloud==1.9.4
httpcore==0.9.1
seaborn==0.13.2
```

## How to Run the Application

1. Clone the repository:
```bash
    git clone https://github.com/lucassoares-eng/sentiment-analysis.git
```

2. Install the dependencies:
```bash
    pip install -r requirements.txt
```

3. Start the application:
```bash
    python app.py
```

4. Open the application in your browser at http://127.0.0.1:5000

## Usage

- On application startup, the default dataset (`tripadvisor_hotel_reviews.csv`) is analyzed. The dataset is sourced from [Kaggle: Trip Advisor Hotel Reviews](https://www.kaggle.com/datasets/andrewmvd/trip-advisor-hotel-reviews).
- The results are saved in `static/results.json`.
- Navigate to the homepage to view analysis results.
- To analyze a new dataset:
  1. Upload a file via the provided interface.
  2. The app will re-run the analysis and update the results.

---

## Customization

### Changing the Dataset

Replace the default dataset file (`tripadvisor_hotel_reviews.csv`) with your dataset in the expected format:

- **Columns required:** `Review`

### Modifying Stop Words

Update the stop word list in `model.py` for a more tailored analysis.

---

## Troubleshooting

### TemplateNotFound Error

- Ensure the `templates` folder is correctly named and located in the root of the project.
- Verify `template_folder` is properly set in `app/routes.py`.

### TypeError: Object Not JSON Serializable

- Ensure that all non-serializable objects (e.g., `np.int64`) are converted using the provided `convert_to_serializable` function.

## **Contributions**

Contributions are welcome! Feel free to open issues and submit pull requests to improve the project.

---

## **License**

This project is licensed under the APACHE License. See the `LICENSE` file for more details.

---

## **Contact**

If you have questions or suggestions, feel free to reach out:

Email: lucasjs.eng@gmail.com

LinkedIn: https://www.linkedin.com/in/lucas-de-jesus-soares-33486b42/