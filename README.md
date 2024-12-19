# Sentiment Analysis Application

This project is a Flask-based web application designed for sentiment analysis of reviews. It showcases a complete data processing pipeline, leveraging natural language processing (NLP) techniques to analyze customer feedback. The application calculates metrics such as the Net Promoter Score (NPS), identifies common words in reviews, and extracts the most relevant positive and negative comments. Results are dynamically visualized on an intuitive, modern homepage.

## Key Features

- Sentiment Analysis: Analyzes customer reviews and classifies sentiment (positive, negative, or neutral).
- NPS Calculation: Computes the Net Promoter Score (NPS) based on user reviews to gauge customer loyalty.
- Word Frequency Analysis: Identifies and visualizes common words in reviews, helping to spot trends and customer concerns.
- Relevant Comment Extraction: Extracts the most relevant positive and negative reviews for further analysis.
- Custom Dataset Upload: Allows users to upload custom datasets and re-run the analysis.
- Interactive Visualizations: Includes wordclouds and sentiment graphs for an engaging user experience.

## Technologies Used

This project was built with the following technologies:

- Python 3.8+: The programming language used for the application and data analysis.
- Flask: A lightweight web framework for Python, used to build the web application.
- scikit-learn: Used for natural language processing and sentiment analysis.
- NumPy and pandas: Utilized for efficient data manipulation and analysis.
- NLTK and spaCy: Used for text processing, tokenization, and linguistic analysis.
- Matplotlib and wordcloud: Libraries for creating visualizations (e.g., word clouds).

### Python Dependencies

Ensure you install the following Python dependencies:

```plaintext
Flask==3.1.0
nltk==3.9.1
pandas==2.2.3
langdetect==1.0.9
googletrans==4.0.0-rc1
matplotlib==3.9.2
wordcloud==1.9.4
scikit-learn==1.5.2
numpy==2.0.2
tqdm==4.67.0
```

## How to Run the Application

1. Clone the repository:
```bash
    git clone https://github.com/lucassoares-eng/sentiment_analysis.git
```

2. Install the dependencies:
```bash
    pip install -r requirements.txt
```

3. Start the application:
```bash
    python app.py
```

4. Access the app: Open your browser and navigate to http://127.0.0.1:5000 to view the application.

## How to Use

- **On startup**, the default dataset (`tripadvisor_hotel_reviews.csv`) is analyzed. You can download it from [Kaggle: Trip Advisor Hotel Reviews](https://www.kaggle.com/datasets/andrewmvd/trip-advisor-hotel-reviews).
- The analysis results (including sentiment scores, word frequency analysis, and word clouds) are displayed on the homepage.

### Upload Your Own Dataset:
1. Upload a custom CSV file via the interface.
2. The app will analyze the new dataset and update the results on the page.

---

## API Endpoints

### `/` (GET):
- Renders the homepage with the sentiment analysis results and visualizations.

### `/analyze-text` (POST):
- Analyzes a single review and returns the sentiment and score.

**Example request**:
```json
{
    "review": "The hotel was amazing, had a great time!"
}
```

### `/upload` (POST):
- Allows users to upload a CSV file containing hotel reviews.
- Returns the sentiment analysis results for the uploaded dataset.

## Customization

### Changing the Dataset

Replace the default dataset file (`tripadvisor_hotel_reviews.csv`) with your dataset in the expected format:

- **Columns required:** `Review`

### Modifying Stop Words

Update the stop word list in `model.py` for a more tailored analysis.

---

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
