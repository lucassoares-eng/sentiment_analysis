import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from googletrans import Translator
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter

# Initialize the VADER sentiment analyzer for English
sia = SentimentIntensityAnalyzer()

# Baixe a lista de stop words
nltk.download("stopwords")
nltk_stop_words = set(stopwords.words("english"))

DEFAULT_DATASET = os.path.join("data", "tripadvisor_hotel_reviews.csv")

def analyze_sentiment(text):
    """
    Analyze the sentiment of a given text in English.
    
    :param text: str, Input text to analyze
    :return: str, Sentiment label (Positive, Negative, Neutral)
             dict, Sentiment scores
    """
    # Use VADER
    scores = sia.polarity_scores(text)
    return scores

def classify_sentiment_from_star_rating(star_rating):
    """
    Classify the sentiment based on the star rating.
    
    :param star_rating: float, Star rating between 1 and 5.
    :return: str, Sentiment label.
    """
    if star_rating > 0 and star_rating <= 2:
        return "Negative"
    elif star_rating > 2 and star_rating < 4:
        return "Neutral"
    elif star_rating >= 4:
        return "Positive"
    else:
        return "Undefined"
    
def calculate_star_rating(scores):
    """
    Calculate a star rating between 1 and 5 based on sentiment scores.
    
    :param scores: dict, Sentiment scores from VADER.
    :return: float, Star rating between 1 and 5.
    """
    # Weighted average of sentiment scores
    rating = 1 + (scores["pos"] * 4)  # Map "pos" to range 1-5
    return round(rating, 1)

def detect_original_language(reviews):
    """
    Determine the predominant language from the first 10 reviews.
    
    :param reviews: list, List of reviews
    :return: str, Detected predominant language code (e.g., "en", "pt")
    """
    detected_languages = []
    for review in reviews[:10]:
        try:
            detected_languages.append(detect(review))
        except LangDetectException:
            continue  # Skip reviews where language detection fails
    if detected_languages:
        # Return the most common language detected
        return Counter(detected_languages).most_common(1)[0][0]
    return "en"  # Default to English if detection fails

def analyze(dataset_path=DEFAULT_DATASET):
    """
    Analyzes the sentiment of hotel reviews, calculates NPS score, and returns
    detailed results, including most common words, sentiment distribution,
    number of reviews, detractors, promoters, and most relevant comments.
    
    Args:
        dataset_path (str): Path to the dataset file (.csv).
    
    Returns:
        dict: Results of the analysis.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"The dataset at {dataset_path} does not exist.")
    
    # Load the dataset
    data = pd.read_csv(dataset_path)
    
    # Filter only the 'Review' column
    if "Review" not in data.columns:
        raise ValueError("The dataset must have a 'Review' column.")
    
    reviews = data["Review"]
    sentiments = []
    sentiment_scores = []
    star_ratings = []

    # Detect the predominant language
    original_language = detect_original_language(reviews)

    # Initialize translator
    translator = Translator()

    # Analyze sentiment for each review
    for review in reviews:
        try:
            # Translate non-English reviews to English
            if original_language != "en":
                review = translator.translate(review, src=original_language, dest="en").text
            scores = analyze_sentiment(review)
            star_rating = calculate_star_rating(scores)  # Calculate star rating
            star_ratings.append(star_rating)
            sentiment = classify_sentiment_from_star_rating(star_rating)  # Classify sentiment
            sentiments.append(sentiment)
            sentiment_scores.append(scores)
        except Exception as e:
            print(f"Error analyzing review: {e}")
            sentiments.append("Error")
            sentiment_scores.append({})
            star_ratings.append(3)  # Default to 3 star in case of error

    data["sentiment"] = sentiments
    data["sentiment_scores"] = sentiment_scores
    data["star_rating"] = star_ratings

    # Calculate NPS (Net Promoter Score)
    positive_count = (data["sentiment"] == "Positive").sum()
    neutral_count = (data["sentiment"] == "Neutral").sum()
    negative_count = (data["sentiment"] == "Negative").sum()
    total_reviews = len(data)
    nps_score = ((positive_count - negative_count) / total_reviews) * 100

    # Detractors, Passives, and Promoters
    detractors = data[data["sentiment"] == "Negative"]
    passives = data[data["sentiment"] == "Neutral"]
    promoters = data[data["sentiment"] == "Positive"]

    # Generate most common words
    # Count the frequency of words in reviews
    stop_words = list(nltk_stop_words)
    vectorizer = CountVectorizer(stop_words=stop_words)
    word_counts = vectorizer.fit_transform(reviews)
    word_sum = word_counts.sum(axis=0)
    words_freq = [
        (word, word_sum[0, idx])
        for word, idx in vectorizer.vocabulary_.items()
    ]
    most_common_words = sorted(words_freq, key=lambda x: x[1], reverse=True)[:10]

    # Convert most common words to original language
    original_language_words = []
    for word, count in most_common_words:
        try:
            if original_language != "en":
                # Convert to original language
                original_word = translator.translate(word, src="en", dest=original_language)
            else:
                original_word = word
            original_language_words.append({"word": original_word, "count": count})
        except Exception as e:
            print(f"Error translating word '{word}': {e}")
            original_language_words.append({"word": word, "count": count})

    # Ordenar os dados com base nas estrelas
    sorted_by_star = data.sort_values(by="star_rating", ascending=False)

    # Extract most relevant comments based on star ratings
    most_relevant_positive = sorted_by_star.head(3)["Review"].tolist()
    most_relevant_negative = sorted_by_star.tail(3)["Review"].tolist()

    # Calculate the overall star rating (average)
    overall_star_rating = round(sum(star_ratings) / len(star_ratings), 2)

    # Prepare results to be returned
    results = {
        "nps_score": round(nps_score, 2),
        "total_reviews": total_reviews,
        "promoters_count": len(promoters),
        "detractors_count": len(detractors),
        "neutral_count": len(passives),
        "common_words": original_language_words,
        "most_relevant_comments": {
            "positive": most_relevant_positive,
            "negative": most_relevant_negative
        },
        "star_ratings": overall_star_rating,  # Return only the overall rating
    }

    return results
