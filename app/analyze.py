from sklearn.feature_extraction.text import CountVectorizer
import nltk
import re
from nltk.corpus import stopwords
from googletrans import Translator
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter

if __name__ == "__main__":
    # Download the list of stop words
    nltk.download("stopwords")

# Initialize the VADER sentiment analyzer for English
sia = SentimentIntensityAnalyzer()

def analyze_sentiment(review, original_language):
    """
    Split a review into positive and negative parts based on sentiment analysis.

    :param review: str, Input review to analyze
    :param original_language: str, Language code of the original review
    :return: dict, containing:
        - scores: dict, Sentiment scores of the entire review
        - sentiment: str, Overall sentiment of the review
        - positive_parts: list of dict, Positive sentences with original and score
        - negative_parts: list of dict, Negative sentences with original and score
    """

    # Translate the full review to English for overall sentiment analysis
    if original_language != 'en':
        translator = Translator()
        try:
            translated_review = translator.translate(review, src=original_language, dest="en").text
        except Exception as e:
            print(f"Error translating review to English: {e}")
            translated_review = review  # Fallback to the original text
    else:
        translated_review = review  # Fallback to the original text

    # Use VADER for sentiment analysis on the entire review
    scores = sia.polarity_scores(translated_review)

    # Determine overall sentiment
    compound_score = (scores['compound'] + 1) / 2 * 10
    if compound_score >= 9:
        sentiment = "Positive"
    elif compound_score <= 6:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    # Split the original review into sentences
    try:
        sentences = split_review_into_sentences(review)
    except Exception as e:
        print(f"Error splitting review into sentences: {e}")
        sentences = [review]

    positive_parts = []
    negative_parts = []

    # Analyze each sentence
    for sentence in sentences:
        try:
            if original_language != 'en':
                translated_sentence = translator.translate(sentence, src=original_language, dest="en").text
            else:
                translated_sentence = sentence

            # Analyze sentence sentiment
            sentence_scores = sia.polarity_scores(translated_sentence)
            compound_score = sentence_scores['compound']

            # Identify the main topic of the sentence
            main_topic = classify_review(translated_sentence)

            # Classify the sentence
            if compound_score >= 0.05:  # Positive sentiment
                positive_parts.append({
                    "original": sentence.lstrip('"'),
                    "score": compound_score,
                    "main_topic": main_topic
                })
            elif compound_score <= -0.05:  # Negative sentiment
                negative_parts.append({
                    "original": sentence.lstrip('"'),
                    "score": compound_score,
                    "main_topic": main_topic
                })

        except Exception as e:
            print(f"Error analyzing sentence: {e}")
            continue

    return {
        "scores": scores,
        "sentiment": sentiment,
        "positive_parts": positive_parts,
        "negative_parts": negative_parts,
    }

def split_review_into_sentences(review, max_words=20):
    # Initial split by common sentence delimiters
    sentences = re.split(r'(?<=[.,;!?])\s*', review)
    
    # Remove empty strings and unnecessary spaces
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    
    # Additional splitting for very long sentences
    refined_sentences = []
    for sentence in sentences:
        words = sentence.split()  # Split the sentence into words
        if len(words) > max_words:
            # Break into chunks of at most max_words words
            for i in range(0, len(words), max_words):
                refined_sentences.append(" ".join(words[i:i + max_words]))
        else:
            refined_sentences.append(sentence)
    
    return refined_sentences
    
def calculate_star_rating(scores):
    """
    Calculate a star rating between 1 and 5 based on the compound sentiment score.
    
    :param scores: dict, Sentiment scores from VADER.
    :return: float, Star rating between 1 and 5.
    """
    # Transform compound score from [-1, 1] to [1, 5]
    compound = scores['compound']
    rating = 1 + ((compound + 1) / 2) * 4  # Map to range 1-5
    return round(rating, 4)

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

# Function to generate most common words for positive and negative parts
def generate_common_words(parts, original_language):
    """
    Generate the most common words from a list of text parts.

    Args:
        parts (list): List of text parts in the original language.
        original_language (str): Language code (e.g., "en", "pt").

    Returns:
        list: A list of dictionaries containing the most common words and their counts.
    """
    # Mapping of language codes to NLTK stop words languages
    language_map = {
        "en": "english", "pt": "portuguese", "es": "spanish", "fr": "french",
        "de": "german", "it": "italian", "nl": "dutch", "sv": "swedish",
    }
    
    # Determine the stop words language
    stop_words_language = language_map.get(original_language, "english")
    
    # Load stop words based on the detected language
    try:
        stop_words = set(stopwords.words(stop_words_language))
    except OSError:
        # Fallback to English if stop words are unavailable for the detected language
        stop_words = set(stopwords.words("english"))
    
    # Initialize CountVectorizer with stop words
    vectorizer = CountVectorizer(stop_words=list(stop_words))
    
    # Fit and transform the input text parts
    word_counts = vectorizer.fit_transform(parts)
    
    # Sum word occurrences and retrieve vocabulary
    word_sum = word_counts.sum(axis=0)
    words_freq = [
        (word, word_sum[0, idx])
        for word, idx in vectorizer.vocabulary_.items()
    ]
    
    # Get the 10 most common words
    most_common_words = sorted(words_freq, key=lambda x: x[1], reverse=True)[:10]
    
    # Format the results as a list of dictionaries
    formatted_words = [{"word": word, "count": count} for word, count in most_common_words]
    
    return formatted_words

# Topics and Keywords
topics_keywords = {
    'service': [
        'customer service', 'reception', 'support', 'staff', 'friendliness', 
        'courtesy', 'responsiveness', 'professionalism', 'helpful', 
        'knowledgeable', 'feedback', 'communication', 'teamwork', 
        'availability', 'politeness', 'quick response', 'proactive', 
        'rude', 'unhelpful', 'unresponsive', 'unprofessional', 'delayed support', 
        'disrespectful', 'slow response', 'lack of communication', 
        'incompetent', 'unavailable', 'ignored', 'bad attitude', 'careless', 
        'poor feedback', 'miscommunication', 'staff shortage'
    ],
    'price': [
        'price', 'cost', 'value', 'affordability', 'cheap', 
        'discount', 'budget', 'reasonable', 'transparent', 'deal', 'low-cost', 
        'economic', 'good value', 'fair price', 
        'expensive', 'overpriced', 'hidden fees', 'unfair price', 
        'high cost', 'not worth it', 'pricey', 'rip-off', 'poor value', 
        'steep', 'inflated', 'unaffordable'
    ],
    'quality': [
        'quality', 'durability', 'performance', 'material', 'design', 
        'reliability', 'features', 'value', 'precision', 'innovation', 
        'superior', 'dependability', 'usability', 'refinement', 
        'top-notch', 'polished', 'well-made', 
        'poor quality', 'fragile', 'unreliable', 'flawed', 'cheap material', 
        'broken', 'ineffective', 'bad design', 'low performance', 'inconsistent', 
        'malfunction', 'disappointing', 'defective', 'subpar', 'short-lived'
    ],
    'delivery': [
        'delivery', 'shipping', 'logistics', 'tracking', 'punctuality', 
        'courier', 'fulfillment', 'response', 'secure', 'fast', 'smooth', 
        'international', 'real-time', 'on-time', 'hassle-free', 'return', 
        'late', 'delayed', 'missing', 'damaged package', 'lost shipment', 
        'untracked', 'slow', 'high delivery fee', 'poor handling', 
        'incomplete order', 'unreliable courier', 'frustrating return', 
        'wrong item', 'no updates'
    ],
    'experience': [
        'experience', 'satisfaction', 'enjoyment', 'comfort', 'atmosphere', 
        'journey', 'engagement', 'trust', 'convenience', 'personalization', 
        'connection', 'feelings', 'impression', 'loyalty', 'delight', 
        'stress-free', 'seamless', 'feedback', 
        'unsatisfactory', 'boring', 'uncomfortable', 'bad atmosphere', 
        'disengaged', 'distrusting', 'inconvenient', 'impersonal', 
        'disconnected', 'negative impression', 'stressful', 'hectic', 
        'poor experience', 'unmemorable', 'disappointing'
    ]
}

# Function to classify a review into topics
def classify_review(review):
    """
    Classifies a review based on the most relevant topics.

    :param review: str, text of the review
    :return: str or None, name of the category of the first match, or None if no keywords are found
    """
    # Convert the review to lowercase
    review_lower = review.lower()
    
    # Check each topic for keywords
    for topic, keywords in topics_keywords.items():
        for keyword in keywords:
            # Use regex to find the keyword in the review
            if re.search(r'\b' + re.escape(keyword) + r'\b', review_lower):
                return topic  # Return the topic as soon as a match is found
    
    # If no keywords are found, return None
    return None
