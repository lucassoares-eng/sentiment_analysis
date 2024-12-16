import os
from tqdm import tqdm
from .analyze import analyze_sentiment, calculate_star_rating, detect_original_language, generate_common_words
from .utils import load_file

DEFAULT_DATASET = os.path.join("data", "tripadvisor_hotel_reviews.csv")

# Function to process a single review
def process_review(review, original_language):
    try:
        analyze_result = analyze_sentiment(review, original_language)
        sentiment_score = analyze_result['scores']
        sentiment = analyze_result['sentiment']
        star_rating = calculate_star_rating(sentiment_score)
        positive_parts = analyze_result['positive_parts']
        negative_parts = analyze_result['negative_parts']
        return sentiment, sentiment_score, star_rating, positive_parts, negative_parts
    except Exception as e:
        print(f"Error analyzing review: {e}")
        return "Error", {}, 3, [], []  # Default values in case of error

def analyze(dataset_path=DEFAULT_DATASET):
    """
    Analyzes the sentiment of reviews, calculates the NPS score, and returns
    detailed results, including the most common words, sentiment distribution,
    number of reviews, detractors, promoters, and the most relevant comments.
    """
    data = load_file(dataset_path)
    reviews = data["Review"]

    # Detect the predominant language
    original_language = detect_original_language(reviews)

    # Initialize lists to store results
    sentiments = []
    sentiment_scores = []
    star_ratings = []
    positive_parts = []
    negative_parts = []

    print('Examining Reviews...')
    for review in tqdm(reviews):
        sentiment, sentiment_score, star_rating, pos_parts, neg_parts = process_review(review, original_language)
        sentiments.append(sentiment)
        sentiment_scores.append(sentiment_score)
        star_ratings.append(star_rating)
        positive_parts.extend(pos_parts)
        negative_parts.extend(neg_parts)

    # Updating the DataFrame with results
    data["sentiment"] = sentiments
    data["sentiment_scores"] = sentiment_scores
    data["star_rating"] = star_ratings

    # Calculate NPS (Net Promoter Score)
    positive_count = (data["sentiment"] == "Positive").sum()
    negative_count = (data["sentiment"] == "Negative").sum()
    total_reviews = len(data)
    nps_score = ((positive_count - negative_count) / total_reviews) * 100

    # Detractors, Passives, and Promoters
    detractors = data[data["sentiment"] == "Negative"]
    passives = data[data["sentiment"] == "Neutral"]
    promoters = data[data["sentiment"] == "Positive"]

    # Extract only the text from tuples (first element)
    positive_sentences = [entry['original'] for entry in positive_parts]
    negative_sentences = [entry['original'] for entry in negative_parts]

    positive_common_words = generate_common_words(positive_sentences, original_language)
    negative_common_words = generate_common_words(negative_sentences, original_language)

    # Sort positive and negative comments by their scores
    sorted_positive_parts = sorted(positive_parts, key=lambda x: x['score'], reverse=True)
    sorted_negative_parts = sorted(negative_parts, key=lambda x: x['score'])

    # Extract the 3 most relevant comments (positive and negative)
    most_relevant_positive = [entry['original'] for entry in sorted_positive_parts[:3]]
    most_relevant_negative = [entry['original'] for entry in sorted_negative_parts[:3]]

    # Calculate the overall star rating (average)
    overall_star_rating = round(sum(star_ratings) / len(star_ratings), 2)

    # Combine the positive_parts and negative_parts lists into one list
    all_parts = positive_parts + negative_parts

    # Initialize dictionaries to store the scores and counts by topic
    topic_scores = {}
    topic_counts = {}

    # Process all parts
    for part in all_parts:
        if part["main_topic"]:
            topic = part["main_topic"]
            if topic not in topic_scores:
                topic_scores[topic] = 0
                topic_counts[topic] = 0
            topic_scores[topic] += part["score"]
            topic_counts[topic] += 1

    # Calculate the average score for each topic and convert it to a scale of 1 to 10
    ratings = {}
    for topic in topic_scores:
        total_score = topic_scores[topic]
        count = topic_counts[topic]
        if count == 0:
            ratings[topic] = 0
        else:
            average_score = total_score / count
            if average_score <= -0.3:
                ratings[topic] = 0
            else:
                rating = ((average_score + 0.3) / 1.3) * 5
                ratings[topic] = round(rating, 2)

    # Prepare results to be returned
    results = {
        "nps_score": round(nps_score, 2),
        "total_reviews": total_reviews,
        "promoters_count": len(promoters),
        "detractors_count": len(detractors),
        "neutral_count": len(passives),
        "positive_common_words": positive_common_words,
        "negative_common_words": negative_common_words,
        "most_relevant_comments": {
            "positive": most_relevant_positive,
            "negative": most_relevant_negative
        },
        "star_ratings": overall_star_rating,
        "topics_rating": ratings
    }

    return results

def analyze_review(review):
    original_language = detect_original_language([review])
    analyze_result = analyze_sentiment(review, original_language)
    sentiment_score = analyze_result['scores']
    sentiment = analyze_result['sentiment']
    star_rating = calculate_star_rating(sentiment_score) 
    result = {
        "sentment": sentiment,
        "star_ratings": star_rating,
    }
    return result