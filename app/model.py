import os
from multiprocessing import Pool, Manager
from collections import defaultdict
from app.analyze import analyze_sentiment, calculate_star_rating, detect_original_language, generate_common_words
from app.utils import load_file

DEFAULT_DATASET = os.path.join("data", "tripadvisor_hotel_reviews.csv")

# Function to process a single review
def process_review(review, original_language, sentiment_list, score_list, star_list, positive_list, negative_list):
    try:
        analyze_result = analyze_sentiment(review, original_language)
        sentiment_score = analyze_result['scores']
        sentiment = analyze_result['sentiment']
        star_rating = calculate_star_rating(sentiment_score)  # Calculates the star rating
        positive_parts = analyze_result['positive_parts']
        negative_parts = analyze_result['negative_parts']

        # Storing results in shared lists
        sentiment_list.append(sentiment)
        score_list.append(sentiment_score)
        star_list.append(star_rating)
        positive_list.extend(positive_parts)
        negative_list.extend(negative_parts)
    except Exception as e:
        print(f"Error analyzing review: {e}")
        sentiment_list.append("Error")
        score_list.append({})
        star_list.append(3)  # Default star rating in case of error
        positive_list.extend([])
        negative_list.extend([])

def analyze(dataset_path=DEFAULT_DATASET):
    """
    Analyzes the sentiment of hotel reviews, calculates the NPS score, and returns
    detailed results, including the most common words, sentiment distribution,
    number of reviews, detractors, promoters, and the most relevant comments.
    
    Args:
        dataset_path (str): Path to the dataset file (.csv).
    
    Returns:
        dict: Results of the analysis.
    """
    data = load_file(dataset_path)
    reviews = data["Review"]

    # Creating a Manager to share lists between processes
    with Manager() as manager:
        sentiment_list = manager.list()
        score_list = manager.list()
        star_list = manager.list()
        positive_list = manager.list()
        negative_list = manager.list()

        # Detect the predominant language
        original_language = detect_original_language(reviews)

        # Using Pool for parallel processing
        print('Examining Reviews...')
        with Pool() as pool:
            # Passing the shared lists to the pool
            pool.starmap(process_review, [(review, original_language, sentiment_list, score_list, star_list, positive_list, negative_list) for review in reviews])

        # Converting the shared lists back to normal lists after processing
        sentiments = list(sentiment_list)
        sentiment_scores = list(score_list)
        star_ratings = list(star_list)
        positive_parts = list(positive_list)
        negative_parts = list(negative_list)

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
    sorted_positive_parts = sorted(positive_parts, key=lambda x: x['score'], reverse=True)  # Descending by score
    sorted_negative_parts = sorted(negative_parts, key=lambda x: x['score'])  # Ascending by score

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
            
            # Initialize in the dictionary if the topic does not exist yet
            if topic not in topic_scores:
                topic_scores[topic] = 0
                topic_counts[topic] = 0
            
            # Add the score and increment the count for the topic
            topic_scores[topic] += part["score"]
            topic_counts[topic] += 1
    
    # Calculate the average score for each topic and convert it to a scale of 1 to 10
    ratings = {}

    for topic in topic_scores:
        total_score = topic_scores[topic]
        count = topic_counts[topic]
        
        if count == 0:
            ratings[topic] = 1  # If there are no parts for the topic, assign a score of 1
        else:
            average_score = total_score / count
            # Convert the average score to a scale of 1 to 10
            rating = ((average_score + 1) / 2) * 9 + 1  # Scale from [-1, 1] to [1, 10]
            ratings[topic] = round(rating, 2)  # Round to 2 decimal places

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