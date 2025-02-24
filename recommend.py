import pandas as pd
import numpy as np
from tfidf import TFIDF
from cosine_similarity import cosine_similarity
import sys

def recommend_movies(query, df, tfidf, top_n=5):
    """
    Recommend movies based on query.
    """

    documents = df["combined_features"].tolist()
    tfidf_matrix = tfidf.transform(documents)
    
    # Transform query
    query_vector = tfidf.transform([query])[0]
    
    # Compute similarities
    similarities = [cosine_similarity(query_vector, doc_vector) 
                   for doc_vector in tfidf_matrix]
    
    # Get top N recommendations
    top_indices = np.argsort(similarities)[::-1][:top_n]
    
    # Return recommendations with titles
    recommendations = []
    for idx in top_indices:
        recommendations.append((
            df.iloc[idx]["title"],
            similarities[idx],
        ))
    
    return recommendations

# Load and prepare data
df = pd.read_csv("processed_movies_simple.csv")
documents = df["combined_features"].tolist()

tfidf = TFIDF() 
tfidf.fit(documents)

# Example usage
if len(sys.argv) > 1:
    query = sys.argv[1]
    recommendations = recommend_movies(query, df, tfidf, top_n=10)
    print("\nTop Movie Recommendations:")
    print("-" * 50)
    for i, (title, score) in enumerate(recommendations, 1):
        print(f"\n{i}. {title}")
        print(f"Similarity Score: {score:.4f}")

else:
    print("Invalid query")



