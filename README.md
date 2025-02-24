# Movie Recommendation System

This project implements a **content-based movie recommendation system** using TF-IDF and cosine similarity. Given a user's query (e.g., "I like action movies set in space"), the system recommends the top N most similar movies based on their plot summaries, genres, and keywords.

---

## **Features**
- **Content-Based Filtering**: Uses TF-IDF to vectorize movie descriptions and cosine similarity to find similar movies.
- **Customizable**: Easily adjust the number of recommendations (`top_n`).
- **Simple and Lightweight**: Built with Python and minimal dependencies.

---

## **Dataset**
The dataset used is a subset of **The Movies Dataset** from [Kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset). It contains the following columns:
- `id`: Unique identifier for each movie.
- `title`: Title of the movie.
- `genres`: Genres associated with the movie.
- `keywords`: Keywords describing the movie.
- `combined_features`: A combination of `genres` and `keywords` used for recommendations.

### **Dataset Source**
- **Name**: The Movies Dataset
- **Source**: [Kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)
- **Description**: A large dataset containing metadata for over 45,000 movies. For this project, a smaller subset (`processed_movies_simple.csv`) is used for simplicity.

---

## **Setup**

### **Prerequisites**
- Python 3.8 or higher.
- Required Python libraries: `pandas`, `numpy`.

### **Install Dependencies**
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/movie-recommendation-system.git
   cd movie-recommendation-system

2. Create a virtual environment:
    python -m venv venv

3. Activate the virtual environment:
    source venv/bin/activate
4. Install the required libraries:
    pip install -r requirements.txt

## **Running the Code**
Run the script with a user query as a command-line argument:
python3 recommend.py "Japanese avtion movies"

Example Output
For the query "Japanese avtion movies", the output might look like this:
Top Movie Recommendations:
--------------------------------------------------

1. Behind the Rising Sun
Similarity Score: 0.8307

2. The Most Beautiful Night In The World
Similarity Score: 0.8240

3. Crows Explode
Similarity Score: 0.7976

4. The Great Passage
Similarity Score: 0.7935

5. Japan's Longest Day
Similarity Score: 0.7240

6. Snakes and Earrings
Similarity Score: 0.6931

7. Early Spring
Similarity Score: 0.6809

8. Take Aim at the Police Van
Similarity Score: 0.6785

9. I Am Waiting
Similarity Score: 0.6705

10. The Kingdom of Dreams and Madness
Similarity Score: 0.6684
