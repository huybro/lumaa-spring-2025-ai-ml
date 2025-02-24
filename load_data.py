import pandas as pd
import ast

def simple_preprocess_movie_data(meta_path, keywords_path):
    """
    Preprocess movie data focusing only on genres and keywords.
    """
    meta_df = pd.read_csv(meta_path)
    keywords_df = pd.read_csv(keywords_path)
    
    # Convert IDs to integers
    meta_df['id'] = pd.to_numeric(meta_df['id'], errors='coerce').astype('Int64')
    keywords_df['id'] = pd.to_numeric(keywords_df['id'], errors='coerce').astype('Int64')
    
    # Merge dataframes
    df = pd.merge(meta_df[['id', 'title', 'genres']], keywords_df, on='id', how='inner')
    
    def extract_names(x, is_keyword=False):
        """
        Extract names from string of dictionaries and clean if keywords
        
        Parameters:
        x (str): String representation of list of dictionaries
        is_keyword (bool): Whether this is processing keywords (to remove 'movies')
        """
        try:
            data = ast.literal_eval(x)
            names = [item['name'] for item in data if 'name' in item]
            
            if is_keyword:
                # Remove the word 'movies' from each keyword
                cleaned_names = []
                for name in names:
                    name = name.lower()
                    name = name.replace('movies', '').strip()
                    name = name.replace('movie', '').strip()
                    # Remove any double spaces created
                    name = ' '.join(name.split())
                    if name:
                        cleaned_names.append(name)
                return ' '.join(cleaned_names)
            else:
                return ' '.join(name.lower() for name in names)
        except:
            return ''
    
    df['genres'] = df['genres'].fillna('[]').apply(lambda x: extract_names(x, is_keyword=False))
    df['keywords'] = df['keywords'].fillna('[]').apply(lambda x: extract_names(x, is_keyword=True))

    df['combined_features'] = df['genres'] + ' ' + df['keywords']
    
    df = df[df['combined_features'].str.strip() != '']
    
    return df[['id', 'title', 'genres', 'keywords', 'combined_features']]

if __name__ == "__main__":
    # Process the data
    processed_df = simple_preprocess_movie_data('./datasets/movies_metadata.csv', './datasets/keywords.csv')
    
    print("\nSample of processed data:")
    print(processed_df.head())
    print(f"\nTotal movies processed: {len(processed_df)}")
    
    # Save processed data
    processed_df.to_csv('processed_movies_simple.csv', index=False)
    print("\nProcessed data saved to 'processed_movies_simple.csv'")
