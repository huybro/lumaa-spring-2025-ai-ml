import numpy as np
class TFIDF:
    def __init__(self):
        self.vocabulary = {}  # Maps terms to indices
        self.idf = {}        # Stores IDF values for each term
        self.doc_count = 0   # Total number of documents
        
    def _compute_tf(self, document):
        """Compute term frequencies for a document"""
        words = document.lower().split()
        word_count = {}
        for word in words:
            word_count[word] = word_count.get(word, 0) + 1
        return word_count
    
    def _compute_idf(self, documents):
        """Compute inverse document frequency for all terms"""
        # Count documents containing each term
        doc_freq = {}
        for doc in documents:
            words = set(doc.lower().split()) 
            for word in words:
                doc_freq[word] = doc_freq.get(word, 0) + 1
        
        # Compute IDF for each term
        self.doc_count = len(documents)
        for word, freq in doc_freq.items():
            self.idf[word] = np.log(self.doc_count / (freq + 1)) + 1
        
        self.vocabulary = {word: idx for idx, word in enumerate(self.idf.keys())}
    
    def fit(self, documents):
        """Fit the TF-IDF model on a list of documents"""
        self._compute_idf(documents)
        return self
    
    def transform(self, documents):
        """Transform documents to TF-IDF vectors"""
        if not isinstance(documents, list):
            documents = [documents]
        
        matrix = np.zeros((len(documents), len(self.vocabulary)))
        
        for doc_idx, doc in enumerate(documents):
            tf = self._compute_tf(doc)
            for word, freq in tf.items():
                if word in self.vocabulary:
                    word_idx = self.vocabulary[word]
                    tf_idf = freq * self.idf.get(word, 0)
                    matrix[doc_idx][word_idx] = tf_idf
        
        return matrix