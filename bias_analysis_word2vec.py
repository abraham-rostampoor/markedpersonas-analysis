# bias_analysis_word2vec.py

import os
import pandas as pd
import re
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import numpy as np
from scipy.spatial.distance import cosine
import argparse

# Function to preprocess text
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
    text = text.lower()  # Convert to lowercase
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return words

# Function to train the Word2Vec model
def train_word2vec(sentences):
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
    return model

# Function to calculate cosine similarity for bias detection
def cosine_similarity(word_vec1, word_vec2):
    return 1 - cosine(word_vec1, word_vec2)

# Word Embedding Association Test (WEAT)
def weat_test(model, target_words1, target_words2, attribute_words1, attribute_words2):
    target_vectors1 = [model.wv[word] for word in target_words1 if word in model.wv.key_to_index]
    target_vectors2 = [model.wv[word] for word in target_words2 if word in model.wv.key_to_index]
    attribute_vectors1 = [model.wv[word] for word in attribute_words1 if word in model.wv.key_to_index]
    attribute_vectors2 = [model.wv[word] for word in attribute_words2 if word in model.wv.key_to_index]

    def mean_cosine_similarity(word_vec, attribute_vectors):
        return np.mean([cosine_similarity(word_vec, attr_vec) for attr_vec in attribute_vectors])

    s1 = np.mean([mean_cosine_similarity(target_vec, attribute_vectors1) - mean_cosine_similarity(target_vec, attribute_vectors2) for target_vec in target_vectors1])
    s2 = np.mean([mean_cosine_similarity(target_vec, attribute_vectors1) - mean_cosine_similarity(target_vec, attribute_vectors2) for target_vec in target_vectors2])
    effect_size = (s1 - s2) / np.std(target_vectors1 + target_vectors2)

    return effect_size

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str, help='Path to the input CSV file containing text data.')
    args = parser.parse_args()

    input_file = args.input_file

    # Load data
    df = pd.read_csv(input_file)
    texts = df['text'].tolist()

    # Preprocess text
    sentences = [preprocess_text(text) for text in texts]

    # Train Word2Vec model
    model = train_word2vec(sentences)

    # Define target words and attribute words for bias analysis
    target_words1 = ['man', 'male', 'boy', 'brother']
    target_words2 = ['woman', 'female', 'girl', 'sister']
    attribute_words1 = ['science', 'technology', 'engineering', 'math']
    attribute_words2 = ['arts', 'humanities', 'literature', 'history']

    # Perform WEAT
    effect_size = weat_test(model, target_words1, target_words2, attribute_words1, attribute_words2)
    print(f'{os.path.basename(input_file)},WEAT effect size: {effect_size}')

if __name__ == "__main__":
    main()

