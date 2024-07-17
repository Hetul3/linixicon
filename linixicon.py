import numpy as np
import json
from difflib import get_close_matches

class WordEmbedding:
    def __init__(self):
        try:
            with open('token2int.json', 'r') as f:
                self.token2int = json.load(f)
            
            with open('int2token.json', 'r') as f:
                self.int2token = json.load(f)
            
            self.weights = np.load('weights.npy')
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def get_embedding(self, word):
        if word in self.token2int:
            return self.weights[self.token2int[word]]
        else:
            raise ValueError('Word not found in dictionary')
        
    def suggest_words(self, word, vocab, topN=5):
        suggestions = get_close_matches(word, vocab, topN)
        return suggestions
    
    def select_random_words(self, n):
        vocab = list(self.token2int.keys())
        random_words = np.random.choice(vocab, n, replace=False)
        return random_words
    
    def cosine_similarity(self, v1, v2):
        dot_product = np.dot(v1, v2)
        norm_vec1 = np.linalg.norm(v1)
        norm_vec2 = np.linalg.norm(v2)
        return dot_product / (norm_vec1 * norm_vec2)
    
    def euclidean_distance(self, v1, v2):
        return np.linalg.norm(v1 - v2)
    
    def manhattan_distance(self, v1, v2):
        return np.sum(np.abs(v1 - v2))
    
    def find_similarity_cosine(self, word1, word2):
        vec1 = self.get_embedding(word1)
        vec2 = self.get_embedding(word2)
        similarity = self.cosine_similarity(vec1, vec2)
        return similarity * 100
    
    def find_similarity_euclidean(self, word1, word2):
        vec1 = self.get_embedding(word1)
        vec2 = self.get_embedding(word2)
        distance = self.euclidean_distance(vec1, vec2)
        similarity = (1 / (1+distance))
        return similarity * 100
    
    def find_similarity_manhattan(self, word1, word2):
        vec1 = self.get_embedding(word1)
        vec2 = self.get_embedding(word2)
        distance = self.manhattan_distance(vec1, vec2)
        similarity = (1 / (1+distance))
        return similarity * 100
    
    def close_words(self, word1, word2, method, percent):
        similiarity = 0
        if method == 'cosine':
            similiarity = self.find_similiarity_cosine(word1, word2)
        elif method == 'euclidean':
            similiarity = self.find_similiarity_euclidean(word1, word2)
        elif method == 'manhatten':
            similiarity = self.find_similiarity_manhatten(word1, word2)
        return similiarity >= percent
    
    def find_all_close_words(self, word, method, percent, words):
        result = []
        for i in range (len(words)):
            if self.close_words(word, words[i], method, percent):
                result.append(words[i])
                
        return result
    
    def in_list(self, word):
        return word in self.token2int
    
    def find_closest_and_furthest(self, word, n, method):
        try:
            vec1 = self.get_embedding(word)
            vocab = list(self.token2int.keys())
            
            distances = []
            for vocab_word in vocab:
                vec2 = self.get_embedding(vocab_word)
                distance = 0
                if method == 'cosine':
                    distance = self.cosine_similiarity(vec1, vec2)
                elif method == 'euclidean':
                    distance = self.euclidean_distance(vec1, vec2)
                elif method == 'manhatten':
                    distance = self.manhatten_distance(vec1, vec2)
                
                distances.append((vocab_word, distance))
            
            distances.sort(key=lambda x: x[1]) 
            closest_words = distances[:n]
            furthest_words = distances[-n:]
            
            return closest_words, furthest_words
        
        except ValueError as e:
            print(e)
            return None, None 