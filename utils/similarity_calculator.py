import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SimilarityCalculator:    
    def cosine_similarity_matrix(self, feature_matrix):
        """
            Calculate cosine similarity between destinations
        """
        # Remove non-numeric columns for similarity calculation
        numeric_cols = feature_matrix.select_dtypes(include=[np.number]).columns    # list
        numeric_features = feature_matrix[numeric_cols].fillna(0)
        
        similarity_matrix = cosine_similarity(numeric_features)   # creates nxn similarity matrix between 2 rows (ie, between row i and j)
        return similarity_matrix
    
    def calculate_user_destination_similarity(self, user_profile_vector, destination_vector):
        """
            Calculate similarity between user profile and a destination
        """
        # Ensure both vectors have the same length
        min_len = min(len(user_profile_vector), len(destination_vector))
        user_vec = user_profile_vector[:min_len]
        dest_vec = destination_vector[:min_len]
        
        # Calculate cosine similarity
        similarity = cosine_similarity([user_vec], [dest_vec])[0][0]
        return similarity
    
    def weighted_feature_similarity(self, dest1_features, dest2_features, weights=None):
        """
            Calculate weighted similarity between two destinations
        """
        if weights is None:
            weights = {
                'activities': 0.4,
                'climate': 0.2,
                'type': 0.2,
                'budget': 0.1,
                'popularity': 0.1
            }
        
        total_similarity = 0
        total_weight = 0
        
        # Activities similarity -> JACCARD similarity for sets
        if 'activities_list' in dest1_features and 'activities_list' in dest2_features:
            set1 = set(dest1_features['activities_list'])
            set2 = set(dest2_features['activities_list'])
            jaccard_sim = len(set1.intersection(set2)) / len(set1.union(set2)) if len(set1.union(set2)) > 0 else 0
            total_similarity += jaccard_sim * weights['activities']
            total_weight += weights['activities']
        
        # Climate similarity
        if 'climate' in dest1_features and 'climate' in dest2_features:
            climate_sim = 1.0 if dest1_features['climate'] == dest2_features['climate'] else 0.0
            total_similarity += climate_sim * weights['climate']
            total_weight += weights['climate']
        
        # Type similarity
        if 'type' in dest1_features and 'type' in dest2_features:
            type_sim = 1.0 if dest1_features['type'] == dest2_features['type'] else 0.0
            total_similarity += type_sim * weights['type']
            total_weight += weights['type']
        
        # Budget similarity (normalized difference)
        if 'budget_level' in dest1_features and 'budget_level' in dest2_features:
            budget_diff = abs(dest1_features['budget_level'] - dest2_features['budget_level'])
            budget_sim = 1.0 - (budget_diff / 4.0)          # Assuming budget levels 1-5
            total_similarity += budget_sim * weights['budget']
            total_weight += weights['budget']
        
        # Popularity similarity
        if 'popularity_score' in dest1_features and 'popularity_score' in dest2_features:
            pop_diff = abs(dest1_features['popularity_score'] - dest2_features['popularity_score'])
            pop_sim = 1.0 - (pop_diff / 10.0)               # Assuming popularity 0-10
            total_similarity += pop_sim * weights['popularity']
            total_weight += weights['popularity']
        
        return total_similarity / total_weight if total_weight > 0 else 0