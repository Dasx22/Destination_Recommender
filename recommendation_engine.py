import pandas as pd
import numpy as np
from data_handling import DataHandler
from utils.similarity_calculator import SimilarityCalculator

class RecommendationEngine:
    def __init__(self):
        self.data_handler = DataHandler()
        self.similarity_calculator = SimilarityCalculator()
        self.feature_matrix = None
        
    def initialize(self):
        """
            Initialize the recommendation engine with data
        """
        self.data_handler.load_data()
        self.feature_matrix = self.data_handler.create_feature_matrix()     # modified input dataframe
        
    def create_user_profile(self, user_id=1):
        """
            Create user profile based on travel history
            
            Returns profile_features
        """
        visited_destinations, ratings = self.data_handler.get_user_profile(user_id)
        
        if visited_destinations.empty:
            return None
        
        # Create weighted average profile based on ratings
        profile_features = {}
        
        # Calculate weighted averages for numerical features
        total_weight = ratings.sum()        # for normalization
        
        for _, dest in visited_destinations.iterrows():         # index, row
            weight = ratings.get(dest['destination_id'], 1.0)
            
            # Activities preference
            activities = dest['activities_list']
            for activity in activities:
                if activity not in profile_features:
                    profile_features[activity] = 0
                profile_features[activity] += weight / total_weight     # for each activity
            
            # Climate preference
            climate = dest['climate']
            climate_key = f'climate_{climate}'
            if climate_key not in profile_features:
                profile_features[climate_key] = 0
            profile_features[climate_key] += weight / total_weight
            
            # Type preference
            dest_type = dest['type']
            type_key = f'type_{dest_type}'
            if type_key not in profile_features:
                profile_features[type_key] = 0
            profile_features[type_key] += weight / total_weight     # current rating / total rating
            
            # Budget and popularity preferences
            if 'avg_budget' not in profile_features:
                profile_features['avg_budget'] = 0
            if 'avg_popularity' not in profile_features:
                profile_features['avg_popularity'] = 0
                
            profile_features['avg_budget'] += dest['budget_level'] * weight / total_weight
            profile_features['avg_popularity'] += dest['popularity_score'] * weight / total_weight
        
        return profile_features
    
    def get_recommendations(self, user_id=1, num_recommendations=5):
        """
            Get destination recommendations for a user
        """
        if self.feature_matrix is None:
            self.initialize()
        
        # Get user profile
        user_profile = self.create_user_profile(user_id)
        if user_profile is None:
            return self._get_popular_destinations(num_recommendations)
        
        # Get unvisited destinations
        unvisited_destinations = self.data_handler.get_unvisited_destinations(user_id)
        
        if unvisited_destinations.empty:
            return pd.DataFrame()
        
        # Calculate similarity scores for unvisited destinations
        recommendations = []
        
        for _, dest in unvisited_destinations.iterrows():
            similarity_score = self._calculate_destination_similarity(user_profile, dest)
            
            recommendations.append({
                'destination_id': dest['destination_id'],
                'name': dest['name'],
                'country': dest['country'],
                'type': dest['type'],
                'activities': dest['activities'],
                'climate': dest['climate'],
                'budget_level': dest['budget_level'],
                'popularity_score': dest['popularity_score'],
                'similarity_score': similarity_score
            })
        
        # Sort by similarity score and get top recommendations
        recommendations_df = pd.DataFrame(recommendations)
        recommendations_df = recommendations_df.sort_values('similarity_score', ascending=False)
        
        return recommendations_df.head(num_recommendations)
    
    def _calculate_destination_similarity(self, user_profile, destination):     # for internal use only bcz it starts with '_'
        """
            Calculate similarity between user profile and destination
        """
        similarity_score = 0.0
        total_weight = 0.0
        
        weights = {
                'activities': 0.4,
                'climate': 0.2,
                'type': 0.2,
                'budget': 0.1,
                'popularity': 0.1
            }
        
        # Activities similarity
        dest_activities = destination['activities_list']
        activity_sim = 0.0
        for activity in dest_activities:
            if activity in user_profile:
                activity_sim += user_profile[activity]
        activity_sim = activity_sim / len(dest_activities) if dest_activities else 0    # so that destinations with more activities donot get unfairly high scores (so, we normalize)
        similarity_score += activity_sim * weights['activities']
        total_weight += weights['activities']
        
        # Climate similarity
        climate_key = f"climate_{destination['climate']}"
        if climate_key in user_profile:
            similarity_score += user_profile[climate_key] * weights['climate']
        total_weight += weights['climate']
        
        # Type similarity
        type_key = f"type_{destination['type']}"
        if type_key in user_profile:
            similarity_score += user_profile[type_key] * weights['type']
        total_weight += weights['type']
        
        # Budget similarity
        if 'avg_budget' in user_profile:
            budget_diff = abs(user_profile['avg_budget'] - destination['budget_level'])
            budget_sim = max(0, 1.0 - budget_diff / 4.0)        # to avoid negative values
            similarity_score += budget_sim * weights['budget']
        total_weight += weights['budget']
        
        # Popularity boost 
        popularity_factor = destination['popularity_score'] / 10.0
        similarity_score += popularity_factor * weights['popularity']
        total_weight += weights['popularity']
        
        return similarity_score / total_weight if total_weight > 0 else 0
    
    def _get_popular_destinations(self, num_recommendations=5):     # for internal use only (protected)
        """
            Fallback: return popular destinations for new users
            
            Acts as a FAILSAFE if no user history is present
        """
        if self.data_handler.destinations_df is None:
            self.data_handler.load_data()
            
        popular = self.data_handler.destinations_df.nlargest(num_recommendations, 'popularity_score')
        popular['similarity_score'] = popular['popularity_score'] / 10.0
        
        return popular[['destination_id', 'name', 'country', 'type', 'activities', 
                       'climate', 'budget_level', 'popularity_score', 'similarity_score']]
    
    def get_similar_destinations(self, destination_id, num_similar=3):
        """
            Get destinations similar to a given destination
        """
        if self.feature_matrix is None:
            self.initialize()
        
        # Get the target destination
        target_dest = self.data_handler.destinations_df[
            self.data_handler.destinations_df['destination_id'] == destination_id
        ]
        
        if target_dest.empty:
            return pd.DataFrame()
        
        target_dest = target_dest.iloc[0]
        
        # Calculate similarities with all other destinations
        similarities = []
        
        for _, dest in self.data_handler.destinations_df.iterrows():
            if dest['destination_id'] != destination_id:
                similarity = self.similarity_calculator.weighted_feature_similarity(
                    target_dest.to_dict(), dest.to_dict()       # dict makes row data easier to work with
                )
                similarities.append({
                    'destination_id': dest['destination_id'],
                    'name': dest['name'],
                    'country': dest['country'],
                    'similarity_score': similarity
                })
        
        similar_df = pd.DataFrame(similarities)
        return similar_df.sort_values('similarity_score', ascending=False).head(num_similar)