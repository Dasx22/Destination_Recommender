import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

class DataHandler:
    def __init__(self):
        self.destinations_df = None
        self.user_history_df = None
        self.mlb_activities = MultiLabelBinarizer()
        self.mlb_climate = MultiLabelBinarizer()
        self.scaler = StandardScaler()
        
    def load_data(self):
        self.destinations_df = pd.read_excel('data/India_Nearby_Travel_Destinations.xlsx')
        self.user_history_df = pd.read_excel('data/user_history.xlsx')
        # self.user_history_df = pd.read_excel('data/empty_user_history.xlsx')
        
        # Process activities and climate into lists
        self.destinations_df['activities_list'] = self.destinations_df['activities'].str.split(',')
        self.destinations_df['climate_list'] = self.destinations_df['climate'].apply(lambda x: [x])
        
        return self.destinations_df, self.user_history_df
    
    def create_feature_matrix(self):
        """
            Create feature matrix for content-based filtering
        """
        if self.destinations_df is None:
            self.load_data()
        
        # One-hot encode activities
        activities_encoded = self.mlb_activities.fit_transform(self.destinations_df['activities_list'])
        activities_df = pd.DataFrame(activities_encoded, columns=self.mlb_activities.classes_)
        
        # One-hot encode climate
        climate_encoded = self.mlb_climate.fit_transform(self.destinations_df['climate_list'])
        climate_df = pd.DataFrame(climate_encoded, columns=[f'climate_{col}' for col in self.mlb_climate.classes_])
        
        # One-hot encode destination type
        type_encoded = pd.get_dummies(self.destinations_df['type'], prefix='type')
        
        # Normalize numerical features
        numerical_features = ['budget_level', 'popularity_score']
        numerical_df = pd.DataFrame(
            self.scaler.fit_transform(self.destinations_df[numerical_features]),
            columns=numerical_features
        )
        
        # Combine all features
        feature_matrix = pd.concat([
            self.destinations_df[['destination_id', 'name', 'country']],
            activities_df,
            climate_df,
            type_encoded,
            numerical_df
        ], axis=1)
        
        return feature_matrix
    
    def get_user_profile(self, user_id=1):
        """
            Create user profile based on travel history
            
            Returns visited_destinations and ratings
        """
        if self.user_history_df is None:
            self.load_data()
            
        user_visits = self.user_history_df[self.user_history_df['user_id'] == user_id]
        visited_destinations = self.destinations_df[
            self.destinations_df['destination_id'].isin(user_visits['destination_id'])
        ]       # to filter out reqd rows from 'destination_df'
        
        # Weight by ratings
        ratings = user_visits.set_index('destination_id')['rating']  # index -> destination_id
                                                                     # value -> user's rating for that destination
        
        return visited_destinations, ratings
    
    def get_unvisited_destinations(self, user_id=1):
        """
            Returns destinations user hasn't visited
        """
        if self.user_history_df is None:
            self.load_data()
            
        visited_ids = self.user_history_df[
            self.user_history_df['user_id'] == user_id
        ]['destination_id'].tolist()
        
        unvisited = self.destinations_df[
            ~self.destinations_df['destination_id'].isin(visited_ids)
        ]           # faster lookup
        
        return unvisited