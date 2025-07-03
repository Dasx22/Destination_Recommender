import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from recommendation_engine import RecommendationEngine
from data_handling import DataHandler

# Initialize session state
if 'recommendation_engine' not in st.session_state:
    st.session_state.recommendation_engine = RecommendationEngine()
    st.session_state.recommendation_engine.initialize()

def main():
    st.set_page_config(
        page_title="Travel Destination Recommender",
        page_icon="🚞",
        layout="wide"
    )
    
    st.title("🚞 Travel Destination Recommendation System")
    st.markdown("---")
    
    st.sidebar.markdown("## 🍃 Help Me Travel")
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", 
                               ["Home", "Get Recommendations", "Explore Destinations", "Travel History"])
    
    if page == "Home":
        show_home_page()
    elif page == "Get Recommendations":
        show_recommendations_page()
    elif page == "Explore Destinations":
        show_explore_page()
    elif page == "Travel History":
        show_history_page()

def show_home_page():
    st.header("Welcome to Your Personal Travel Recommender!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 How It Works")
        st.write("""
        Our recommendation system uses **content-based filtering** to suggest destinations based on:
        - Your travel history and ratings
        - Preferred activities and experiences
        - Climate preferences
        - Budget considerations
        - Destination popularity
        """)
        
        st.subheader("🔮 Features")
        st.write("""
        - **Personalized Recommendations**: Get suggestions tailored to your preferences
        - **Similar Destinations**: Find places similar to ones you've loved
        - **Detailed Information**: Explore activities, climate, and budget info
        - **Visual Analytics**: See your travel patterns and preferences
        """)
    
    with col2:
        st.subheader("📊 System Stats")
        
        # Get some stats from the data
        data_handler = DataHandler()
        destinations_df, user_history_df = data_handler.load_data()
        
        stats_col1, stats_col2 = st.columns(2)
        with stats_col1:
            st.metric("Total Destinations", len(destinations_df))
            st.metric("Countries Covered", destinations_df['country'].nunique())
        
        with stats_col2:
            st.metric("Destination Types", destinations_df['type'].nunique())
            st.metric("Your Visits", len(user_history_df))

def show_recommendations_page():
    st.header("🎯 Get Personalized Recommendations")
    
    # User preferences
    st.subheader("Customize Your Preferences")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        num_recommendations = st.slider("Number of recommendations", 1, 10, 5)
        
    with col2:
        preferred_budget = st.selectbox("Budget Level", 
                                      ["Any", "Low (1-2)", "Medium (3)", "High (4-5)"])
    
    with col3:
        preferred_climate = st.selectbox("Preferred Climate", 
                                       ["Any", "Tropical", "Temperate", "Mediterranean", "Desert", "Cold"])
    
    if st.button("Get Recommendations", type="primary"):
        with st.spinner("Generating personalized recommendations..."):
            recommendations = st.session_state.recommendation_engine.get_recommendations(
                user_id=1, num_recommendations=num_recommendations
            )
            
            if not recommendations.empty:
                st.subheader("🌟 Your Personalized Recommendations")
                
                for idx, rec in recommendations.iterrows():
                    with st.expander(f"🏖️ {rec['name']}, {rec['country']} (Match: {rec['similarity_score']:.2f})", expanded=True):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.write(f"**Type:** {rec['type']}")
                            st.write(f"**Activities:** {rec['activities']}")
                            st.write(f"**Climate:** {rec['climate']}")
                        
                        with col2:
                            st.write(f"**Budget Level:** {rec['budget_level']}/5")
                            st.write(f"**Popularity:** {rec['popularity_score']:.1f}/10")
                            st.write(f"**Match Score:** {rec['similarity_score']:.2f}")
                        
                        with col3:
                            # Add to wishlist button
                            if st.button(f"Add to Wishlist", key=f"wishlist_{rec['destination_id']}"):
                                st.success(f"Added {rec['name']} to your wishlist!")
                            
                            # Find similar destinations
                            if st.button(f"Find Similar", key=f"similar_{rec['destination_id']}"):
                                similar = st.session_state.recommendation_engine.get_similar_destinations(
                                    rec['destination_id'], num_similar=3
                                )
                                if not similar.empty:
                                    st.write("**Similar destinations:**")
                                    for _, sim in similar.iterrows():
                                        st.write(f"• {sim['name']}, {sim['country']} (Score: {sim['similarity_score']:.2f})")
            else:
                st.info("No recommendations available. Try adjusting your preferences!")

def show_explore_page():
    st.header("🗺️ Explore All Destinations")
    
    # Get all destinations
    data_handler = DataHandler()
    destinations_df, _ = data_handler.load_data()
    
    # Filters
    st.subheader("Filter Destinations")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        country_filter = st.multiselect("Countries", 
                                      options=destinations_df['country'].unique(),
                                      default=[])
    
    with col2:
        type_filter = st.multiselect("Destination Types", 
                                   options=destinations_df['type'].unique(),
                                   default=[])
    
    with col3:
        climate_filter = st.multiselect("Climate", 
                                      options=destinations_df['climate'].unique(),
                                      default=[])
    
    # apply filters
    filtered_df = destinations_df.copy()
    
    if country_filter:
        filtered_df = filtered_df[filtered_df['country'].isin(country_filter)]
    if type_filter:
        filtered_df = filtered_df[filtered_df['type'].isin(type_filter)]
    if climate_filter:
        filtered_df = filtered_df[filtered_df['climate'].isin(climate_filter)]
    
    # display results
    st.subheader(f"Found {len(filtered_df)} destinations")
    
    # Create a more visual display
    for idx, dest in filtered_df.iterrows():
        with st.expander(f"🏖️ {dest['name']}, {dest['country']}", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Type:** {dest['type']}")
                st.write(f"**Activities:** {dest['activities']}")
                st.write(f"**Climate:** {dest['climate']}")
            
            with col2:
                st.write(f"**Budget Level:** {dest['budget_level']}/5")
                st.progress(dest['budget_level'] / 5)   # progress bar
                
                st.write(f"**Popularity Score:** {dest['popularity_score']:.1f}/10")
                st.progress(dest['popularity_score'] / 10)
    
    # Visualization
    st.subheader("📊 Destination Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Popularity vs Budget scatter plot
        fig_scatter = px.scatter(filtered_df, 
                               x='budget_level', 
                               y='popularity_score',
                               color='type',
                               hover_data=['name', 'country'],
                               title="Popularity vs Budget Level")
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        # Distribution by type
        type_counts = filtered_df['type'].value_counts()
        fig_pie = px.pie(values=type_counts.values, 
                        names=type_counts.index,
                        title="Distribution by Destination Type")
        st.plotly_chart(fig_pie, use_container_width=True)

def show_history_page():
    st.header("📚 Your Travel History")
    
    # get user history
    data_handler = DataHandler()
    destinations_df, user_history_df = data_handler.load_data()
    
    # merge history with destination details
    history_with_details = user_history_df.merge(destinations_df, on='destination_id')
    
    # Travel analytics
    st.subheader("Your Travel Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # average rating
        avg_rating = history_with_details['rating'].mean()
        st.metric("Average Trip Rating", f"{avg_rating:.1f}/5")
        
        # favorite destination type
        fav_type = history_with_details['type'].mode().iloc[0] if not history_with_details['type'].mode().empty else "None"
        st.metric("Favorite Destination Type", fav_type)
        
        # Travel preferences chart
        type_ratings = history_with_details.groupby('type')['rating'].mean().sort_values(ascending=False)
        fig_bar = px.bar(x=type_ratings.index, y=type_ratings.values,
                        title="Your Average Ratings by Destination Type",
                        labels={'x': 'Destination Type', 'y': 'Average Rating'})
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        # Climate preferences
        climate_counts = history_with_details['climate'].value_counts()
        fig_climate = px.pie(values=climate_counts.values, 
                           names=climate_counts.index,
                           title="Your Climate Preferences")
        st.plotly_chart(fig_climate, use_container_width=True)
        
        # Budget vs Rating scatter
        fig_budget = px.scatter(history_with_details, 
                              x='budget_level', 
                              y='rating',
                              size='popularity_score',      # size of data-point wrt popularity
                              hover_data=['name'],
                              title="Budget vs Your Ratings")
        st.plotly_chart(fig_budget, use_container_width=True)
    
    # User profile insights
    st.subheader("🎯 Your Travel Profile")
    
    profile = st.session_state.recommendation_engine.create_user_profile(user_id=1)
    if profile:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Top Activity Preferences:**")
            activity_prefs = {k: v for k, v in profile.items() if not k.startswith(('climate_', 'type_', 'avg_'))}
            if activity_prefs:
                sorted_activities = sorted(activity_prefs.items(), key=lambda x: x[1], reverse=True)[:5]
                for activity, score in sorted_activities:
                    st.write(f"• {activity}: {score:.2f}")
        
        with col2:
            st.write("**Profile Summary:**")
            if 'avg_budget' in profile:
                st.write(f"• Average Budget Preference: {profile['avg_budget']:.1f}/5")
            if 'avg_popularity' in profile:
                st.write(f"• Popularity Preference: {profile['avg_popularity']:.1f}/10")
            
            # Climate preferences
            climate_prefs = {k.replace('climate_', ''): v for k, v in profile.items() if k.startswith('climate_')}
            if climate_prefs:
                top_climate = max(climate_prefs.items(), key=lambda x: x[1])
                st.write(f"• Preferred Climate: {top_climate[0]} ({top_climate[1]:.2f})")
    
    st.subheader("Your Previous Trips")
    
    for idx, trip in history_with_details.iterrows():
        with st.expander(f"⭐ {trip['name']}, {trip['country']} - Rated {trip['rating']}/5", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"**Visit Date:** {trip['visit_date']}")
                st.write(f"**Type:** {trip['type']}")
                st.write(f"**Activities:** {trip['activities']}")
            
            with col2:
                st.write(f"**Climate:** {trip['climate']}")
                st.write(f"**Budget Level:** {trip['budget_level']}/5")
                st.write(f"**Your Rating:** {trip['rating']}/5")
            
            with col3:
                # Rating visualization
                stars = "⭐" * int(trip['rating']) + "☆" * (5 - int(trip['rating']))
                st.write(f"**Rating:** {stars}")
                
                # Find similar button
                if st.button(f"Find Similar to {trip['name']}", key=f"find_similar_{trip['destination_id']}"):
                    similar_destinations = st.session_state.recommendation_engine.get_similar_destinations(
                        trip['destination_id'], num_similar=3
                    )
                    if not similar_destinations.empty:
                        st.write("**Similar destinations you might like:**")
                        for _, sim_dest in similar_destinations.iterrows():
                            st.write(f"• {sim_dest['name']}, {sim_dest['country']} (Similarity: {sim_dest['similarity_score']:.2f})")
    
if __name__ == "__main__":
    main()