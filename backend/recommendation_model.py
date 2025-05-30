#Recommendation Model

import numpy as np
import pandas as pd
import time
#Clean the Text Data:
import re
from bs4 import BeautifulSoup

# Function to clean DESCRIPTION text
def clean_description(description):
    # Remove HTML tags using BeautifulSoup
    clean_text = BeautifulSoup(description, "html.parser").get_text()

    # Optionally: Remove phone numbers (you can adjust this based on your data)
    clean_text = re.sub(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', '', clean_text)

    # Remove excessive punctuation or any non-alphabetic characters (keep alphanumeric and space)
    clean_text = re.sub(r'[^a-zA-Z0-9\s]', '', clean_text)

    # Convert text to lowercase for uniformity
    clean_text = clean_text.lower()

    return clean_text

# Function to clean HIGHLIGHTS text
def clean_highlights(highlights):
    # Remove HTML tags using BeautifulSoup
    clean_text = BeautifulSoup(highlights, "html.parser").get_text()

    # Convert text to lowercase for uniformity
    clean_text = clean_text.lower()

    return clean_text

def filter_properties_mod(df, user_preferences):
    
    # Remove rows with NaN in PRICE_Modified for sale properties (if type is 'sale')
    if user_preferences.get('type') == 'sale' and 'price' not in user_preferences:
        df = df.dropna(subset=['PRICE_Modified'])

    # Remove rows with NaN in rate_per_sf_year for lease properties (if type is 'lease')
    if user_preferences.get('type') == 'lease' and 'price' not in user_preferences:
        df = df.dropna(subset=['rate_per_sf_year'])

    filtered_df = df.copy()

    # Filter by Property Type (Sale or Lease)
    if 'type' in user_preferences:
        filtered_df = filtered_df[filtered_df['TYPE'] == user_preferences['type']]
    
    # Filter by Cities (multiple selections supported)
    if 'cities' in user_preferences and user_preferences['cities']:
        filtered_df = filtered_df[filtered_df['CITY'].isin(user_preferences['cities'])]

    # Filter by State Abbreviation
    if 'state_abbr' in user_preferences:
        filtered_df = filtered_df[filtered_df['STATE_ABBR'] == user_preferences['state_abbr']]
    
    # Filter by County
    # Filter by County (for multiple counties in a list)
    if 'county' in user_preferences and user_preferences['county']:
    # Check if the COUNTY column value is in the list of selected counties
        filtered_df = filtered_df[filtered_df['COUNTY'].isin(user_preferences['county'])]


    # Filter by Price
    if 'price' in user_preferences:
        min_price, max_price = user_preferences['price']
        if user_preferences.get('type') == 'sale':
            filtered_df = filtered_df[
                (filtered_df['PRICE_Modified'] >= min_price) & 
                (filtered_df['PRICE_Modified'] <= max_price)
            ]
        elif user_preferences.get('type') == 'lease':
            filtered_df = filtered_df[
                (filtered_df['rate_per_sf_year'] >= min_price) & 
                (filtered_df['rate_per_sf_year'] <= max_price)
            ]
    
    # Filter by Building Size
    if 'min_building_sf' in user_preferences and 'max_building_sf' in user_preferences:
        min_sf, max_sf = user_preferences['min_building_sf'], user_preferences['max_building_sf']
        filtered_df = filtered_df[
            (filtered_df['BUILDING_SF_Modified_Numeric_Final'] >= min_sf) & 
            (filtered_df['BUILDING_SF_Modified_Numeric_Final'] <= max_sf)
        ]
    
    # Filter by Property Type (One-hot encoded)
    if 'property_types' in user_preferences and user_preferences['property_types']:
        property_mask = filtered_df[user_preferences['property_types']].sum(axis=1) > 0
        filtered_df = filtered_df[property_mask]
    
    return filtered_df

#Content Based Similarity (TF-IDF)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Content-Based Similarity Calculation (Description + Highlights)
def calculate_content_similarity(df):
    # Combine description and highlights for content-based similarity
    df['combined_features'] = df['CLEANED_DESCRIPTION'] + " " + df['CLEANED_HIGHLIGHTS']
    
    # TF-IDF Vectorizer
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    
    # Cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

#Content Based Similarity (Sentence Bert)
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Sentence-BERT Model
model = SentenceTransformer('all-MiniLM-L6-v2')  # lightweight and accurate

def calculate_sbert_similarity(df):
    # Combine 
    df['combined_features'] = (
        df['CLEANED_DESCRIPTION'] + " " + df['CLEANED_HIGHLIGHTS']
    )

    # Generate sentence embeddings
    embeddings = model.encode(df['combined_features'].tolist(), show_progress_bar=True)

    # Compute cosine similarity
    cosine_sim = cosine_similarity(embeddings, embeddings)
    return cosine_sim

def recommend_properties(user_preferences, sales_df, lease_df, top_n):

    # Start the timer
    start_time = time.time()

    # Determine the dataset to use based on user preferences
    if user_preferences.get('type') == 'sale':
        filtered_df = filter_properties_mod(sales_df, user_preferences)
        price_column = 'PRICE_Modified'  # Sales data uses PRICE_Modified
        roi_column = 'Normalized_ROI'  # ROI column for sales properties
    elif user_preferences.get('type') == 'lease':
        filtered_df = filter_properties_mod(lease_df, user_preferences)
        price_column = 'rate_per_sf_year'  # Lease data uses RATE_PER_SF_YEAR
        roi_column = None  # No ROI for lease properties
    else:
        # Default case: if no property_type is specified, you can handle it by either raising an error or combining both datasets
        raise ValueError("Property type must be either 'sale' or 'lease'.")

    # Step 1: Check if description or highlights are present or have non-informative text
    def prioritize_description(row):
        description = row.get('DESCRIPTION', '').strip().lower()
        highlights = row.get('HIGHLIGHTS', '').strip().lower()
        if 'no description' in description and 'no highlights' in highlights:
            return 0  # Low priority
        return 1  # High priority

    #print(filtered_df.apply(prioritize_description, axis=1).head())
    # Apply the prioritization function to each property
    filtered_df['description_priority'] = filtered_df.apply(prioritize_description, axis=1)

    # Step 2: Handle Transit Preferences (based on user input)
    def transit_preference(row, user_preferences):
        # Check the user's preference for transit distance
        preferred_transit_range = user_preferences.get('transit_range', 1)  # Default to 1 mile if not specified

        # Map preferred transit range to relevant column(s)
        if preferred_transit_range == 1:
            return 1 if row['has_transit_1mi'] else 0
        elif preferred_transit_range == 2:
            return 1 if row['has_transit_2mi'] else 0
        elif preferred_transit_range == 3:
            return 1 if row['has_transit_3to5mi'] else 0
        elif preferred_transit_range == 5:
            return 1 if row['has_transit_5mi'] else 0
        return 0  # Default: no transit preference

    # Apply the transit preference function
    filtered_df['transit_score'] = filtered_df.apply(lambda row: transit_preference(row, user_preferences), axis=1)

    # Step 3: Calculate Cosine Similarity for filtered properties
    cosine_sim = calculate_content_similarity(filtered_df)

    # Step 4: Calculate final score (for sales, consider ROI as well)
    def calculate_final_score(row, cosine_sim, idx, roi_column=None):
    # Get the similarity score for the current property to the base property (property at idx)
        cosine_score = cosine_sim[idx][idx]  # Fix to get a single cosine similarity value (scalar)
        # Rest of the attributes
        transit_score = row['transit_score']
        description_priority = row['description_priority']
    
    # Use ROI for sales properties
        if roi_column:
            roi_score = row[roi_column]
        else:
            roi_score = 0  # No ROI for lease properties
    
    # Different formula for sale vs lease properties
        if user_preferences.get('type') == 'sale':
            # For sale properties, combine cosine similarity, ROI, and transit score
            final_score = (0.3 * cosine_score) + (0.2 * transit_score) + (0.4 * roi_score) + (0.1 * description_priority)
        elif user_preferences.get('type') == 'lease':
            # For lease properties, combine cosine similarity and transit score
            final_score = (0.6 * cosine_score) + (0.4 * transit_score) + (0.2 * description_priority)
        else:
            raise ValueError("Invalid property type. Please use 'sale' or 'lease'.")
        return final_score

    # Ensure final_score calculation returns a valid number for each row
    filtered_df['final_score'] = [calculate_final_score(filtered_df.iloc[i], cosine_sim, i, roi_column) for i in range(len(filtered_df))]

    

    # Check if the final_score column is numeric
    if not pd.api.types.is_numeric_dtype(filtered_df['final_score']):
        print("Error: final_score column contains non-numeric values.")
        print(filtered_df[['ID', 'final_score']].head())
        return []

    # Step 5: Rank based on final scores (top N most similar properties)
    top_similarities = filtered_df.sort_values(by='final_score', ascending=False).head(top_n)

    # End the timer
    end_time = time.time()

    # Calculate execution time
    execution_time = end_time - start_time
    print(f"Recommendation execution time: {execution_time:.4f} seconds")

    columns_to_return = [
    'ID', 'NAME', 'CITY', 'STATE_ABBR', price_column, 'ADDRESS', 
    'COUNTY', 'CLEANED_DESCRIPTION', 'CLEANED_HIGHLIGHTS', 'final_score',
    'LATITUDE', 'LONGITUDE', 'Deal_Score']

    # Add 'Estimated_ROI' only if 'Normalized_ROI' is present
    if 'Normalized_ROI' in top_similarities.columns:
        top_similarities = top_similarities.rename(columns={'Normalized_ROI': 'Estimated_ROI'})
        columns_to_return.insert(-1, 'Estimated_ROI')  # Insert before Deal_Score


   # Return selected columns with the renamed column in the return statement
    return top_similarities[columns_to_return], price_column, execution_time

#Performance Metrics
def calculate_diversity(recommended_df):
    cosine_sim = calculate_content_similarity(recommended_df)
    avg_similarity = cosine_sim.mean()  # Lower is better (more diversity)
    return 1 - avg_similarity  # Convert similarity to diversity score