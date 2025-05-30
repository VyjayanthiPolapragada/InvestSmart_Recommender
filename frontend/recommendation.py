import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pydeck as pdk

import streamlit as st
import pandas as pd
from backend.EDA import (
    plot_property_type_pie_by_state,
    plot_price_vs_size_bubble,
    plot_property_distribution_on_map,
    plot_lease_rate_distribution_by_state,
    plot_price_distribution_by_state
)

from backend.recommendation_model import (
    clean_description,
    clean_highlights,
    filter_properties_mod,
    calculate_content_similarity,
    recommend_properties,
    calculate_diversity
)

from backend.deal_Score_model import (
    compute_lease_deal_scores,
    compute_sale_deal_scores
)

def run_recommender(sales_data, lease_data, sales_transit, lease_transit):

#Merge data
    sales_merged_df = pd.merge(sales_data, sales_transit[['ID', 'has_transit_1mi', 'has_transit_2mi', 'has_transit_5mi']], 
                     on='ID', 
                     how='left')

    lease_merged_df = pd.merge(lease_data, lease_transit[['ID', 'has_transit_1mi', 'has_transit_2mi', 'has_transit_5mi']], 
                     on='ID', 
                     how='left')

#Compute deal scores
    sales_merged_df=compute_sale_deal_scores(sales_merged_df)
    lease_merged_df=compute_lease_deal_scores(lease_merged_df)


# Streamlit configuration
    #st.set_page_config(page_title="InvestSmart - Personalized CRE", layout="wide")

    st.title("🏙️ InvestSmart Recommender")

    st.markdown("""
    Welcome to **InvestSmart**, your personalized Commercial Real Estate (CRE) recommendation engine.

    This tool leverages machine learning and geospatial analytics to:
    - Recommend high-ROI properties tailored to your preferences.
    - Analyze proximity to public transit.
    - Evaluate listings based on price fairness using deal scores.

    📊 Explore trends in property types, pricing, lease rates, and discover smart investment opportunities across the U.S.
    """)

# Sidebar for user input
    st.sidebar.title("🔍 Property Recommendation Filters")

# Sidebar for user selection: sale or lease
    property = st.sidebar.radio(
    "🏠 Choose Property Type",
    ("Sale Properties", "Lease Properties")
    )

    if property=='Sale Properties':
        property_type='sale'
    else:
        property_type='lease'

# Get unique values for state, city, and county from the dataset
    if property == "Sale Properties":
        # Sale properties options
        state_options = sales_merged_df['STATE_ABBR'].unique()  
        county_options = []
        city_options = []

    elif property == "Lease Properties":
        # Lease properties options
        state_options = lease_merged_df['STATE_ABBR'].unique()  
        county_options = []
        city_options = []

# State Dropdown (using unique values from the dataset)
    selected_state = st.sidebar.selectbox("State", state_options, index=0)

# Dynamically update county and city options based on the selected state
    if selected_state:
        if property == "Sale Properties":
            county_options = sales_merged_df[sales_merged_df['STATE_ABBR'] == selected_state]['COUNTY'].unique()
            city_options = sales_merged_df[sales_merged_df['STATE_ABBR'] == selected_state]['CITY'].unique()
        elif property == "Lease Properties":
            county_options = lease_merged_df[lease_merged_df['STATE_ABBR'] == selected_state]['COUNTY'].unique()
            city_options = lease_merged_df[lease_merged_df['STATE_ABBR'] == selected_state]['CITY'].unique()

# County Dropdown (default to no selection)
    selected_county = st.sidebar.multiselect("County", county_options, default=[])

# Cities Dropdown (default to no selection)
    selected_cities = st.sidebar.multiselect("Cities", city_options, default=[])

# Price/Lease Rate Range Slider
    if property == 'Sale Properties':
        price_range = st.sidebar.slider("Price Range (USD)", min_value=10000, max_value=5000000, value=(10000, 1000000))
    else:
        lease_range = st.sidebar.slider("Lease Rate Range (USD per year)", min_value=1, max_value=100, value=(10, 50))

# Building Size Range
    building_size_range = st.sidebar.slider("Building Size (sq ft)", min_value=10, max_value=150000, value=(50, 5000))

# Transit Range (1, 2, 3, 5 miles)
    transit_range = st.sidebar.selectbox("Transit Range (miles)", [1, 2, 3, 5], index=0)


# Property Type Checkboxes
    property_types = ['Office', 'Retail', 'Industrial', 'Healthcare', 'Other',
        'Land', 'Investment', 'Multifamily', 'Hospitality']  
    selected_property_types = st.sidebar.multiselect("Property Types", property_types, default=['Office', 'Retail'])

# Apply filters to recommendation model when user clicks 'Get Recommendations'
    if st.sidebar.button('Get Recommendations'):
    # Set user preferences based on inputs
        user_preferences = {
        'type': property_type,
        'state_abbr': selected_state,
        'county': selected_county,
        'cities': selected_cities,
        'price': price_range if property == 'Sale Properties' else lease_range,
        'min_building_sf': building_size_range[0],
        'max_building_sf': building_size_range[1],
        'property_types': selected_property_types,
        'transit_range': transit_range
        }

    # Clean the description and highlights before giving input to the recommendation model
        sales_merged_df['CLEANED_DESCRIPTION'] = sales_merged_df['DESCRIPTION'].apply(clean_description)
        sales_merged_df['CLEANED_HIGHLIGHTS'] = sales_merged_df['HIGHLIGHTS'].apply(clean_highlights)
        lease_merged_df['CLEANED_DESCRIPTION'] = lease_merged_df['DESCRIPTION'].apply(clean_description)
        lease_merged_df['CLEANED_HIGHLIGHTS'] = lease_merged_df['HIGHLIGHTS'].apply(clean_highlights)

    # Get recommendations
        recommended_df,price_column,execution_time = recommend_properties(user_preferences, sales_merged_df, lease_merged_df, top_n=5)

    # Compute diversity score
        diversity_score = calculate_diversity(recommended_df)

    # Display results in UI
        st.subheader("🎯 Top Property Recommendations")
    # Columns to display (conditionally include Estimated_ROI)
        display_columns = ['NAME', 'CITY', 'STATE_ABBR', 'COUNTY', 'ADDRESS', price_column, 'Deal_Score']
        if 'Estimated_ROI' in recommended_df.columns:
            display_columns.insert(-1, 'Estimated_ROI')  # Add before Deal_Score
        
        st.dataframe(recommended_df[display_columns])

        st.markdown(f"**🧠 Diversity Score:** `{diversity_score:.2f}` ")

    # ⏱️ Execution Time
        st.markdown(f"**⏱️ Execution Time:** `{execution_time:.2f}` seconds")

    

# Clean up and prepare map data
        map_df = recommended_df.dropna(subset=['LATITUDE', 'LONGITUDE']).copy()
        map_df['LATITUDE'] = map_df['LATITUDE'].astype(float)
        map_df['LONGITUDE'] = map_df['LONGITUDE'].astype(float)
        map_df = map_df.rename(columns={'LATITUDE': 'lat', 'LONGITUDE': 'lon'})

# Auto-zoom logic based on spread
        lat_range = map_df["lat"].max() - map_df["lat"].min()
        lon_range = map_df["lon"].max() - map_df["lon"].min()

        if lat_range < 1 and lon_range < 1:
            zoom_level = 11
        elif lat_range < 5 and lon_range < 5:
            zoom_level = 9
        elif lat_range < 10 and lon_range < 10:
            zoom_level = 7
        else:
            zoom_level = 5

# Define the layer with tooltip info
        layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position='[lon, lat]',
        get_radius=500,
        get_fill_color='[0, 128, 255, 140]',
        pickable=True,
        )

# Tooltip to show on hover
        tooltip = {
        "html": "<b>{NAME}</b><br/>"
            "{ADDRESS}, {STATE_ABBR}<br/>"
            f"<b>Price:</b> {{{price_column}}}",
        "style": {
            "backgroundColor": "steelblue",
            "color": "white",
            "fontSize": "12px"
            }
        }

# Define the view with dynamic zoom
        view_state = pdk.ViewState(
            latitude=map_df["lat"].mean(),
            longitude=map_df["lon"].mean(),
            zoom=zoom_level,
            pitch=0,
        )

# Create the deck.gl chart
        deck = pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9",
            initial_view_state=view_state,
            layers=[layer],
            tooltip=tooltip,
        )

# Show in Streamlit
        st.subheader("🗺️ Recommended Properties Map with Details")
        st.pydeck_chart(deck)




