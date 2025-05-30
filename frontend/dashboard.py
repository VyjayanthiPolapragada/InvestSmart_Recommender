import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from streamlit_folium import st_folium

import streamlit as st
import pandas as pd
from backend.EDA import (
    plot_property_type_pie_by_state,
    plot_price_vs_size_bubble,
    plot_property_distribution_on_map,
    plot_lease_rate_distribution_by_state,
    plot_price_distribution_by_state,
    plot_roi_choropleth
)

# Load data 
def run_dashboard(sales_data, lease_data, sales_transit, lease_transit):

#Merge data
    sales_merged_df = pd.merge(sales_data, sales_transit[['ID', 'has_transit_1mi', 'has_transit_2mi', 'has_transit_5mi']], 
                     on='ID', 
                     how='left')

    lease_merged_df = pd.merge(lease_data, lease_transit[['ID', 'has_transit_1mi', 'has_transit_2mi', 'has_transit_5mi']], 
                     on='ID', 
                     how='left')


    #st.set_page_config(page_title="InvestSmart - Personlized CRE", layout="wide")

    st.title("InvestSmart: Dashboard")
    st.markdown("Explore trends in property types, pricing, and lease rates across the U.S.")

# Sidebar for user selection 
    option = st.sidebar.radio(
        "📊 What would you like to explore?",
        (
         "🏘️ Property Types by State",
         "📏 Price vs. Property Size",
         "💰 Lease Rates Across States",
         "🗺️ See Properties on the Map",
         "🏷️ Sale Price Comparison by State",
         "📍 ROI by State" 
     )
    )

# Display the selected chart based on the user's choice
    if option == "🏘️ Property Types by State":
        st.subheader("🏘️ Property Types by State")
        fig1 = plot_property_type_pie_by_state(sales_merged_df)
        st.plotly_chart(fig1)

    elif option == "📏 Price vs. Property Size":
        st.subheader("📏 Price vs. Property Size")
        fig2 = plot_price_vs_size_bubble(sales_merged_df)
        st.plotly_chart(fig2)

    elif option == "💰 Lease Rates Across States":
        st.subheader("💰 Lease Rates Across States")
        fig3 = plot_lease_rate_distribution_by_state(lease_merged_df)
        st.plotly_chart(fig3)

    elif option == "🗺️ See Properties on the Map":
        st.subheader("🗺️ Property Locations on the Map")

    # Optional: You can keep the state selector but don't filter if you're using a static map
    #selected_state = st.selectbox("Select a State to Filter", 
                                  #options=['All'] + sorted(sales_merged_df['STATE_ABBR'].dropna().unique()))
    #if selected_state != 'All':
        #st.warning("This is a pre-generated map. State filtering is disabled for faster loading.")

    # Show spinner while loading HTML
        with st.spinner("Loading the map..."):
            with open("frontend/preloaded_property_map.html", "r", encoding="utf-8") as f:
                folium_html = f.read()

        st.success("Map loaded successfully!")
        st.components.v1.html(folium_html, width=1100, height=600, scrolling=False)


    elif option == "🏷️ Sale Price Comparison by State":
        st.subheader("🏷️ Sale Price Comparison by State")
        fig5 = plot_price_distribution_by_state(sales_merged_df)
        st.plotly_chart(fig5)

    elif option == "📍 ROI by State":
        st.subheader("📍 ROI by State")
        fig_roi = plot_roi_choropleth(sales_merged_df)  # Call the ROI choropleth function
        st.plotly_chart(fig_roi)  # Display the ROI choropleth map
    


