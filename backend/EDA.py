#Define EDA functions required for Dashboard

import numpy as np
import pandas as pd
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import MarkerCluster
import plotly.express as px

def plot_property_type_pie_by_state(data):
    type_cols = [
        'Office', 'Retail', 'Industrial', 'Healthcare', 'Other',
        'Land', 'Investment', 'Multifamily', 'Hospitality'
    ]
    
    states = sorted(data['STATE_ABBR'].unique())
    state_type_data = {}
    
    for state in states:
        df_state = data[data['STATE_ABBR'] == state]
        type_counts = df_state[type_cols].sum()
        state_type_data[state] = type_counts

    first_state = states[0]
    values = state_type_data[first_state].values
    labels = state_type_data[first_state].index

    fig = go.Figure()

    # Initial Pie chart trace (with animation)
    fig.add_trace(go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        textinfo='percent+label',
        hovertemplate='%{label}<br>Count: %{value}<extra></extra>',
        pull=[0.05] * len(labels),  # Pull each slice out a little
        sort=False
    ))

    # Animation Frames
    frames = [go.Frame(
        data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            pull=[0.05] * len(labels),
            textinfo='percent+label',
            hovertemplate='%{label}<br>Count: %{value}<extra></extra>',
            sort=False
        )],
        name=str(state)
    ) for state, values in state_type_data.items()]

    fig.frames = frames

    # Dropdown for state selection
    dropdown_buttons = []
    for state in states:
        counts = state_type_data[state].values
        dropdown_buttons.append(
            dict(
                method="animate",
                label=state,
                args=[ [str(state)], {
                    'frame': {'duration': 500, 'redraw': True},
                    'fromcurrent': True,
                    'transition': {'duration': 300, 'easing': 'cubic-in-out'}
                }]
            )
        )

    fig.update_layout(
        title_text="Property Type Distribution by State",
        updatemenus=[{
            "buttons": dropdown_buttons,
            "direction": "down",
            "showactive": True,
            "x": 0.1,
            "xanchor": "left",
            "y": 1.15,
            "yanchor": "top"
        }],
        colorway=px.colors.qualitative.Pastel,
        legend=dict(orientation="h", y=-0.1),
        paper_bgcolor='white',
        title_font_size=22,
        transition={'duration': 500, 'easing': 'cubic-in-out'}  # Apply smooth loading effect
    )

    fig.update_traces(marker=dict(line=dict(color='#000000', width=1)))

    return fig


#Visualization 2
def plot_price_vs_size_bubble(data):

    data = data[(data['PRICE_Modified'] < 300_000_000)]
    
    data = data[
    (data['ACRES_Modified'] <= 100) &
    (data['BUILDING_SF_Modified_Numeric_Final'] <= 1_000_000)]

    data['ACRES_Modified'] = data['ACRES_Modified'].round(3)
    data['BUILDING_SF_Modified_Numeric_Final'] = data['BUILDING_SF_Modified_Numeric_Final'].round(3)

    
    # Determine dominant property type
    type_cols = ['Office', 'Retail', 'Industrial', 'Healthcare', 'Other', 'Land',
                 'Investment', 'Multifamily', 'Hospitality']
    def get_primary_type(row):
        for t in type_cols:
            if row[t] == True:
                return t
        return 'Uncategorized'

    data['PrimaryType'] = data.apply(get_primary_type, axis=1)

    fig = px.scatter(
        data,
        x='BUILDING_SF_Modified_Numeric_Final',
        y='PRICE_Modified',
        size='ACRES_Modified',
        color='PrimaryType',
        hover_data=['NAME', 'CITY', 'STATE_ABBR', 'PRICE_Modified'],
        title='Price vs. Building Size by Property Type',
        labels={
            'BUILDING_SF_Modified_Numeric_Final': 'Building Size (SF)',
            'PRICE_Modified': 'Price ($)',
        },
        size_max=30
    )

    fig.update_layout(
        xaxis_title="Building Size (SF)",
        yaxis_title="Price (USD)",
        legend_title="Property Type"
    )

    return fig

#Visualization 3
def plot_property_distribution_on_map(sale_data, lease_data, state_filter=None):
    """
    Plot sale and lease properties on a Folium map using CircleMarker for better performance.
    Optionally filters by state.
    """
    # Label property type
    sale_data = sale_data.copy()
    lease_data = lease_data.copy()
    sale_data['Property_Type'] = 'Sale'
    lease_data['Property_Type'] = 'Lease'

    # Combine datasets
    combined_data = pd.concat([sale_data, lease_data], ignore_index=True)

    # Filter by state if provided
    if state_filter and 'STATE_ABBR' in combined_data.columns:
        combined_data = combined_data[combined_data['STATE_ABBR'] == state_filter]

    # Create map
    folium_map = folium.Map(location=[37.0902, -95.7129], zoom_start=4)
    marker_cluster = MarkerCluster().add_to(folium_map)

    for _, row in combined_data.iterrows():
        lat = row['LATITUDE']
        lon = row['LONGITUDE']
        prop_type = row['Property_Type']
        popup_info = f"<b>Type:</b> {prop_type}"

        if 'NAME' in row:
            popup_info += f"<br><b>Name:</b> {row['NAME']}"
        if 'RATE_MAX' in row and not pd.isna(row['RATE_MAX']):
            popup_info += f"<br><b>Rate Max:</b> ${row['RATE_MAX']:,.0f}"
        if 'SALE_PRICE' in row and not pd.isna(row['SALE_PRICE']):
            popup_info += f"<br><b>Sale Price:</b> ${row['SALE_PRICE']:,.0f}"

        color = "green" if prop_type == "Lease" else "blue"

        folium.CircleMarker(
            location=[lat, lon],
            radius=4,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=folium.Popup(popup_info, max_width=300)
        ).add_to(marker_cluster)

    return folium_map


#Visualization 4
def plot_lease_rate_distribution_by_state(data):
    """
    Function to plot the Lease Rate Distribution by State for Lease Properties
    (with lease rates restricted to 300 million or less).
    Args:
    data (DataFrame): The dataset containing lease property details, including lease rates and state information.
    """
    # Filter out rows with missing or zero lease rates, and restrict rates to a maximum of 300 million
    data = data[(data['rate_per_sf_year'] >= 1.1) & (data['rate_per_sf_year'] <= 850)]

    # Create the boxplot for Lease Rate Distribution by State
    fig = px.box(
        data_frame=data,
        x='STATE_ABBR',  # X-axis: State
        y='rate_per_sf_year',  # Y-axis: Lease Rate
        title='Lease Rate Distribution Across States for Lease Properties ',
        labels={'STATE_ABBR': 'State', 'rate_per_sf_year': 'Lease Rate ($)'},
        points='all',  # Show all data points (outliers, etc.)
        hover_data={'STATE_ABBR': True, 'rate_per_sf_year': True},
        color='STATE_ABBR'  # Color by state for differentiation
    )
    
    # Update layout for better visualization
    fig.update_layout(
        xaxis_tickangle=45,  # Rotate state names for better readability
        title='Lease Rate Distribution Across States for Lease Properties',
        yaxis_title='Lease Rate ($)',
        showlegend=False,  # Hide the legend (as it’s not necessary here)
        boxmode='group'  # Group the boxplots by state
    )
    
    return fig

#Visualization 5
def plot_price_distribution_by_state(data):
    """
    Function to plot the Price Distribution by State for Sale Properties
    (with prices restricted to 300 million or less).
    Args:
    data (DataFrame): The dataset containing property details, including prices and state information.
    """
    # Filter out rows with missing or zero prices, and restrict prices to a maximum of 300 million
    data = data[(data['PRICE_Modified'] > 0) & (data['PRICE_Modified'] <= 300_000_000)]

    # Create the boxplot for Price Distribution by State
    fig = px.box(
        data_frame=data,
        x='STATE_ABBR',  # X-axis: State
        y='PRICE_Modified',  # Y-axis: Price
        title='Price Distribution Across States for Sale Properties)',
        labels={'STATE_ABBR': 'State', 'PRICE_Modified': 'Price ($)'},
        points='all',  # Show all data points (outliers, etc.)
        hover_data={'STATE_ABBR': True, 'PRICE_Modified': True},
        color='STATE_ABBR'  # Color by state for differentiation
    )
    
    # Update layout for better visualization
    fig.update_layout(
        xaxis_tickangle=45,  # Rotate state names for better readability
        title='Price Distribution Across States for Sale Properties',
        yaxis_title='Price ($)',
        showlegend=False,  # Hide the legend (as it’s not necessary here)
        boxmode='group'  # Group the boxplots by state
    )
    
    return fig

#Visualization 6
def plot_roi_choropleth(data):
    # Filter the data where Normalized_ROI > 1
    filtered_data = data[data["Normalized_ROI"] > 1]
    
    # Round the Normalized_ROI to 3 decimals
    filtered_data["Normalized_ROI"] = filtered_data["Normalized_ROI"].round(3)
    
    # Group by STATE_ABBR and calculate average, min, and max ROI for each state
    roi_by_state = filtered_data.groupby("STATE_ABBR")["Normalized_ROI"].agg(["mean", "min", "max"]).reset_index()
    
    # Round the average ROI (mean) to 3 decimals
    roi_by_state["mean"] = roi_by_state["mean"].round(3)
    
    # Create the choropleth map
    fig = px.choropleth(
        roi_by_state,
        locations="STATE_ABBR",
        locationmode="USA-states",
        color="mean",  # Use the mean for the color scale
        color_continuous_scale="YlGnBu",
        scope="usa",
        title="Average ROI by State",
        hover_name="STATE_ABBR",  # Show state name on hover
        hover_data={
            "mean": True, 
            "min": True,
            "max": True
        },  # Show average, min, and max ROI on hover
        labels={
            "mean": "Average ROI (%)", 
            "min": "Min ROI (%)", 
            "max": "Max ROI (%)"
        }  # Custom labels for hover data
    )

    return fig


