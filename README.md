# InvestSmart: Personalized Commercial Real Estate Deal Discovery Engine

InvestSmart is an AI-driven tool designed to simplify commercial real estate investments by delivering personalized property recommendations with a focus on high return on investment (ROI). By integrating natural language processing, geospatial analysis, and machine learning, InvestSmart helps investors quickly discover well-priced commercial properties aligned with their preferences.

## Key Features

- **Personalized Recommendations:**  
  Uses a hybrid recommendation model combining Sentence-BERT embeddings and geo-spatial nearest neighbor searches to match user preferences such as location, budget, property size, and transit proximity.

- **Deal Scoring:**  
  Employs Random Forest models to predict property price percentiles and calculates an unsupervised deal score to identify fairly priced, high-value opportunities.

- **Large-Scale Dataset:**  
  Built on a dataset of over 1 million U.S. commercial properties from diverse public and private sources.

- **Interactive Dashboard:**  
  Provides real-time visualizations of market trends, ROI estimates, pricing ranges, and key attributes to support informed decision-making.

## Performance Metrics

- Diversity score between 0.73 and 0.85, balancing relevance and variety.
- Fast execution with recommendation times ranging from 0.02 to 0.06 seconds even on 1 million+ listings.

## Technology Stack

- Python, Sentence-BERT, Scikit-learn (Random Forest), GeoPandas, BallTree
- Interactive dashboard with Streamlit UI

## Future Enhancements

- Incorporate collaborative filtering to leverage user behavior.
- Integrate user feedback for continuous recommendation improvements.
- Add wishlist functionality for saved properties.
- Enhance price and ROI prediction models.
- Expand geospatial analysis for neighborhood trend insights.

## How to Use

1. Input your preferences: location, budget, size, transit access.
2. Get top 5 personalized property recommendations with deal scores.
3. Explore visualizations to analyze market trends and investment potential.
