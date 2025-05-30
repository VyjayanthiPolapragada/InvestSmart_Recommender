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

## How It Works

1. Users input preferences (e.g., location, budget, size, transit access).
2. The hybrid model finds top 5 listings using semantic similarity and spatial filtering.
3. Deal scoring highlights properties with the highest estimated ROI.
4. Results and trends are visualized through an interactive dashboard.

## Performance Metrics

- Diversity score between 0.73 and 0.85, balancing relevance and variety.
- Fast execution with recommendation times ranging from 0.02 to 0.06 seconds even on 1 million+ listings.

## Data Sources

- **Commercial Property Listings:** Dewey Data  
- **Transit Location Data:** U.S. Department of Transportation Bureau of Transportation Statistics

## Technology Stack

- **Languages/Tools:** Python, Pandas, Scikit-learn, GeoPandas, Sentence-BERT
- **Models:** Hybrid recommender (BERT + geospatial), Random Forest percentile prediction
- **Spatial Search:** BallTree for efficient transit proximity filtering


## Future Enhancements

- Incorporate collaborative filtering to leverage user behavior.
- Integrate user feedback for continuous recommendation improvements.
- Add wishlist functionality for saved properties.
- Enhance price and ROI prediction models.
- Expand geospatial analysis for neighborhood trend insights.

Built for smarter investing, InvestSmart empowers commercial property investors with intelligent, data-driven insights.

