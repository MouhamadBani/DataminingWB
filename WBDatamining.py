import pandas as pd
import numpy as np
from scipy import stats

# Read the CSV file
df = pd.read_csv('WBData.csv')

# 1. Basic Statistics
def get_basic_stats(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    stats_df = df[numeric_columns].agg(['mean', 'median', 'std', 'min', 'max'])
    return stats_df

# 2. Time Series Analysis
def analyze_trends(df):
    # Group by year and calculate means
    yearly_means = df.groupby('year').agg({
        'GDP per capita (current US$)': 'mean',
        'Agricultural land (% of land area)': 'mean',
        'Forest area (% of land area)': 'mean',
        'Rural population (% of total population)': 'mean'
    }).round(2)
    return yearly_means

# 3. Correlation Analysis
def get_correlations(df):
    important_vars = [
        'GDP per capita (current US$)',
        'Agricultural land (% of land area)',
        'Forest area (% of land area)',
        'Rural population (% of total population)',
        'Literacy rate, adult total (% of people ages 15 and above)'
    ]
    return df[important_vars].corr().round(3)

# 4. Country Rankings
def get_country_rankings(df, year=2020):
    latest_data = df[df['year'] == year]
    rankings = pd.DataFrame({
        'GDP_Rank': latest_data['GDP per capita (current US$)'].rank(ascending=False),
        'Agricultural_Land_Rank': latest_data['Agricultural land (% of land area)'].rank(ascending=False),
        'Forest_Area_Rank': latest_data['Forest area (% of land area)'].rank(ascending=False)
    })
    return rankings.sort_values('GDP_Rank').head(10)

# 5. Regional Analysis
def analyze_regions(df, year=2020):
    # Assuming first two letters of country code represent region
    df['region'] = df['Country Code'].str[:2]
    regional_stats = df[df['year'] == year].groupby('region').agg({
        'GDP per capita (current US$)': 'mean',
        'Agricultural land (% of land area)': 'mean',
        'Forest area (% of land area)': 'mean'
    }).round(2)
    return regional_stats

# Execute analysis
print("\nBasic Statistics:")
print(get_basic_stats(df))

print("\nTime Series Analysis:")
print(analyze_trends(df))

print("\nCorrelation Analysis:")
print(get_correlations(df))

print("\nTop 10 Country Rankings (2020):")
print(get_country_rankings(df))

print("\nRegional Analysis (2020):")
print(analyze_regions(df))

# Additional Analysis: Agricultural Trends
ag_trends = df.groupby('year')[['Agricultural land (% of land area)', 
                               'Agricultural raw materials exports (% of merchandise exports)',
                               'Agriculture, forestry, and fishing, value added (% of GDP)']].mean()
print("\nAgricultural Trends Over Time:")
print(ag_trends.round(2))
