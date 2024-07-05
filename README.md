# Geospatial Analysis of Voting Behavior in Akwa Ibom State

## Overview
This project conducts a geospatial analysis of voting patterns in Akwa Ibom state, Nigeria, to identify potential voting irregularities and enhance election transparency. It's part of the HNG Internship program (July 2024).

## Features
- Geospatial clustering of polling units
- Calculation of outlier scores for voting patterns
- Visualization of outliers through histograms and maps
- Identification of top outliers for each political party

## Data
The analysis uses the `AKWA IBOM_crosschecked_geocoded.csv` dataset, which includes:
- Polling unit locations (latitude and longitude)
- Vote counts for different parties (APC, PDP, LP, NNPP)

## Requirements
- Python 3.x
- Libraries: pandas, numpy, scipy, matplotlib, seaborn, cartopy, geopandas

## Usage
1. Clone the repository
2. Install required libraries: `pip install -r requirements.txt`
3. Run the Jupyter notebook or Python script

## Key Outputs
1. `Polling_Unit_Outlier_Scores.xlsx`: Excel file with sorted outlier scores for each party
2. Visualizations:
   - Histograms of outlier score distributions
   - Map of top 5 outliers for each party in Akwa Ibom state

## Insights
- APC and PDP show more varied support across polling units
- LP exhibits more localized support patterns
- NNPP demonstrates consistency or limited support across units
- Geographic clusters of outliers identified for each party

## Future Work
- Detailed investigation of top outliers
- More granular geographic analysis
- Correlation with demographic or socio-economic factors

## Author
Aniekan Daniel - HNG Intern

## Acknowledgments
- HNG Internship Program
- Independent National Electoral Commission (INEC) for the election data
