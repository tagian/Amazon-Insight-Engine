# AmazonInsight

AmazonInsight is a Python/Google Colab notebook project for analyzing Amazon product reviews from the **McAuley-Lab Amazon Reviews 2023** dataset. The project explores ratings, review text, product metadata, clustering, sentiment scoring, and classification models across five Amazon product categories.

## Project Overview

This notebook performs exploratory data analysis and machine learning on Amazon review data for the following categories:

- Gift Cards
- Magazine Subscriptions
- Subscription Boxes
- All Beauty
- Digital Music

The analysis focuses on understanding product ratings, identifying low-rated high-review products, extracting common review keywords, grouping similar products, and building sentiment classification models.

## Features

### 1. Ratings and Review Analysis

The notebook analyzes rating distributions for each selected category using histograms and calculates the average product rating per category.

It also compares categories to identify which ones have noticeably higher or lower average ratings.

Example finding included in the notebook:

- Gift Cards had a relatively high average rating.
- Subscription Boxes had a relatively lower average rating.

### 2. Low-Rated High-Review Product Analysis

The project identifies products that have:

- A high number of reviews
- Ratings below the category average

For these products, the notebook extracts common words from review text and visualizes them using a word cloud.

### 3. Best-Selling Product Identification

For each category, the notebook identifies the top 5 products with the highest number of ratings/reviews.

For each product, it prints:

- Product title
- Review count
- Product features

### 4. Rating Trends Over Time

The notebook analyzes how average ratings change by month for each category.

It creates line plots showing monthly average rating trends to help identify possible seasonal patterns or changes in customer satisfaction.

### 5. Sentiment Feature Engineering

The project creates a custom sentiment score by combining:

- VADER sentiment score from review text
- Normalized numerical product rating

The final sentiment score is calculated as a weighted combination:

```python
final_score = 0.6 * vader_score + 0.4 * normalized_rating
