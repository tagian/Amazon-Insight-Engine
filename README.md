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

final_score = 0.6 * vader_score + 0.4 * normalized_rating
This produces a combined sentiment feature that reflects both written review sentiment and star rating.

### 6. Product Clustering

The notebook groups similar products within each category using:

- Product price
- Average rating
- Product description text

Text descriptions are preprocessed using:

- Lowercasing
- Punctuation removal
- Stopword removal
- Porter stemming

The clustering pipeline uses:

- TF-IDF vectorization for text features
- StandardScaler for numerical features
- K-Means clustering
- Elbow method with KneeLocator to choose the best number of clusters
- PCA visualization for 2D cluster plots
- Silhouette score for cluster quality evaluation

Some categories may be skipped during clustering if they do not contain enough valid price, rating, or description data.

### 7. Sentiment Classification

The notebook builds classification models to predict sentiment labels based on review text.

The workflow includes:

- Cleaning review text
- Tokenization
- Stopword removal
- Creating discrete sentiment labels from final sentiment scores
- Splitting data into train and test sets
- Comparing TF-IDF and FastText embeddings
- Training and evaluating multiple classifiers

Models used:

- Gaussian Naive Bayes
- K-Nearest Neighbors
- Random Forest

Evaluation metrics:

- Precision
- Recall
- F1-score
- Accuracy

The notebook uses 10-fold cross-validation on the training set, then evaluates the best feature representation for each model on the test set.
