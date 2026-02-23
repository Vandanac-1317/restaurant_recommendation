import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# ---------------------------
# Load and Preprocess Dataset
# ---------------------------
df = pd.read_csv('restaurant_dataset.csv')

def preprocess_data(df):
    # Handle missing values
    df['Cuisines'] = df['Cuisines'].fillna('Unknown')
    df['Average Cost for two'] = df['Average Cost for two'].fillna(df['Average Cost for two'].median())
    df['Aggregate rating'] = df['Aggregate rating'].fillna(df['Aggregate rating'].median())
    df['Price range'] = df['Price range'].fillna(2)  # default to medium
    df['City'] = df['City'].fillna('Unknown')

    # Convert cuisines to list
    df['Cuisine List'] = df['Cuisines'].apply(lambda x: [c.strip() for c in str(x).split(',')])
    return df

df = preprocess_data(df)

# ---------------------------
# Feature Engineering
# ---------------------------
def create_feature_matrix(df):
    mlb = MultiLabelBinarizer()
    cuisine_matrix = mlb.fit_transform(df['Cuisine List'])
    cuisine_df = pd.DataFrame(cuisine_matrix, columns=mlb.classes_)

    # Normalize numerical features
    scaler = StandardScaler()
    numerical_features = ['Average Cost for two', 'Aggregate rating', 'Price range', 'Votes']
    numerical_df = pd.DataFrame(
        scaler.fit_transform(df[numerical_features]),
        columns=numerical_features
    )

    feature_matrix = pd.concat([cuisine_df.reset_index(drop=True), numerical_df.reset_index(drop=True)], axis=1)
    return feature_matrix, mlb.classes_, scaler

feature_matrix, cuisine_classes, global_scaler = create_feature_matrix(df)

# ---------------------------
# Recommender Class
# ---------------------------
class RestaurantRecommender:
    def __init__(self, df, feature_matrix, scaler):
        self.df = df
        self.feature_matrix = feature_matrix
        self.similarity_matrix = cosine_similarity(feature_matrix)
        self.scaler = scaler

    def get_recommendations(self, user_preferences, top_n=10):
        """
        Get restaurant recommendations based on user preferences
    
        Parameters:
        user_preferences: dict with keys 'cuisines', 'price_range', 'min_rating', 'city'
        top_n: number of recommendations to return
        """
    # Create user profile vector
        user_vector = self._create_user_vector(user_preferences)

    # Calculate similarity scores
        similarity_scores = cosine_similarity([user_vector], self.feature_matrix)[0]

    # Get top recommendations
        top_indices = similarity_scores.argsort()[-top_n:][::-1]
        recommendations = self.df.iloc[top_indices].copy()
        recommendations['Similarity Score'] = similarity_scores[top_indices]

    # Apply additional filters
        if 'min_rating' in user_preferences and user_preferences['min_rating']:
            recommendations = recommendations[recommendations['Aggregate rating'] >= user_preferences['min_rating']]

        if 'city' in user_preferences and user_preferences['city']:
            recommendations = recommendations[recommendations['City'].str.contains(user_preferences['city'], case=False)]

    # If no results remain, return fallback (top-rated restaurants)
        if recommendations.empty:
            print("⚠️ No exact matches found, showing fallback top-rated restaurants...")
            recommendations = self.df.nlargest(top_n, 'Aggregate rating').copy()
            recommendations['Similarity Score'] = 0.0  # ensure column exists

        return recommendations[['Restaurant Name', 'City', 'Cuisines',
                            'Average Cost for two', 'Price range',
                            'Aggregate rating', 'Similarity Score']]


    def _create_user_vector(self, preferences):
        user_vector = np.zeros(self.feature_matrix.shape[1])

        # Cuisine preferences
        if 'cuisines' in preferences:
            for cuisine in preferences['cuisines']:
                if cuisine in cuisine_classes:
                    idx = list(cuisine_classes).index(cuisine)
                    user_vector[idx] = 1

        # Numerical preferences
        numerical_features = ['Average Cost for two', 'Aggregate rating', 'Price range', 'Votes']
        default_values = {
            'Average Cost for two': self.df['Average Cost for two'].median(),
            'Aggregate rating': self.df['Aggregate rating'].median(),
            'Price range': 2,
            'Votes': self.df['Votes'].median()
        }

        values = [
            preferences.get('average_cost_for_two', default_values['Average Cost for two']),
            preferences.get('min_rating', default_values['Aggregate rating']),
            preferences.get('price_range', default_values['Price range']),
            preferences.get('votes', default_values['Votes'])
        ]

        normalized_values = self.scaler.transform([values])[0]
        user_vector[len(cuisine_classes):] = normalized_values
        return user_vector

# ---------------------------
# Initialize Recommender
# ---------------------------
recommender = RestaurantRecommender(df, feature_matrix, global_scaler)

# Example run
user_prefs = {'cuisines': ['Italian', 'Pizza'], 'price_range': 3, 'min_rating': 4.0, 'city': 'New York'}
print("Top Recommendations:")
print(recommender.get_recommendations(user_prefs, top_n=10))

user_prefs2 = {'cuisines': ['Japanese', 'Sushi'], 'price_range': 4, 'min_rating': 4.5, 'city': 'Tokyo'}
print("\nJapanese Restaurant Recommendations:")
print(recommender.get_recommendations(user_prefs2, top_n=10))

# ---------------------------
# Evaluation
# ---------------------------
def evaluate_recommendations(recommender, test_cases):
    results = []
    for i, (user_prefs, expected_cuisines) in enumerate(test_cases):
        recommendations = recommender.get_recommendations(user_prefs)
        matched = 0
        for _, row in recommendations.iterrows():
            restaurant_cuisines = [c.strip() for c in str(row['Cuisines']).split(',')]
            if any(cuisine in restaurant_cuisines for cuisine in expected_cuisines):
                matched += 1
        precision = matched / len(recommendations) if len(recommendations) > 0 else 0
        results.append({'test_case': i, 'precision': precision})
    return pd.DataFrame(results)

test_cases = [
    ({'cuisines': ['Italian'], 'min_rating': 4.0}, ['Italian', 'Pizza']),
    ({'cuisines': ['Japanese'], 'price_range': 4}, ['Japanese', 'Sushi']),
    ({'cuisines': ['Indian'], 'city': 'Delhi'}, ['Indian', 'North Indian'])
]

evaluation_results = evaluate_recommendations(recommender, test_cases)
print("\nEvaluation Results:")
print(evaluation_results)
print(f"Average Precision: {evaluation_results['precision'].mean():.2f}")

# ---------------------------
# Interactive CLI
# ---------------------------
def interactive_recommender():
    print("\nWelcome to the Restaurant Recommendation System!")
    cuisines = input("Enter preferred cuisines (comma-separated): ").split(',')
    cuisines = [c.strip() for c in cuisines if c.strip()]
    price_range = input("Enter price range (1-4, or press Enter for any): ")
    price_range = int(price_range) if price_range else None
    min_rating = input("Enter minimum rating (0-5, or press Enter for any): ")
    min_rating = float(min_rating) if min_rating else None
    city = input("Enter city preference (or press Enter for any): ").strip()

    preferences = {}
    if cuisines: preferences['cuisines'] = cuisines
    if price_range: preferences['price_range'] = price_range
    if min_rating: preferences['min_rating'] = min_rating
    if city: preferences['city'] = city

    recommendations = recommender.get_recommendations(preferences)
    print(f"\nHere are your top {len(recommendations)} recommendations:")
    for i, (_, row) in enumerate(recommendations.iterrows(), 1):
        print(f"{i}. {row['Restaurant Name']} - {row['City']} | {row['Cuisines']} | "
              f"Price: {row['Price range']}, Rating: {row['Aggregate rating']} "
              f"(Similarity: {row['Similarity Score']:.3f})")

# ---------------------------
# Visualizations
# ---------------------------
cuisine_counter = Counter([c for sublist in df['Cuisine List'] for c in sublist])
top_cuisines = pd.DataFrame(cuisine_counter.most_common(10), columns=['Cuisine', 'Count'])

plt.figure(figsize=(10,6))
sns.barplot(x='Count', y='Cuisine', data=top_cuisines, hue=None, palette='viridis')
plt.title("Top 10 Cuisines in Dataset")
plt.show()

plt.figure(figsize=(8,6))
sns.countplot(x='Price range', data=df, palette='Set2', hue=None)
plt.title("Distribution of Restaurants by Price Range")
plt.show()

plt.figure(figsize=(8,6))
sns.histplot(df['Aggregate rating'], bins=20, kde=True, color='skyblue')
plt.title("Distribution of Restaurant Ratings")
plt.show()

sample_idx = np.random.choice(len(feature_matrix), 20, replace=False)
similarities = cosine_similarity(feature_matrix.iloc[sample_idx])

plt.figure(figsize=(10,8))
sns.heatmap(similarities, cmap="YlGnBu")
plt.title("Restaurant Similarity Heatmap (Sample of 20)")
plt.show()

plt.figure(figsize=(8,6))
sns.barplot(x='test_case', y='precision', data=evaluation_results, hue=None, palette='magma')
plt.title("Recommendation System Evaluation (Precision per Test Case)")
plt.ylim(0,1)
plt.show()
