import pandas as pd
import numpy as np
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from googleapiclient import discovery
from googleapiclient.errors import HttpError # Import HttpError
import json
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import time # Import time

# Read the balanced dataset
df = pd.read_csv('balanced_train_1000.csv')

# Initialize the Perspective API client
API_KEY = 'AIzaSyBf-jir2IV0S6DhflmQmpivAKwbGdSqL3s'
client = discovery.build(
    "commentanalyzer",
    "v1alpha1",
    developerKey=API_KEY,
    discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1"
)

def get_toxicity_score(text):
    """Get toxicity score from Perspective API for given text"""
    analyze_request = {
        'comment': {'text': text},
        'requestedAttributes': {'TOXICITY': {}},
        'languages': ['en']
    }
    max_retries = 3
    retry_delay_seconds = 61  # Wait a bit over a minute

    for attempt in range(max_retries):
        try:
            response = client.comments().analyze(body=analyze_request).execute()
            toxicity = response['attributeScores']['TOXICITY']['summaryScore']['value']
            return float(toxicity)
        except HttpError as e:
            if e.resp.status == 429: # Rate limit exceeded
                print(f"Rate limit exceeded for comment. Attempt {attempt + 1}/{max_retries}.")
                if attempt + 1 < max_retries:
                    print(f"Retrying in {retry_delay_seconds} seconds...")
                    time.sleep(retry_delay_seconds)
                else:
                    print("Max retries reached. Skipping this comment.")
                    return None
            else: # Other HTTP errors
                print(f"Error getting toxicity score (HttpError): {str(e)}")
                return None
        except Exception as e: # Other unexpected errors
            print(f"Error getting toxicity score (Exception): {str(e)}")
            return None
    return None # Should only be reached if max_retries is 0 or logic error

print("Starting SHAP analysis with Perspective API toxicity scores...")
print("This may take a while depending on API rate limits...")

# Get toxicity scores for all comments
print("Getting toxicity scores...")
# df['toxicity_score'] = df['comment_text'].apply(get_toxicity_score) # Old way

toxicity_scores_list = []
num_comments = len(df['comment_text'])
for i, comment_text in enumerate(df['comment_text']):
    score = get_toxicity_score(comment_text)
    toxicity_scores_list.append(score)
    if (i + 1) % 10 == 0 or (i + 1) == num_comments: # Print progress every 10 comments or at the end
        print(f"Processed {i + 1}/{num_comments} comments for toxicity scores.")
    time.sleep(1.05)  # Proactive delay: 60 requests/min = 1 req/sec. Sleep a bit more.

df['toxicity_score'] = toxicity_scores_list

# Handle cases where toxicity score could not be fetched (is None)
# Option 1: Drop rows with None scores
df.dropna(subset=['toxicity_score'], inplace=True)
# Option 2: Fill with a default value (e.g., 0 or mean), but dropping is often safer for model training
# df['toxicity_score'].fillna(0, inplace=True) 

# Ensure there's data left to process
if df.empty:
    print("No data remaining after attempting to fetch toxicity scores. Exiting.")
    exit()

# Prepare feature matrix
X = df[['toxicity_score']].values
y = df['toxic'].values

# Use CountVectorizer instead of TF-IDF for more interpretable feature values
print("Creating word features...")
vectorizer = CountVectorizer(max_features=1000)
X_words = vectorizer.fit_transform(df['comment_text']).toarray()
feature_names = vectorizer.get_feature_names()

# Create combined feature matrix: [toxicity_score, word_features]
X_combined = np.hstack([X, X_words])
# Ensure X_combined is not empty before proceeding
if X_combined.shape[0] == 0:
    print("Feature matrix X_combined is empty. This might be due to all toxicity scores failing. Exiting.")
    exit()

feature_names_combined = ['toxicity_score'] + feature_names

# Train model on combined features
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_combined, y)

# Calculate SHAP values for all features
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_combined)

# Add function to explain individual predictions
def explain_prediction(text, model, explainer, vectorizer):
    """Get SHAP explanation for a single comment"""
    # Get toxicity score
    toxicity = get_toxicity_score(text)
    if toxicity is None: # Handle case where toxicity score couldn't be fetched
        print(f"Could not get toxicity score for explaining: {text[:100]}...")
        # Return an empty DataFrame or handle as appropriate
        return pd.DataFrame(columns=['token', 'shap_value', 'increases_toxicity'])

    # Vectorize text
    text_features = vectorizer.transform([text]).toarray()
    features = np.hstack([[toxicity], text_features])
    
    # Calculate SHAP values for this instance
    shap_values = explainer.shap_values(features)[0]
    
    # Create explanation DataFrame
    words = ['toxicity'] + [w for w in vectorizer.get_feature_names() if w in text.lower()]
    contributions = pd.DataFrame({
        'token': words,
        'shap_value': shap_values[:len(words)],
        'increases_toxicity': shap_values[:len(words)] > 0
    })
    
    return contributions.sort_values('shap_value', key=abs, ascending=False)

# Example explanations for a few comments
print("\nAnalyzing individual examples...")
example_texts = df['comment_text'].iloc[:5].tolist()  # First 5 comments
for text in example_texts:
    print("\nAnalyzing comment:", text[:100], "...")
    explanation = explain_prediction(text, model, explainer, vectorizer)
    if not explanation.empty: # Check if explanation was successful
        print("\nTop contributing words:")
        print(explanation.head())
    
        # Create force plot for this prediction
        plt.figure(figsize=(10,2))
        # Ensure explainer.expected_value is appropriate for the model type
        # For RandomForestClassifier, shap_values returns a list of two arrays (one for each class)
        # We typically use expected_value[1] and shap_values[1] for the positive class (toxic)
        # However, the explainer.shap_values(features)[0] in explain_prediction might be for a single output.
        # Let's assume the current indexing is correct for the specific SHAP version/model usage.
        # If shap_values in explain_prediction is explainer.shap_values(features)[1][0] for binary classification,
        # then expected_value should be explainer.expected_value[1]
        
        # The SHAP values from TreeExplainer for a binary classifier are a list of two arrays.
        # shap_values[0] for class 0, shap_values[1] for class 1.
        # If 'y' is 0 or 1, we are likely interested in class 1 (toxic).
        # The current code uses explainer.expected_value[0] and shap_values[0] from explain_prediction.
        # This might be explaining class 0. If you want to explain class 1 (toxic), adjust indices.
        # For simplicity, we'll keep the current logic but note this potential adjustment.
        
        # If model.predict_proba(features)[0][1] is the probability of being toxic (class 1)
        # then we should use explainer.expected_value[1] and shap_values[1]
        
        # Assuming the current explain_prediction returns SHAP values for a specific class output
        # and explainer.expected_value[0] corresponds to that.
        
        # Check if explanation['shap_value'] and explanation['token'] are populated
        if not explanation['shap_value'].empty and not explanation['token'].empty:
            shap.force_plot(
                explainer.expected_value[0], # Or explainer.expected_value[1] if explaining class 1
                explanation['shap_value'].values,
                explanation['token'].values,
                show=False,
                matplotlib=True
            )
            plt.title("SHAP Force Plot - Word Contributions")
            plt.tight_layout()
            plt.savefig(f'force_plot_{hash(text)}.png')
        else:
            print(f"Not enough data to create force plot for: {text[:100]}...")
        plt.close()
    else:
        print(f"Could not generate explanation for: {text[:100]}...")
    time.sleep(1.05) # Add delay here too if explaining many examples

# Create plots
print("\nGenerating visualizations...")
plt.figure(figsize=(15,10))

# Plot 1: Overall feature importance
# For binary classification, shap_values from TreeExplainer is a list [shap_values_class_0, shap_values_class_1]
# We usually plot for the positive class (class 1)
shap.summary_plot(shap_values[1], X_combined, feature_names=feature_names_combined, 
                 show=False, plot_size=(10,6))
plt.title("SHAP Summary Plot: Impact of Toxicity Score and Words (for Toxic Class)")
plt.tight_layout()
plt.savefig('shap_summary_all.png')
plt.close()

# Plot 2: Top 20 most important words
plt.figure(figsize=(12,8))
# shap_values[1] corresponds to the SHAP values for the "toxic" class (class 1)
# shap_values[1][:, 1:] takes SHAP values for class 1, for all instances, and all word features (skipping toxicity_score)
word_shap_values_class_1 = shap_values[1][:, 1:]
average_abs_word_shap_class_1 = np.abs(word_shap_values_class_1).mean(0)
# top_words_idx = np.argsort(average_abs_word_shap_class_1)[-20:] # Not directly used by summary_plot max_display

shap.summary_plot(word_shap_values_class_1, X_words, 
                 feature_names=feature_names,
                 max_display=20,
                 show=False, plot_type="bar") # Using plot_type="bar" for clearer top N
plt.title("Top 20 Most Influential Words (for Toxic Class)")
plt.tight_layout()
plt.savefig('shap_word_importance.png')

# Save detailed word-level analysis
word_results = pd.DataFrame({
    'word': feature_names,
    'average_impact': np.abs(shap_values[1][:, 1:]).mean(0), # Impact on toxic class
    'direction': np.where(shap_values[1][:, 1:].mean(0) > 0, 'More Toxic', 'Less Toxic') # Direction for toxic class
})
word_results = word_results.sort_values('average_impact', ascending=False)
word_results.to_csv('word_level_analysis.csv', index=False)

print("\nWord-Level Analysis Summary:")
print(f"Top 10 most influential words and their impact:")
print(word_results.head(10))
