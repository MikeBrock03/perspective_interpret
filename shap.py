import pandas as pd
import numpy as np
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from googleapiclient import discovery
import json
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

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
        'requestedAttributes': {'TOXICITY': {}}
    }
    try:
        response = client.comments().analyze(body=analyze_request).execute()
        # Extract just the toxicity probability value from the response
        toxicity = response['attributeScores']['TOXICITY']['summaryScore']['value']
        return float(toxicity)
    except Exception as e:
        print(f"Error getting toxicity score: {str(e)}")
        return None

print("Starting SHAP analysis with Perspective API toxicity scores...")
print("This may take a while depending on API rate limits...")

# Get toxicity scores for all comments
print("Getting toxicity scores...")
df['toxicity_score'] = df['comment_text'].apply(get_toxicity_score)

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
    print("\nTop contributing words:")
    print(explanation.head())
    
    # Create force plot for this prediction
    plt.figure(figsize=(10,2))
    shap.force_plot(
        explainer.expected_value[0],
        explanation['shap_value'].values,
        explanation['token'].values,
        show=False,
        matplotlib=True
    )
    plt.title("SHAP Force Plot - Word Contributions")
    plt.tight_layout()
    plt.savefig(f'force_plot_{hash(text)}.png')
    plt.close()

# Create plots
print("\nGenerating visualizations...")
plt.figure(figsize=(15,10))

# Plot 1: Overall feature importance
shap.summary_plot(shap_values, X_combined, feature_names=feature_names_combined, 
                 show=False, plot_size=(10,6))
plt.title("SHAP Summary Plot: Impact of Toxicity Score and Words")
plt.tight_layout()
plt.savefig('shap_summary_all.png')
plt.close()

# Plot 2: Top 20 most important words
plt.figure(figsize=(12,8))
word_importance = np.abs(shap_values).mean(0)[1:]  # Skip toxicity_score
top_words_idx = np.argsort(word_importance)[-20:]
shap.summary_plot(shap_values[0][:,1:], X_words, 
                 feature_names=feature_names,
                 max_display=20,
                 show=False)
plt.title("Top 20 Most Influential Words")
plt.tight_layout()
plt.savefig('shap_word_importance.png')

# Save detailed word-level analysis
word_results = pd.DataFrame({
    'word': feature_names,
    'average_impact': np.abs(shap_values).mean(0)[1:],
    'direction': np.where(shap_values.mean(0)[1:] > 0, 'More Toxic', 'Less Toxic')
})
word_results = word_results.sort_values('average_impact', ascending=False)
word_results.to_csv('word_level_analysis.csv', index=False)

print("\nWord-Level Analysis Summary:")
print(f"Top 10 most influential words and their impact:")
print(word_results.head(10))
