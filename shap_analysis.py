import pandas as pd
import numpy as np
import shap
from googleapiclient import discovery
from googleapiclient.errors import HttpError
import json
import matplotlib.pyplot as plt
import time
import re
from typing import List, Union
from sklearn.feature_extraction.text import CountVectorizer
import xgboost as xgb
from sklearn.linear_model import LinearRegression

# Configuration variables for testing
TEST_CONFIG = {
    'max_samples': 500,  # Number of texts to analyze (500 toxic + 500 non-toxic)
    'max_words': 50,    # Maximum number of words to analyze per text
    'rate_limit_delay': 1.2,  # Delay between API calls in seconds
    'min_word_frequency': 5,  # Minimum number of times a word must appear to be included in analysis
    'top_n_words': 20,   # Number of top words to show in concise visualizations
    'detailed_top_n_words': 100,  # Number of words to show in detailed visualizations
    'max_word_length': 3,  # Minimum word length to consider
    'use_xgboost': True,  # Flag to use XGBoost instead of linear regression
    'common_words': {  # Words to exclude from analysis
        'the', 'and', 'that', 'have', 'for', 'not', 'this', 'but', 'with', 'you', 'from', 'they',
        'say', 'will', 'one', 'all', 'would', 'there', 'their', 'what', 'about', 'which', 'when',
        'make', 'like', 'time', 'just', 'know', 'people', 'into', 'year', 'some', 'could',
        'them', 'see', 'other', 'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over',
        'think', 'also', 'back', 'after', 'use', 'two', 'how', 'our', 'first', 'well',
        'way', 'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day', 'most', 'us',
        'been', 'much', 'using', 'their', 'both',  'user', 'made'
    }
}

# Read the dataset
df = pd.read_csv('balanced_train_1000.csv')

# Balance the dataset
toxic_samples = df[df['toxic'] == 1].head(TEST_CONFIG['max_samples'] // 2)
non_toxic_samples = df[df['toxic'] == 0].head(TEST_CONFIG['max_samples'] // 2)
df = pd.concat([toxic_samples, non_toxic_samples])
df = df.sample(frac=1)  # Shuffle the dataset

print(f"Selected {len(toxic_samples)} toxic and {len(non_toxic_samples)} non-toxic samples")

# Initialize the Perspective API client
API_KEY = 'AIzaSyBf-jir2IV0S6DhflmQmpivAKwbGdSqL3s'
client = discovery.build(
    "commentanalyzer",
    "v1alpha1",
    developerKey=API_KEY,
    discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1"
)

def get_toxicity_score(text: str) -> float:
    """Get toxicity score from Perspective API for given text"""
    if not text.strip():  # Handle empty text
        return 0.0
        
    analyze_request = {
        'comment': {'text': text},
        'requestedAttributes': {'TOXICITY': {}},
        'languages': ['en']
    }
    max_retries = 3
    retry_delay_seconds = 61

    for attempt in range(max_retries):
        try:
            response = client.comments().analyze(body=analyze_request).execute()
            toxicity = response['attributeScores']['TOXICITY']['summaryScore']['value']
            return float(toxicity)
        except HttpError as e:
            if e.resp.status == 429:  # Rate limit exceeded
                print(f"Rate limit exceeded. Attempt {attempt + 1}/{max_retries}.")
                if attempt + 1 < max_retries:
                    print(f"Retrying in {retry_delay_seconds} seconds...")
                    time.sleep(retry_delay_seconds)
                else:
                    print("Max retries reached. Returning default score.")
                    return 0.0
            else:
                print(f"HTTP Error: {str(e)}")
                return 0.0
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            return 0.0
    return 0.0

def tokenize_text(text: str) -> List[str]:
    """Simple tokenization that preserves word boundaries"""
    # Split on whitespace and punctuation, keeping structure
    tokens = re.findall(r'\w+|\s+|[^\w\s]', text)
    return [token for token in tokens if token.strip()]

class PerspectiveModel:
    """Model wrapper for Perspective API that supports both linear and XGBoost models"""
    def __init__(self, use_xgboost=True):
        self.vectorizer = CountVectorizer()
        self.use_xgboost = use_xgboost
        self.model = None
        self.X_train_mean = None
        
    def fit(self, texts: List[str]):
        """Fit the model by getting toxicity scores and learning coefficients"""
        # First, create the vocabulary
        self.vectorizer.fit(texts)
        
        # Get toxicity scores for each text
        scores = []
        for text in texts:
            score = get_toxicity_score(text)
            # Validate score is between 0 and 1
            if not 0 <= score <= 1:
                print(f"Warning: Score {score} outside expected range [0,1] for text: {text[:50]}...")
                score = max(0, min(1, score))  # Clamp to [0,1]
            scores.append(score)
            time.sleep(TEST_CONFIG['rate_limit_delay'])
        
        # Transform texts to feature matrix
        X = self.vectorizer.transform(texts).toarray()
        self.X_train_mean = X.mean(axis=0)
        
        if self.use_xgboost:
            # Initialize and fit XGBoost model
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,  # L1 regularization helps with sparse features
                reg_lambda=1.0,  # L2 regularization
                random_state=42
            )
            self.model.fit(X, scores)
        else:
            # Use linear regression
            self.model = LinearRegression()
            self.model.fit(X, scores)
        
        return self
    
    def predict(self, texts: List[str]) -> np.ndarray:
        """Predict toxicity scores for new texts"""
        X = self.vectorizer.transform(texts).toarray()
        predictions = self.model.predict(X)
        # Ensure predictions are in [0,1] range
        return np.clip(predictions, 0, 1)
    
    def get_feature_importance(self):
        """Get feature importance scores"""
        if self.use_xgboost:
            return self.model.feature_importances_
        else:
            return self.model.coef_

def calculate_shap_values(text: str, model: PerspectiveModel) -> dict:
    """
    Calculate SHAP values for words in the text using both manual calculation
    and SHAP's explainer
    """
    print(f"\nAnalyzing text: {text[:100]}...")
    
    # Transform the text to feature vector
    X = model.vectorizer.transform([text]).toarray()[0]
    
    # Get baseline score
    baseline_score = model.predict([text])[0]
    print(f"Baseline toxicity score: {baseline_score:.4f}")
    
    # Get unique words in vocabulary
    words = model.vectorizer.get_feature_names_out()
    word_indices = {word: idx for idx, word in enumerate(words)}
    
    # Select words to analyze
    text_words = set(tokenize_text(text))
    selected_words = [word for word in text_words if word in word_indices][:TEST_CONFIG['max_words']]
    selected_indices = [word_indices[word] for word in selected_words]
    
    # Calculate SHAP values using appropriate explainer
    if model.use_xgboost:
        # For XGBoost, we'll use a background dataset
        background = model.X_train_mean.reshape(1, -1)
        explainer = shap.TreeExplainer(model.model, background)
        shap_values = explainer.shap_values(X.reshape(1, -1))[0]
    else:
        explainer = shap.LinearExplainer(
            (model.model.coef_, model.model.intercept_),
            shap.maskers.Independent(model.X_train_mean.reshape(1, -1))
        )
        shap_values = explainer.shap_values(X.reshape(1, -1))[0]
    
    # Create results DataFrame with corrected interpretation
    results_df = pd.DataFrame({
        'word': selected_words,
        'shap_value': [shap_values[idx] for idx in selected_indices],
        'contributes_to_toxicity': [shap_values[idx] > 0 for idx in selected_indices],
        'absolute_impact': [abs(shap_values[idx]) for idx in selected_indices]
    })
    
    # Sort by absolute impact
    results_df = results_df.reindex(results_df['absolute_impact'].sort_values(ascending=False).index)
    
    return {
        "baseline_score": baseline_score,
        "results": results_df,
        "shap_values_full": shap_values,
        "text": text
    }

def visualize_explanation(word_summary_df: pd.DataFrame, output_prefix: str = 'perspective_api'):
    """Create visualizations for the overall SHAP explanation"""
    # Add model type to output prefix
    model_type = "xgboost" if TEST_CONFIG['use_xgboost'] else "linear"
    output_prefix = f"{output_prefix}_{model_type}"
    
    # Filter out common words and short words
    word_summary_df = word_summary_df[
        (~word_summary_df['word'].isin(TEST_CONFIG['common_words'])) &
        (word_summary_df['word'].str.len() > TEST_CONFIG['max_word_length'])
    ]
    
    # Separate positive and negative impacts
    positive_impacts = word_summary_df[word_summary_df['mean_impact'] > 0].sort_values('mean_impact', ascending=False)
    negative_impacts = word_summary_df[word_summary_df['mean_impact'] < 0].sort_values('mean_impact')
    
    # 1. Top positive contributors
    plt.figure(figsize=(15, 10))
    top_positive = positive_impacts.head(TEST_CONFIG['top_n_words'])
    plt.barh(range(len(top_positive)), top_positive['mean_impact'], color='red')
    plt.yticks(range(len(top_positive)), top_positive['word'])
    plt.xlabel('Mean SHAP Value')
    plt.title('Top Words Increasing Toxicity')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_positive_contributors.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Top negative contributors
    plt.figure(figsize=(15, 10))
    top_negative = negative_impacts.head(TEST_CONFIG['top_n_words'])
    plt.barh(range(len(top_negative)), top_negative['mean_impact'], color='blue')
    plt.yticks(range(len(top_negative)), top_negative['word'])
    plt.xlabel('Mean SHAP Value')
    plt.title('Top Words Decreasing Toxicity')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_negative_contributors.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Impact distribution
    plt.figure(figsize=(15, 10))
    plt.hist(word_summary_df['mean_impact'], bins=50, color='gray', alpha=0.7)
    plt.xlabel('Mean SHAP Value')
    plt.ylabel('Number of Words')
    plt.title('Distribution of Word Impacts')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_impact_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create detailed visualizations for positive and negative impacts
    for impact_type, data in [('positive', positive_impacts), ('negative', negative_impacts)]:
        plt.figure(figsize=(20, 30))  # Larger figure for more words
        top_words = data.head(TEST_CONFIG['detailed_top_n_words'])
        
        # Create bar plot with error bars
        plt.barh(range(len(top_words)), top_words['mean_impact'],
                xerr=top_words['std_impact'],
                color='red' if impact_type == 'positive' else 'blue',
                alpha=0.7)
        
        plt.yticks(range(len(top_words)), top_words['word'])
        plt.xlabel('Mean SHAP Value (with standard deviation)')
        plt.title(f'Detailed Analysis of {TEST_CONFIG["detailed_top_n_words"]} Words {impact_type.capitalize()}ly Impacting Toxicity')
        plt.grid(axis='x', alpha=0.3)
        
        # Add frequency information
        for i, (_, row) in enumerate(top_words.iterrows()):
            plt.text(row['mean_impact'], i, f" (freq: {row['frequency']})",
                    va='center', ha='left' if impact_type == 'positive' else 'right')
        
        plt.tight_layout()
        plt.savefig(f'{output_prefix}_{impact_type}_impacts_detailed.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Visualizations saved as:")
    print(f"- {output_prefix}_positive_contributors.png (top positive contributors)")
    print(f"- {output_prefix}_negative_contributors.png (top negative contributors)")
    print(f"- {output_prefix}_impact_distribution.png (distribution of impacts)")
    print(f"- {output_prefix}_positive_impacts_detailed.png (detailed positive contributors)")
    print(f"- {output_prefix}_negative_impacts_detailed.png (detailed negative contributors)")

# Main analysis
print("Starting SHAP analysis of Perspective API...")
print(f"Using {'XGBoost' if TEST_CONFIG['use_xgboost'] else 'Linear Regression'} model")
print("This will make API calls and take time due to rate limits...")

# First, fit the model on a subset of the data
print("\nFitting model on training data...")
model = PerspectiveModel(use_xgboost=TEST_CONFIG['use_xgboost'])
model.fit(df['comment_text'].tolist())

# Analyze example texts
sample_texts = df['comment_text'].tolist()

all_explanations = []
word_impacts = {}  # Dictionary to track word impacts across all texts

for i, text in enumerate(sample_texts):
    print(f"\n{'='*60}")
    print(f"Analyzing example {i+1}/{len(sample_texts)}")
    print(f"{'='*60}")
    
    try:
        explanation = calculate_shap_values(text, model)
        
        if "error" not in explanation:
            # Update word impacts dictionary
            for _, row in explanation["results"].iterrows():
                word = row['word']
                if word not in word_impacts:
                    word_impacts[word] = []
                word_impacts[word].append({
                    'shap_value': row['shap_value'],
                    'text': text[:100] + "..."
                })
            
            all_explanations.append(explanation)
        else:
            print(f"Error in explanation: {explanation['error']}")
            
    except Exception as e:
        print(f"Error analyzing text: {str(e)}")
        import traceback
        traceback.print_exc()
        continue

# Create overall word impact summary
print("\nCreating overall word impact summary...")
word_summary = []
for word, impacts in word_impacts.items():
    if len(impacts) < TEST_CONFIG['min_word_frequency']:
        continue
        
    shap_values = [imp['shap_value'] for imp in impacts]
    
    # Calculate statistics on the SHAP values
    mean_impact = np.mean(shap_values)
    std_impact = np.std(shap_values)
    max_impact = np.max(shap_values)
    min_impact = np.min(shap_values)
    
    word_summary.append({
        'word': word,
        'mean_impact': mean_impact,
        'std_impact': std_impact,
        'frequency': len(impacts),
        'max_impact': max_impact,
        'min_impact': min_impact,
        'impact_range': max_impact - min_impact
    })

# Convert to DataFrame and sort by absolute mean impact
word_summary_df = pd.DataFrame(word_summary)
if not word_summary_df.empty:
    # Filter out words with very small impacts
    word_summary_df = word_summary_df[abs(word_summary_df['mean_impact']) > 0.001]
    
    # Sort by absolute impact
    word_summary_df = word_summary_df.reindex(
        word_summary_df['mean_impact'].abs().sort_values(ascending=False).index
    )
    
    # Save overall summary with model type in filename
    model_type = "xgboost" if TEST_CONFIG['use_xgboost'] else "linear"
    word_summary_df.to_csv(f'perspective_api_word_analysis_{model_type}.csv', index=False)
    
    # Print separate summaries for positive and negative impacts
    print("\nTop 10 words increasing toxicity:")
    positive_impacts = word_summary_df[word_summary_df['mean_impact'] > 0].head(10)
    print(positive_impacts[['word', 'mean_impact', 'std_impact', 'frequency', 'impact_range']])
    
    print("\nTop 10 words decreasing toxicity:")
    negative_impacts = word_summary_df[word_summary_df['mean_impact'] < 0].head(10)
    print(negative_impacts[['word', 'mean_impact', 'std_impact', 'frequency', 'impact_range']])
    
    print(f"\nDetailed results saved to 'perspective_api_word_analysis_{model_type}.csv'")
    
    # Create visualizations
    visualize_explanation(word_summary_df)

print(f"\nAnalysis complete! Analyzed {len(all_explanations)} texts.")
print(f"Found {len(word_summary_df)} unique words that appeared at least {TEST_CONFIG['min_word_frequency']} times.")
