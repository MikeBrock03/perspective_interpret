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

# Configuration variables for testing
TEST_CONFIG = {
    'max_samples': 1000,  # Number of texts to analyze (500 toxic + 500 non-toxic)
    'max_words': 50,    # Maximum number of words to analyze per text
    'rate_limit_delay': 1.2,  # Delay between API calls in seconds
    'min_word_frequency': 5,  # Minimum number of times a word must appear to be included in analysis
    'top_n_words': 20,   # Number of top words to show in concise visualizations
    'detailed_top_n_words': 100,  # Number of words to show in detailed visualizations
    'max_word_length': 3,  # Minimum word length to consider
    'common_words': {  # Words to exclude from analysis
        'the', 'and', 'that', 'have', 'for', 'not', 'this', 'but', 'with', 'you', 'from', 'they',
        'say', 'will', 'one', 'all', 'would', 'there', 'their', 'what', 'about', 'which', 'when',
        'make', 'like', 'time', 'just', 'know', 'people', 'into', 'year', 'good', 'some', 'could',
        'them', 'see', 'other', 'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over',
        'think', 'also', 'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well',
        'way', 'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day', 'most', 'us',
        'been', 'much', 'using', 'their', 'both', 'always', 'think', 'point', 'user', 'made'
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
    """Simple linear model wrapper for Perspective API"""
    def __init__(self):
        self.vectorizer = CountVectorizer()
        self.coefficients = None
        self.intercept = None
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
        
        # Fit a simple linear regression
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X, scores)
        
        self.coefficients = model.coef_
        self.intercept = model.intercept_
        self.X_train_mean = X.mean(axis=0)
        
        # Validate intercept is reasonable for 0-1 range
        if self.intercept < 0 or self.intercept > 1:
            print(f"Warning: Model intercept {self.intercept:.4f} outside expected range [0,1]")
            # Adjust intercept to be within [0,1]
            self.intercept = max(0, min(1, self.intercept))
        
        return self
    
    def predict(self, texts: List[str]) -> np.ndarray:
        """Predict toxicity scores for new texts"""
        X = self.vectorizer.transform(texts).toarray()
        predictions = self.intercept + np.dot(X, self.coefficients)
        # Ensure predictions are in [0,1] range
        return np.clip(predictions, 0, 1)

def calculate_shap_values(text: str, model: PerspectiveModel) -> dict:
    """
    Calculate SHAP values for words in the text using both manual calculation
    and SHAP's LinearExplainer
    """
    print(f"\nAnalyzing text: {text[:100]}...")
    
    # Transform the text to feature vector
    X = model.vectorizer.transform([text]).toarray()[0]
    
    # Get baseline score
    baseline_score = model.predict([text])[0]
    print(f"Baseline toxicity score: {baseline_score:.4f}")
    
    # Calculate base value
    base_value = model.intercept + np.sum(model.coefficients * model.X_train_mean)
    # Ensure base value is in [0,1] range
    base_value = max(0, min(1, base_value))
    print(f"Base value: {base_value:.4f}")
    
    # Get unique words in vocabulary
    words = model.vectorizer.get_feature_names_out()
    word_indices = {word: idx for idx, word in enumerate(words)}
    
    # Select words to analyze
    text_words = set(tokenize_text(text))
    selected_words = [word for word in text_words if word in word_indices][:TEST_CONFIG['max_words']]
    selected_indices = [word_indices[word] for word in selected_words]
    
    # Calculate manual SHAP values
    manual_shap_values = []
    for idx in selected_indices:
        shap_value = model.coefficients[idx] * (X[idx] - model.X_train_mean[idx])
        manual_shap_values.append(shap_value)
    
    # Calculate SHAP values using LinearExplainer
    masker = (model.X_train_mean, np.zeros((len(words), len(words))))
    explainer = shap.LinearExplainer((model.coefficients, model.intercept), masker)
    shap_values_full = explainer.shap_values(X)
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'word': selected_words,
        'shap_value': manual_shap_values,
        'contributes_to_toxicity': [value > 0 for value in manual_shap_values],
        'absolute_impact': [abs(value) for value in manual_shap_values]
    })
    
    # Sort by absolute impact
    results_df = results_df.reindex(results_df['absolute_impact'].sort_values(ascending=False).index)
    
    return {
        "baseline_score": baseline_score,
        "base_value": base_value,
        "results": results_df,
        "shap_values_full": shap_values_full,
        "text": text
    }

def visualize_explanation(word_summary_df: pd.DataFrame, output_prefix: str = 'perspective_api'):
    """Create visualizations for the overall SHAP explanation"""
    # Filter out common words and short words
    word_summary_df = word_summary_df[
        (~word_summary_df['word'].isin(TEST_CONFIG['common_words'])) &
        (word_summary_df['word'].str.len() > TEST_CONFIG['max_word_length'])
    ]
    
    # Separate positive and negative impacts
    positive_impacts = word_summary_df[word_summary_df['mean_impact'] > 0].sort_values('mean_impact', ascending=False)
    negative_impacts = word_summary_df[word_summary_df['mean_impact'] < 0].sort_values('mean_impact')
    
    # Create figure with subplots for concise view
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # 1. Top positive contributors (concise)
    top_positive = positive_impacts.head(TEST_CONFIG['top_n_words'])
    ax1.barh(range(len(top_positive)), top_positive['mean_impact'], color='red')
    ax1.set_yticks(range(len(top_positive)))
    ax1.set_yticklabels(top_positive['word'])
    ax1.set_xlabel('Mean SHAP Value')
    ax1.set_title('Top Words Increasing Toxicity')
    ax1.grid(axis='x', alpha=0.3)
    
    # 2. Top negative contributors (concise)
    top_negative = negative_impacts.head(TEST_CONFIG['top_n_words'])
    ax2.barh(range(len(top_negative)), top_negative['mean_impact'], color='blue')
    ax2.set_yticks(range(len(top_negative)))
    ax2.set_yticklabels(top_negative['word'])
    ax2.set_xlabel('Mean SHAP Value')
    ax2.set_title('Top Words Decreasing Toxicity')
    ax2.grid(axis='x', alpha=0.3)
    
    # 3. Impact vs Frequency scatter plot
    scatter = ax3.scatter(word_summary_df['frequency'], word_summary_df['mean_impact'],
                         alpha=0.5, c=['red' if x > 0 else 'blue' for x in word_summary_df['mean_impact']],
                         s=word_summary_df['std_impact'] * 100)  # Size based on standard deviation
    ax3.set_xlabel('Word Frequency')
    ax3.set_ylabel('Mean SHAP Value')
    ax3.set_title('Word Frequency vs Impact on Toxicity\n(Size indicates impact variability)')
    ax3.grid(True, alpha=0.3)
    
    # Add labels for the most impactful words
    for _, row in word_summary_df.nlargest(10, 'mean_impact').iterrows():
        ax3.annotate(row['word'], 
                    (row['frequency'], row['mean_impact']),
                    xytext=(5, 5), textcoords='offset points')
    
    # 4. Impact distribution
    ax4.hist(word_summary_df['mean_impact'], bins=50, color='gray', alpha=0.7)
    ax4.set_xlabel('Mean SHAP Value')
    ax4.set_ylabel('Number of Words')
    ax4.set_title('Distribution of Word Impacts')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_concise_analysis.png', dpi=300, bbox_inches='tight')
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
    print(f"- {output_prefix}_concise_analysis.png (4-panel analysis)")
    print(f"- {output_prefix}_positive_impacts_detailed.png (detailed positive contributors)")
    print(f"- {output_prefix}_negative_impacts_detailed.png (detailed negative contributors)")

# Main analysis
print("Starting SHAP analysis of Perspective API...")
print("This will make API calls and take time due to rate limits...")

# First, fit the model on a subset of the data
print("\nFitting model on training data...")
model = PerspectiveModel()
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
    if len(impacts) >= TEST_CONFIG['min_word_frequency']:
        shap_values = [imp['shap_value'] for imp in impacts]
        word_summary.append({
            'word': word,
            'mean_impact': np.mean(shap_values),
            'std_impact': np.std(shap_values),
            'frequency': len(impacts),
            'max_impact': max(shap_values),
            'min_impact': min(shap_values),
            'impact_range': max(shap_values) - min(shap_values)
        })

# Convert to DataFrame and sort by absolute mean impact
word_summary_df = pd.DataFrame(word_summary)
if not word_summary_df.empty:
    # Sort by absolute impact
    word_summary_df = word_summary_df.reindex(
        word_summary_df['mean_impact'].abs().sort_values(ascending=False).index
    )
    
    # Save overall summary
    word_summary_df.to_csv('perspective_api_word_analysis.csv', index=False)
    
    # Print separate summaries for positive and negative impacts
    print("\nTop 10 words increasing toxicity:")
    positive_impacts = word_summary_df[word_summary_df['mean_impact'] > 0].head(10)
    print(positive_impacts[['word', 'mean_impact', 'std_impact', 'frequency', 'impact_range']])
    
    print("\nTop 10 words decreasing toxicity:")
    negative_impacts = word_summary_df[word_summary_df['mean_impact'] < 0].head(10)
    print(negative_impacts[['word', 'mean_impact', 'std_impact', 'frequency', 'impact_range']])
    
    print("\nDetailed results saved to 'perspective_api_word_analysis.csv'")
    
    # Create visualizations
    visualize_explanation(word_summary_df)

print(f"\nAnalysis complete! Analyzed {len(all_explanations)} texts.")
print(f"Found {len(word_summary_df)} unique words that appeared at least {TEST_CONFIG['min_word_frequency']} times.")
