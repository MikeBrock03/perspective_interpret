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
from tqdm import tqdm
from functools import lru_cache

# Configuration variables for testing
TEST_CONFIG = {
    'max_samples': 5,  # Number of texts to analyze (500 toxic + 500 non-toxic)
    'max_words': 50,    # Maximum number of words to analyze per text
    'rate_limit_delay': 2.0,  # Increased delay between API calls in seconds
    'min_word_frequency': 5,  # Minimum number of times a word must appear to be included in analysis
    'top_n_words': 20,   # Number of top words to show in concise visualizations
    'detailed_top_n_words': 100,  # Number of words to show in detailed visualizations
    'max_word_length': 3,  # Minimum word length to consider
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

@lru_cache(maxsize=1000)
def get_toxicity_score(text: str) -> float:
    """Get toxicity score from Perspective API for given text with caching"""
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
            time.sleep(TEST_CONFIG['rate_limit_delay'])  # Add delay after each successful call
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

class SimpleTokenizer:
    """A simple tokenizer class that matches SHAP's expected format"""
    def __init__(self):
        self.vocab = {}
        self.reverse_vocab = {}
        self.next_id = 0
    
    def __call__(self, text: str) -> dict:
        """Tokenize text and return in format expected by SHAP"""
        tokens = tokenize_text(text)
        input_ids = []
        
        for token in tokens:
            if token not in self.vocab:
                self.vocab[token] = self.next_id
                self.reverse_vocab[self.next_id] = token
                self.next_id += 1
            input_ids.append(self.vocab[token])
        
        return {
            "input_ids": input_ids,
            "tokens": tokens
        }
    
    def decode(self, ids: List[int]) -> str:
        """Convert token IDs back to text"""
        return " ".join(self.reverse_vocab[id] for id in ids)

def explain_perspective_directly(text: str) -> dict:
    """Use SHAP's masking to explain the Perspective API directly"""
    def perspective_predict(texts):
        return np.array([get_toxicity_score(text) for text in texts])
    
    # Create a text masker that will handle word-level masking
    tokenizer = SimpleTokenizer()
    masker = shap.maskers.Text(tokenizer=tokenizer)
    
    # Calculate minimum required evaluations based on number of words
    num_words = len(tokenize_text(text))
    min_evals = 2 * num_words + 1
    
    # Use a smaller number of evaluations to reduce API calls
    max_evals = min(min_evals, 50)  # Cap at 50 evaluations
    
    # Create the explainer with appropriate number of evaluations
    explainer = shap.Explainer(
        perspective_predict, 
        masker,
        max_evals=max_evals,  # Use capped number of evaluations
        algorithm='permutation'  # Use permutation algorithm which is faster
    )
    
    # Get the explanation
    explanation = explainer([text])
    
    # Extract the SHAP values and words
    shap_values = explanation.values[0]
    words = explanation.data[0]
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'word': words,
        'shap_value': shap_values,
        'contributes_to_toxicity': shap_values > 0,
        'absolute_impact': np.abs(shap_values)
    })
    
    # Sort by absolute impact
    results_df = results_df.reindex(results_df['absolute_impact'].sort_values(ascending=False).index)
    
    return {
        "baseline_score": perspective_predict([text])[0],
        "results": results_df,
        "text": text
    }

# Main analysis
print("Starting SHAP analysis of Perspective API...")
print("This will make API calls and take time due to rate limits...")

# Analyze example texts
sample_texts = df['comment_text'].tolist()

all_explanations = []
word_impacts = {}  # Dictionary to track word impacts across all texts

for i, text in enumerate(tqdm(sample_texts, desc="Analyzing texts")):
    print(f"\n{'='*60}")
    print(f"Analyzing example {i+1}/{len(sample_texts)}")
    print(f"{'='*60}")
    
    try:
        explanation = explain_perspective_directly(text)
        
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
