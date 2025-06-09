import numpy as np
import pandas as pd
from lime.lime_text import LimeTextExplainer
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from collections import defaultdict
import re
import string

# Configuration variables
TEST_CONFIG = {
    'max_samples': 998,  # Number of texts to analyze
    'max_words': 50,    # Maximum number of words to analyze per text
    'min_word_frequency': 5,  # Minimum number of times a word must appear
    'top_n_words': 20,   # Number of top words to show in visualizations
    'use_xgboost': True,  # Flag to use XGBoost instead of linear regression
    'common_words': {  # Words to exclude from analysis
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'is', 'are', 'at', 'to', 'it', 'for', 
        'of', 'with', 'by', 'from', 'up', 'about', 'into', 'over', 'after', 'usertimothyhorrigan', 
        'too', 'this', 'their'
    }
}

# Read dataset containing text examples and their pre-computed toxicity scores
df = pd.read_csv('comments_with_scores.csv')

def clean_word(word):
    """Clean and normalize individual words"""
    # Remove special characters and punctuation
    word = re.sub(r'[^a-zA-Z0-9\s]', '', word)
    # Convert to lowercase
    return word.lower().strip()

def preprocess_text(text):
    """Normalize text by removing stopwords, numbers, special chars and converting to lowercase"""
    # Split text into words and clean each word
    words = [clean_word(word) for word in text.split()]
    # Remove empty strings and stopwords
    words = [w for w in words if w and w not in TEST_CONFIG['common_words']]
    return ' '.join(words)

class ToxicityModel:
    """Model wrapper for toxicity prediction that supports both XGBoost and Linear Regression"""
    def __init__(self, use_xgboost=True):
        self.vectorizer = TfidfVectorizer(
            max_features=2000, 
            ngram_range=(1,2),
            min_df=TEST_CONFIG['min_word_frequency']
        )
        self.use_xgboost = use_xgboost
        if use_xgboost:
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,  # L1 regularization for sparse features
                reg_lambda=1.0,  # L2 regularization
                random_state=42
            )
        else:
            self.model = LinearRegression()
        self.X_train_mean = None

    def fit(self, texts, scores):
        """Fit the model on text data and corresponding scores"""
        # Create feature matrix
        X = self.vectorizer.fit_transform(texts)
        self.X_train_mean = X.mean(axis=0)
        
        # Fit model
        self.model.fit(X, scores)
        return self
        
    def predict(self, texts):
        """Predict scores for texts - returns format expected by LIME"""
        X = self.vectorizer.transform(texts)
        predictions = self.model.predict(X)
        # Ensure predictions are in [0,1] range
        predictions = np.clip(predictions, 0, 1)
        # Reshape to 2D array with one column for LIME
        return predictions.reshape(-1, 1)

def calculate_lime_values(text: str, model: ToxicityModel) -> dict:
    """
    Calculate LIME values for words in the text
    """
    print(f"\nAnalyzing text: {text[:100]}...")
    
    # Get baseline score
    baseline_score = model.predict([text])[0][0]  # Note the [0][0] to get scalar value
    print(f"Baseline toxicity score: {baseline_score:.4f}")
    
    # Configure LIME explainer
    explainer = LimeTextExplainer(
        class_names=['toxicity'],
        split_expression=lambda x: x.split(),
        bow=False,
        mask_string='MASKED',
        random_state=42
    )
    
    # Get LIME explanation
    exp = explainer.explain_instance(
        text,
        model.predict,
        num_features=min(100, len(text.split())),
        num_samples=2000,
        labels=(0,)  # Specify we want explanation for the first (and only) class
    )
    
    try:
        # Get explanation for the first class (index 0)
        explanation_list = exp.as_list(label=0)
        
        # Clean and normalize words in the explanation
        cleaned_explanation = [(clean_word(word), value) for word, value in explanation_list]
        
        # Create results DataFrame
        results_df = pd.DataFrame(cleaned_explanation, columns=['word', 'lime_value'])
        results_df['contributes_to_toxicity'] = results_df['lime_value'] > 0
        results_df['absolute_impact'] = abs(results_df['lime_value'])
        
        # Sort by absolute impact
        results_df = results_df.reindex(results_df['absolute_impact'].sort_values(ascending=False).index)
        
        return {
            "baseline_score": baseline_score,
            "results": results_df,
            "text": text
        }
    except Exception as e:
        print(f"Error getting LIME explanation: {str(e)}")
        # Return empty results if explanation fails
        return {
            "baseline_score": baseline_score,
            "results": pd.DataFrame(columns=['word', 'lime_value', 'contributes_to_toxicity', 'absolute_impact']),
            "text": text
        }

def visualize_explanation(word_summary_df: pd.DataFrame, output_prefix: str = 'lime_api'):
    """Create visualizations for the overall LIME explanation"""
    # Add model type to output prefix
    model_type = "xgboost" if TEST_CONFIG['use_xgboost'] else "linear"
    output_prefix = f"{output_prefix}_{model_type}"
    
    # Filter out common words and short words
    word_summary_df = word_summary_df[
        (~word_summary_df['word'].isin(TEST_CONFIG['common_words'])) &
        (word_summary_df['word'].str.len() > 3)
    ]
    
    # Separate positive and negative impacts
    positive_impacts = word_summary_df[word_summary_df['mean_impact'] > 0].sort_values('mean_impact', ascending=False)
    negative_impacts = word_summary_df[word_summary_df['mean_impact'] < 0].sort_values('mean_impact')
    
    # 1. Top positive contributors
    plt.figure(figsize=(15, 10))
    top_positive = positive_impacts.head(TEST_CONFIG['top_n_words'])
    plt.barh(range(len(top_positive)), top_positive['mean_impact'], color='red')
    plt.yticks(range(len(top_positive)), top_positive['word'])
    plt.xlabel('Mean LIME Value')
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
    plt.xlabel('Mean LIME Value')
    plt.title('Top Words Decreasing Toxicity')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_negative_contributors.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Impact distribution
    plt.figure(figsize=(15, 10))
    plt.hist(word_summary_df['mean_impact'], bins=50, color='gray', alpha=0.7)
    plt.xlabel('Mean LIME Value')
    plt.ylabel('Number of Words')
    plt.title('Distribution of Word Impacts')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_impact_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("Starting LIME analysis...")
    print(f"Using {'XGBoost' if TEST_CONFIG['use_xgboost'] else 'Linear Regression'} model")
    
    # Sample texts from dataset
    mask = df['comment_text'].str.strip().str.len() > 0
    sampled_df = df[mask].sample(n=TEST_CONFIG['max_samples'], random_state=42)
    examples = sampled_df['comment_text'].tolist()
    
    # Initialize and fit model
    print("\nFitting model on training data...")
    model = ToxicityModel(use_xgboost=TEST_CONFIG['use_xgboost'])
    model.fit(examples, sampled_df['toxicity_score'].values)
    
    # Analyze examples
    all_explanations = []
    word_impacts = {}  # Dictionary to track word impacts across all texts
    
    for i, text in enumerate(examples):
        print(f"\n{'='*60}")
        print(f"Analyzing example {i+1}/{len(examples)}")
        print(f"{'='*60}")
        
        try:
            explanation = calculate_lime_values(text, model)
            
            # Update word impacts dictionary
            for _, row in explanation["results"].iterrows():
                word = row['word']
                if word not in word_impacts:
                    word_impacts[word] = []
                word_impacts[word].append({
                    'lime_value': row['lime_value'],
                    'text': text[:100] + "..."
                })
            
            all_explanations.append(explanation)
            
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
            
        lime_values = [imp['lime_value'] for imp in impacts]
        
        # Calculate statistics on the LIME values
        mean_impact = np.mean(lime_values)
        std_impact = np.std(lime_values)
        max_impact = np.max(lime_values)
        min_impact = np.min(lime_values)
        
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
        word_summary_df.to_csv(f'lime_api_word_analysis_{model_type}.csv', index=False)
        
        # Print separate summaries for positive and negative impacts
        print("\nTop 10 words increasing toxicity:")
        positive_impacts = word_summary_df[word_summary_df['mean_impact'] > 0].head(10)
        print(positive_impacts[['word', 'mean_impact', 'std_impact', 'frequency', 'impact_range']])
        
        print("\nTop 10 words decreasing toxicity:")
        negative_impacts = word_summary_df[word_summary_df['mean_impact'] < 0].head(10)
        print(negative_impacts[['word', 'mean_impact', 'std_impact', 'frequency', 'impact_range']])
        
        print(f"\nDetailed results saved to 'lime_api_word_analysis_{model_type}.csv'")
        
        # Create visualizations
        visualize_explanation(word_summary_df)
    
    print(f"\nAnalysis complete! Analyzed {len(all_explanations)} texts.")
    print(f"Found {len(word_summary_df)} unique words that appeared at least {TEST_CONFIG['min_word_frequency']} times.")

if __name__ == "__main__":
    main()
