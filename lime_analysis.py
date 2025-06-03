import numpy as np
import pandas as pd
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt
from collections import defaultdict
import re
import string

# Read dataset containing text examples and their pre-computed toxicity scores
df = pd.read_csv('comments_with_scores.csv')

# Add stopwords list
STOPWORDS = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'over', 'after'}

def clean_word(word):
    """Clean and normalize individual words"""
    # Remove numbers and special characters
    word = re.sub(r'[^a-zA-Z\s]', '', word)
    # Convert to lowercase
    return word.lower().strip()

def preprocess_text(text):
    """Normalize text by removing stopwords, numbers, special chars and converting to lowercase"""
    # Split text into words and clean each word
    words = [clean_word(word) for word in text.split()]
    # Remove empty strings and stopwords
    words = [w for w in words if w and w not in STOPWORDS]
    return ' '.join(words)

def get_toxicity_scores(examples):
    """Get toxicity scores from cached CSV file"""
    # Create lookup dict for faster access
    text_scores = dict(zip(df['comment_text'].apply(preprocess_text), df['toxicity_score']))
    default_score = np.mean(list(text_scores.values()))  # Calculate mean once
    
    scores = []
    for text in examples:
        # Get original text by removing MASKED tokens
        orig_text = ' '.join([word for word in text.split() if word != 'MASKED'])
        orig_text = preprocess_text(orig_text)
        
        if orig_text in text_scores:
            score = text_scores[orig_text]
            if 'MASKED' in text:
                # Scale score based on remaining words
                orig_words = len(orig_text.split())
                remaining_words = len([w for w in text.split() if w != 'MASKED'])
                score = score * (remaining_words/orig_words)
            scores.append(score)
        else:
            scores.append(default_score)  # Use pre-calculated mean
            
    return np.array(scores).reshape(-1, 1)

def aggregate_word_importances(examples, scorer, num_features=20):
    """Use LIME to explain toxicity scores by analyzing word importance
    
    Process:
    1. LIME creates perturbations of each text by randomly masking words
    2. Get toxicity scores for all perturbations 
    3. Train local linear model to approximate how words affect toxicity
    4. Extract and aggregate word importance scores across examples
    
    Args:
        examples: List of text examples to analyze
        scorer: Function that returns toxicity scores for texts
        num_features: Number of top/bottom words to return
    Returns:
        top_positive, top_negative: Lists of (word, score) tuples
    """
    # Configure LIME explainer for faster processing
    explainer = LimeTextExplainer(
        class_names=['toxicity'],
        split_expression=lambda x: x.split(),  # Simple word tokenization
        bow=False,  # Faster processing without bag-of-words
        mask_string='MASKED'  # String used to mask words in perturbations
    )
    
    # Track importance scores for each normalized word
    word_scores = defaultdict(float)
    
    print(f"Analyzing {len(examples)} examples...")
    for i, (text, base_score) in enumerate(zip(examples, scorer(examples))):
        print(f"Processing example {i+1}/{len(examples)}...")
        
        # Generate LIME explanation for this example
        exp = explainer.explain_instance(
            text,
            scorer,
            num_features=100,  # Number of words to analyze per example
            num_samples=500,  # Number of perturbations to generate
            top_labels=1  # Only explain toxicity score
        )
        
        # Clean and aggregate scores for normalized words
        for word, score in exp.as_list(label=0):
            clean = clean_word(word)
            if clean and clean not in STOPWORDS:  # Only include non-empty, non-stopwords
                word_scores[clean] += score
    
    # Scale final scores to [-1,1] range for interpretability
    max_score = max(abs(score) for score in word_scores.values()) if word_scores else 1
    scaled_scores = {word: score/max_score for word, score in word_scores.items()}
    
    # Sort words by importance score
    sorted_words = sorted(scaled_scores.items(), key=lambda x: x[1], reverse=True)
    top_positive = sorted_words[:num_features]  # Words that increase toxicity
    top_negative = sorted_words[-num_features:][::-1]  # Words that decrease toxicity

    return top_positive, top_negative

def plot_word_importances(word_scores, title):
    """Plot horizontal bar chart of word importance scores and save to PNG."""
    if not word_scores:
        print(f"No words to plot for: {title}")
        return
        
    words = [str(word) for word, _ in word_scores]
    scores = [score for _, score in word_scores]
    
    plt.figure(figsize=(12, 8))
    bars = plt.barh(words, scores, color=['skyblue' if s > 0 else 'salmon' for s in scores])
    plt.xlabel('Aggregated LIME Score', fontsize=10)
    plt.ylabel('Words', fontsize=10)
    plt.title(title, fontsize=12, pad=20)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.gca().invert_yaxis()
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2,
                f'{width:.3f}', ha='left', va='center', fontsize=8)
    
    plt.tight_layout()
    
    # Create filename from title (remove spaces and special characters)
    filename = "".join(x for x in title if x.isalnum()) + '.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    print(f"Saved plot to {filename}")

def main():
    # Sample 999 random examples from non-empty texts
    mask = df['comment_text'].str.strip().str.len() > 0
    sampled_df = df[mask].sample(n=999, random_state=42)
    examples = sampled_df['comment_text'].tolist()
    
    # Get scores
    scores = get_toxicity_scores(examples).flatten().tolist()
    print("Toxicity scores:", scores)
    
    print("Aggregating word importances...")
    top_positive, top_negative = aggregate_word_importances(examples, get_toxicity_scores, num_features=20)

    print("Top 20 positively correlated words:", top_positive)
    print("Top 20 negatively correlated words:", top_negative)

    plot_word_importances(top_positive, "Top 20 Positively Correlated Words with Toxicity")
    plot_word_importances(top_negative, "Top 20 Negatively Correlated Words with Toxicity")

if __name__ == "__main__":
    main()
