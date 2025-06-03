import numpy as np
import pandas as pd
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt
from collections import defaultdict

# Read dataset with cached scores
df = pd.read_csv('comments_with_scores.csv')

def preprocess_text(text):
    """Normalize text by removing extra spaces"""
    return ' '.join(text.split())

def get_toxicity_scores(examples):
    """Get toxicity scores from cached CSV file"""
    # Create lookup dict for faster access
    text_scores = dict(zip(df['comment_text'].apply(preprocess_text), df['toxicity_score']))
    
    scores = []
    for text in examples:
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
            scores.append(text_scores.values().mean())
    
    return np.array(scores).reshape(-1, 1)  # Return as column vector

def aggregate_word_importances(examples, scorer, num_features=20):
    # Configure LIME for faster processing
    explainer = LimeTextExplainer(
        class_names=['toxicity'],
        split_expression=lambda x: x.split(),
        bow=False,  # Faster processing
        mask_string='MASKED'
    )
    word_scores = defaultdict(float)
    
    print(f"Analyzing {len(examples)} examples...")
    for i, (text, base_score) in enumerate(zip(examples, scorer(examples))):
        print(f"Processing example {i+1}/{len(examples)}...")
        exp = explainer.explain_instance(
            text,
            scorer,
            num_features=50,  # Reduced from 100
            num_samples=500,  # Reduced sample size
            top_labels=1
        )
        # Normalize scores relative to base toxicity
        for word, score in exp.as_list(label=0):
            word_scores[word] += score / max(abs(base_score), 0.001)
    
    # Sort and scale scores for better interpretability  
    max_score = max(abs(score) for score in word_scores.values())
    scaled_scores = {word: score/max_score for word, score in word_scores.items()}
    
    sorted_words = sorted(scaled_scores.items(), key=lambda x: x[1], reverse=True)
    top_positive = sorted_words[:num_features]
    top_negative = sorted_words[-num_features:][::-1]  # Reverse for most negative first

    return top_positive, top_negative

def plot_word_importances(word_scores, title):
    """
    Plot horizontal bar chart of word importance scores.
    
    Args:
        word_scores (list): List of (word, score) tuples
        title (str): Title for the plot
    """
    if not word_scores:
        print(f"No words to plot for: {title}")
        return
        
    words, scores = zip(*word_scores)
    plt.figure(figsize=(12, 8))
    bars = plt.barh(words, scores, color=['skyblue' if s > 0 else 'salmon' for s in scores])
    plt.xlabel('Aggregated LIME Score', fontsize=10)
    plt.ylabel('Words', fontsize=10)
    plt.title(title, fontsize=12, pad=20)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.gca().invert_yaxis()
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2,
                f'{width:.3f}', ha='left', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.show()


def main():
    # Sample 5 random examples from non-empty texts
    mask = df['comment_text'].str.strip().str.len() > 0
    sampled_df = df[mask].sample(n=5, random_state=42)
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
