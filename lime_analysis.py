import numpy as np
import pandas as pd
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt
from collections import defaultdict
from googleapiclient import discovery
import time
from googleapiclient.errors import HttpError

# Read dataset and initialize API client
df = pd.read_csv('balanced_train_1000.csv')
API_KEY = 'AIzaSyBf-jir2IV0S6DhflmQmpivAKwbGdSqL3s'
client = discovery.build(
    "commentanalyzer",
    "v1alpha1",
    developerKey=API_KEY,
    discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1"
)

class CachedToxicityScorer:
    def __init__(self, client):
        self.client = client
        self.cache = {}
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Minimum 1 second between requests
        
    def _wait_for_rate_limit(self):
        """Ensure we wait appropriate time between requests"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    def _get_score_with_retries(self, text, max_retries=5):
        """Make API call with exponential backoff"""
        retry_count = 0
        while retry_count < max_retries:
            try:
                self._wait_for_rate_limit()
                analyze_request = {
                    'comment': {'text': text},
                    'requestedAttributes': {'TOXICITY': {}},
                    'languages': ['en']
                }
                response = self.client.comments().analyze(body=analyze_request).execute()
                return response['attributeScores']['TOXICITY']['summaryScore']['value']
            except HttpError as e:
                if e.resp.status == 429:  # Rate limit error
                    retry_count += 1
                    sleep_time = (2 ** retry_count) + np.random.uniform(0, 1)
                    print(f"Rate limit hit, waiting {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)
                else:
                    print(f"HTTP Error {e.resp.status}: {str(e)}")
                    return np.nan
            except Exception as e:
                print(f"Unexpected error: {str(e)}")
                return np.nan
        print(f"Max retries ({max_retries}) exceeded for text: {text[:50]}...")
        return np.nan
        
    def get_scores(self, examples):
        scores = []
        for text in examples:
            if text in self.cache:
                scores.append(self.cache[text])
            else:
                score = self._get_score_with_retries(text)
                self.cache[text] = score
                scores.append(score)
                    
        return np.array(scores).reshape(-1, 1)

def aggregate_word_importances(examples, scorer, num_features=20):
    explainer = LimeTextExplainer()
    word_scores = defaultdict(float)

    # Get initial scores for full examples
    base_scores = scorer.get_scores(examples)
    
    for text, base_score in zip(examples, base_scores.flatten()):
        exp = explainer.explain_instance(
            text,
            scorer.get_scores,
            num_features=100
        )
        for word, score in exp.as_list():
            word_scores[word] += score

    # Sort words by their aggregated scores
    sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
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
    # Sample 5 random examples
    sampled_df = df.sample(n=5, random_state=42)
    examples = sampled_df['comment_text'].tolist()
    
    # Initialize cached scorer
    scorer = CachedToxicityScorer(client)
    
    # Get scores using cached scorer
    scores = scorer.get_scores(examples).flatten().tolist()
    print("Toxicity scores:", scores)
    
    # Filter out any failed scores
    filtered_examples = []
    filtered_scores = []
    for text, score in zip(examples, scores):
        if score is not None and not pd.isna(score):
            filtered_examples.append(text)
            filtered_scores.append(score)

    print("Aggregating word importances...")
    top_positive, top_negative = aggregate_word_importances(filtered_examples, scorer, num_features=20)

    print("Top 20 positively correlated words:", top_positive)
    print("Top 20 negatively correlated words:", top_negative)

    plot_word_importances(top_positive, "Top 20 Positively Correlated Words with Toxicity")
    plot_word_importances(top_negative, "Top 20 Negatively Correlated Words with Toxicity")

if __name__ == "__main__":
    main()
