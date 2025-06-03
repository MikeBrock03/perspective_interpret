import pandas as pd
import numpy as np
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
    """Get toxicity score for a single text"""
    try:
        analyze_request = {
            'comment': {'text': text},
            'requestedAttributes': {'TOXICITY': {}},
            'languages': ['en'],  # Specify English language
        }
        response = client.comments().analyze(body=analyze_request).execute()
        return response['attributeScores']['TOXICITY']['summaryScore']['value']
    except HttpError as e:
        print(f"API Error for text: {text[:50]}... Error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

def create_toxicity_cache():
    """Process all examples and save toxicity scores"""
    try:
        existing_df = pd.read_csv('toxicity_scores_cache.csv')
        print(f"Loaded existing cache with {len(existing_df)} entries")
        print(f"Found {existing_df['toxicity_score'].notna().sum()} valid scores")
        
        # Keep only valid scores
        valid_scores = existing_df[existing_df['toxicity_score'].notna()]
        processed_texts = set(valid_scores['text'])
        results = valid_scores.to_dict('records')
        print(f"Kept {len(results)} valid cached scores")
    except FileNotFoundError:
        results = []
        processed_texts = set()
    
    texts = df['comment_text'].tolist()
    texts_to_process = [t for t in texts if t not in processed_texts]
    
    print(f"Need to process {len(texts_to_process)} examples...")
    try:
        for i, text in enumerate(texts_to_process):
            if i % 10 == 0:
                print(f"Processing example {i}/{len(texts_to_process)}")
            
            time.sleep(1)
            score = get_toxicity_score(text)
            if score is not None:  # Only add valid scores
                results.append({'text': text, 'toxicity_score': score})
            
            # Save intermediate results every 50 examples
            if i % 50 == 0 and results:
                pd.DataFrame(results).to_csv('toxicity_scores_cache.csv', index=False)
    
    except KeyboardInterrupt:
        print("\nSaving progress before exit...")
    finally:
        if results:
            results_df = pd.DataFrame(results)
            results_df.to_csv('toxicity_scores_cache.csv', index=False)
            print(f"Saved {len(results_df)} scores to toxicity_scores_cache.csv")
            print(f"Number of valid scores: {results_df['toxicity_score'].notna().sum()}")

if __name__ == "__main__":
    create_toxicity_cache()
