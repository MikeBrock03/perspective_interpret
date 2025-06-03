import pandas as pd
import time
from googleapiclient import discovery
from googleapiclient.errors import HttpError
import numpy as np

class ToxicityScorer:
    def __init__(self):
        self.API_KEY = 'AIzaSyBf-jir2IV0S6DhflmQmpivAKwbGdSqL3s'
        self.client = discovery.build(
            "commentanalyzer",
            "v1alpha1",
            developerKey=self.API_KEY,
            discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1"
        )
        self.last_request_time = 0
        self.min_request_interval = 2.0  # Increase to 2 seconds between requests
        self.max_retries = 3
    
    def get_score(self, text):
        # Rate limiting
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        
        # Try with exponential backoff
        for attempt in range(self.max_retries):
            try:
                analyze_request = {
                    'comment': {'text': text},
                    'requestedAttributes': {'TOXICITY': {}},
                    'languages': ['en']
                }
                response = self.client.comments().analyze(body=analyze_request).execute()
                score = response['attributeScores']['TOXICITY']['summaryScore']['value']
                self.last_request_time = time.time()
                return score
                
            except HttpError as e:
                if e.resp.status == 429:  # Rate limit error
                    wait_time = (2 ** attempt) + np.random.uniform(0, 1)
                    print(f"Rate limit hit, waiting {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"HTTP Error processing text: {str(e)[:100]}")
                    return np.nan
            except Exception as e:
                print(f"Error processing text: {str(e)[:100]}")
                return np.nan
        
        print("Max retries exceeded, skipping example")
        return np.nan

def main():
    # Read input CSV
    print("Reading input CSV...")
    df = pd.read_csv('balanced_train_1000.csv')
    
    # Initialize scorer
    scorer = ToxicityScorer()
    
    # Get scores for each text
    print("Getting toxicity scores...")
    scores = []
    total = len(df)
    
    for i, text in enumerate(df['comment_text']):
        score = scorer.get_score(text)
        scores.append(score)
        
        # Print progress
        if (i + 1) % 10 == 0:
            print(f"Processed {i+1}/{total} examples...")
    
    # Add scores to dataframe and save
    df['toxicity_score'] = scores
    output_file = 'comments_with_scores.csv'
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
