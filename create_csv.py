import pandas as pd
import numpy as np

def create_balanced_sample(input_file='train.csv', output_file='balanced_train_1000.csv', sample_size=500):
    # Read the original CSV
    df = pd.read_csv(input_file)
    
    # Split into toxic and non-toxic
    toxic = df[df['toxic'] == 1]
    non_toxic = df[df['toxic'] == 0]
    
    # Sample equal amounts from each
    toxic_sample = toxic.sample(n=sample_size, random_state=42)
    non_toxic_sample = non_toxic.sample(n=sample_size, random_state=42)
    
    # Combine and shuffle
    balanced_df = pd.concat([toxic_sample, non_toxic_sample])
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save to new CSV
    balanced_df.to_csv(output_file, index=False)
    print(f"Created balanced dataset with {len(balanced_df)} rows in {output_file}")

if __name__ == "__main__":
    create_balanced_sample()
