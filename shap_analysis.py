import pandas as pd
import numpy as np
import shap # Corrected import: should be the shap library
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
df = df.head(10) # Limit to the first 20 comments for testing

# Initialize the Perspective API client
API_KEY = 'AIzaSyBf-jir2IV0S6DhflmQmpivAKwbGdSqL3s'
client = discovery.build(
    "commentanalyzer",
    "v1alpha1",
    developerKey=API_KEY,
    discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1"
)

def get_toxicity_score(text):
    """Get toxicity score from Perspective API for given text"""
    analyze_request = {
        'comment': {'text': text},
        'requestedAttributes': {'TOXICITY': {}},
        'languages': ['en']
    }
    max_retries = 3
    retry_delay_seconds = 61  # Wait a bit over a minute

    for attempt in range(max_retries):
        try:
            response = client.comments().analyze(body=analyze_request).execute()
            toxicity = response['attributeScores']['TOXICITY']['summaryScore']['value']
            return float(toxicity)
        except HttpError as e:
            if e.resp.status == 429: # Rate limit exceeded
                print(f"Rate limit exceeded for comment. Attempt {attempt + 1}/{max_retries}.")
                if attempt + 1 < max_retries:
                    print(f"Retrying in {retry_delay_seconds} seconds...")
                    time.sleep(retry_delay_seconds)
                else:
                    print("Max retries reached. Skipping this comment.")
                    return None
            else: # Other HTTP errors
                print(f"Error getting toxicity score (HttpError): {str(e)}")
                return None
        except Exception as e: # Other unexpected errors
            print(f"Error getting toxicity score (Exception): {str(e)}")
            return None
    return None # Should only be reached if max_retries is 0 or logic error

print("Starting SHAP analysis with Perspective API toxicity scores...")
print("This may take a while depending on API rate limits...")

# Get toxicity scores for all comments
print("Getting toxicity scores...")
# df['toxicity_score'] = df['comment_text'].apply(get_toxicity_score) # Old way

toxicity_scores_list = []
num_comments = len(df['comment_text'])
for i, comment_text in enumerate(df['comment_text']):
    # The get_toxicity_score function already prints detailed info about retries/errors.
    score = get_toxicity_score(comment_text)
    
    if score is not None:
        print(f"Processed comment {i + 1}/{num_comments}: Score = {score:.4f}")
    else:
        # get_toxicity_score would have printed an error message.
        print(f"Processed comment {i + 1}/{num_comments}: Failed to retrieve score (see error details above).")
        
    toxicity_scores_list.append(score)
    
    # Proactive delay: slightly more than 1 second to stay under 1 QPS.
    # Only sleep if there are more comments to process.
    if i < num_comments - 1:
        time.sleep(1.2)

df['toxicity_score'] = toxicity_scores_list

# Handle cases where toxicity score could not be fetched (is None)
# Option 1: Drop rows with None scores
df.dropna(subset=['toxicity_score'], inplace=True)
# Option 2: Fill with a default value (e.g., 0 or mean), but dropping is often safer for model training
# df['toxicity_score'].fillna(0, inplace=True) 

# Ensure there's data left to process
if df.empty:
    print("No data remaining after attempting to fetch toxicity scores. Exiting.")
    exit()

# Prepare feature matrix
X = df[['toxicity_score']].values
# Ensure y aligns with X after potential dropna
y = df['toxic'].values # Re-assign y after df might have changed

print(f"Shape of y after dropna and before model training: {y.shape}")
unique_y, counts_y = np.unique(y, return_counts=True)
print(f"Unique values in y after dropna: {unique_y}")
print(f"Counts of unique values in y after dropna: {counts_y}")

if len(unique_y) < 2:
    print("\nCritical Error: Target variable 'y' has less than two unique classes after filtering and processing.")
    print("This commonly happens if the data subset (e.g., from df.head()) is too small or homogenous.")
    print("SHAP analysis for binary classification requires at least two classes in the training data.")
    print("Please check your data subset or ensure 'df.head(10)' provides diverse samples for 'toxic' column.")
    exit()

# Use CountVectorizer instead of TF-IDF for more interpretable feature values
print("Creating word features...")
vectorizer = CountVectorizer(max_features=1000)
X_words = vectorizer.fit_transform(df['comment_text']).toarray()
feature_names = vectorizer.get_feature_names_out()

# Create combined feature matrix: [toxicity_score, word_features]
X_combined = np.hstack([X, X_words])
# Ensure X_combined is not empty before proceeding
if X_combined.shape[0] == 0:
    print("Feature matrix X_combined is empty. This might be due to all toxicity scores failing. Exiting.")
    exit()

feature_names_combined = ['toxicity_score'] + list(feature_names) # Ensure feature_names is a list

# Train model on combined features
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_combined, y)
print(f"Model fitted. model.classes_: {model.classes_}")

# Calculate SHAP values for all features
explainer = shap.TreeExplainer(model) 
shap_values = explainer.shap_values(X_combined) 

print(f"Type of global shap_values: {type(shap_values)}")
# For RandomForestClassifier, shap_values can be a list of arrays (one per class)
# or a single 3D array (n_samples, n_features, n_classes)
is_shap_values_3d_array = isinstance(shap_values, np.ndarray) and shap_values.ndim == 3 and shap_values.shape[-1] == len(model.classes_)

if isinstance(shap_values, list) and len(shap_values) == len(model.classes_):
    print(f"Length of global shap_values list: {len(shap_values)}")
    if len(shap_values) > 0:
        print(f"  Type of first element in global shap_values: {type(shap_values[0])}")
        if hasattr(shap_values[0], 'shape'):
            print(f"  Shape of first element in global shap_values: {shap_values[0].shape}")
    if len(shap_values) > 1:
        print(f"  Type of second element in global shap_values: {type(shap_values[1])}")
        if hasattr(shap_values[1], 'shape'):
            print(f"  Shape of second element in global shap_values: {shap_values[1].shape}")
elif is_shap_values_3d_array:
    print(f"Global shap_values is a 3D NumPy array with shape: {shap_values.shape}")
else:
    print(f"Global shap_values is in an unexpected format. Type: {type(shap_values)}")
    if hasattr(shap_values, 'shape'):
        print(f"  Shape: {shap_values.shape}")

# Add function to explain individual predictions
def explain_prediction(text, model, explainer, vectorizer):
    """Get SHAP explanation for a single comment, focusing on the 'toxic' class (class 1)."""
    # Get toxicity score
    toxicity = get_toxicity_score(text)
    if toxicity is None: # Handle case where toxicity score couldn't be fetched
        print(f"Could not get toxicity score for explaining: {text[:100]}...")
        return pd.DataFrame(columns=['token', 'shap_value', 'increases_toxicity'])

    # Vectorize text
    text_features = vectorizer.transform([text]).toarray()
    # Explicitly create a 2D array for the toxicity score
    toxicity_feature_array = np.array([toxicity]).reshape(1, 1)
    features = np.hstack((toxicity_feature_array, text_features))
    
    # Calculate SHAP values for this instance
    # explainer.shap_values(features) returns a list [class0_shap_values, class1_shap_values]
    # Each element is 2D, e.g., (1, num_features) for a single instance.
    shap_output_for_instance = explainer.shap_values(features)

    print(f"\n--- Debugging explain_prediction for text: {text[:30]}... ---")
    print(f"Type of shap_output_for_instance: {type(shap_output_for_instance)}")
    
    is_instance_shap_3d_array = isinstance(shap_output_for_instance, np.ndarray) and \
                                shap_output_for_instance.ndim == 3 and \
                                shap_output_for_instance.shape[0] == 1 and \
                                shap_output_for_instance.shape[-1] == len(model.classes_)
    
    is_instance_shap_list = isinstance(shap_output_for_instance, list) and \
                            len(shap_output_for_instance) == len(model.classes_)

    if is_instance_shap_list:
        print(f"Length of shap_output_for_instance list: {len(shap_output_for_instance)}")
        # ... (existing debug prints for list format)
    elif is_instance_shap_3d_array:
        print(f"Shape of shap_output_for_instance (3D NumPy array): {shap_output_for_instance.shape}")
    else:
        print(f"shap_output_for_instance is in an unexpected format for individual prediction.")
        if hasattr(shap_output_for_instance, 'shape'):
            print(f"  Shape: {shap_output_for_instance.shape}")
    print("--- End debugging explain_prediction ---")

    # Determine how to extract SHAP values for class 1
    if is_instance_shap_list:
        # Expected: list of 2 arrays, each (1, n_features) for a single instance
        if len(shap_output_for_instance) == 2 and shap_output_for_instance[1].ndim == 2 and shap_output_for_instance[1].shape[0] == 1:
            instance_shap_values_class1_1d = shap_output_for_instance[1][0]
        else:
            print("Warning: SHAP output list for instance is not as expected. Cannot proceed.")
            return pd.DataFrame(columns=['token', 'shap_value', 'increases_toxicity'])
    elif is_instance_shap_3d_array:
        # Expected: (1, n_features, n_classes)
        instance_shap_values_class1_1d = shap_output_for_instance[0, :, 1] # For class 1
    else:
        print("Warning: SHAP output for instance is not in a recognized format (list of 2 arrays or 3D array). Cannot proceed.")
        return pd.DataFrame(columns=['token', 'shap_value', 'increases_toxicity'])
    
    # --- Construct list of tokens and their SHAP contributions ---
    token_contributions_list = []

    # 1. Contribution of 'toxicity_score'
    token_contributions_list.append({
        'token': 'toxicity_score', # Name of the first feature
        'shap_value': instance_shap_values_class1_1d[0]
    })

    # 2. Contributions of word features present in the text
    vocab = vectorizer.get_feature_names_out()
    word_to_vocab_idx = {word: i for i, word in enumerate(vocab)}
    
    # Simple tokenization of the input text (can be made more robust if needed)
    # Consider using vectorizer.build_analyzer()(text) for perfect alignment with vectorizer's tokenization
    text_tokens = set(text.lower().split()) 

    for token_in_text in text_tokens:
        if token_in_text in word_to_vocab_idx:
            vocab_idx = word_to_vocab_idx[token_in_text]
            # SHAP values for word features start at index 1 of instance_shap_values_class1_1d
            shap_value_for_token = instance_shap_values_class1_1d[1 + vocab_idx]
            token_contributions_list.append({
                'token': token_in_text,
                'shap_value': shap_value_for_token
            })
            
    if not token_contributions_list:
        # This case should ideally not happen if toxicity_score is always present
        # or if the text is not empty and contains some vocab words.
        print(f"No contributing tokens found for text: {text[:100]}...")
        return pd.DataFrame(columns=['token', 'shap_value', 'increases_toxicity'])

    contributions_df = pd.DataFrame(token_contributions_list)
    contributions_df['increases_toxicity'] = contributions_df['shap_value'] > 0
    
    return contributions_df.sort_values('shap_value', key=abs, ascending=False)

# Example explanations for a few comments
print("\nAnalyzing individual examples...")
# Adjust number of examples if df is smaller than 5
num_example_texts = min(5, len(df))
example_texts = df['comment_text'].iloc[:num_example_texts].tolist()
for text in example_texts:
    print("\nAnalyzing comment:", text[:100], "...")
    explanation = explain_prediction(text, model, explainer, vectorizer)
    if not explanation.empty: # Check if explanation was successful
        print("\nTop contributing words:")
        print(explanation.head())
    
        # Create force plot for this prediction
        plt.figure(figsize=(10,2))
        
        # For force plot, use expected value for class 1 (toxic)
        # explainer.expected_value is a list [E[f(x)]_class0, E[f(x)]_class1]
        if isinstance(explainer.expected_value, (list, np.ndarray)) and len(explainer.expected_value) == 2:
            expected_value_for_plot = explainer.expected_value[1]
        else:
            # Fallback if expected_value is not as expected (e.g. single value for regression)
            expected_value_for_plot = explainer.expected_value 


        if not explanation['shap_value'].empty and not explanation['token'].empty:
            shap.force_plot( 
                expected_value_for_plot, 
                explanation['shap_value'].values,
                explanation['token'].values,
                show=False,
                matplotlib=True
            )
            plt.title("SHAP Force Plot - Word Contributions")
            plt.tight_layout()
            plt.savefig(f'force_plot_{hash(text)}.png')
        else:
            print(f"Not enough data to create force plot for: {text[:100]}...")
        plt.close()
    else:
        print(f"Could not generate explanation for: {text[:100]}...")
    time.sleep(1.05) # Add delay here too if explaining many examples

# Create plots
print("\nGenerating visualizations...")
plt.figure(figsize=(15,10))

# Plot 1: Overall feature importance
# For binary classification, shap_values from TreeExplainer is a list [shap_values_class_0, shap_values_class_1]
# We usually plot for the positive class (class 1)

shap_values_for_summary_plot = None
if isinstance(shap_values, list) and len(shap_values) == 2:
    shap_values_for_summary_plot = shap_values[1]
elif is_shap_values_3d_array: # global shap_values is 3D array (n_samples, n_features, n_classes)
    shap_values_for_summary_plot = shap_values[:, :, 1] # SHAP values for class 1

if shap_values_for_summary_plot is not None:
    if X_combined.shape[0] == shap_values_for_summary_plot.shape[0] and \
       X_combined.shape[1] == shap_values_for_summary_plot.shape[1]:
        shap.summary_plot(shap_values_for_summary_plot, X_combined, feature_names=feature_names_combined,
                         show=False, plot_size=(10,6))
        plt.title("SHAP Summary Plot: Impact of Toxicity Score and Words (for Toxic Class)")
        plt.tight_layout()
        plt.savefig('shap_summary_all.png')
        plt.close()
    else:
        print(f"Warning: Mismatch between X_combined shape {X_combined.shape} and shap_values_for_summary_plot shape {shap_values_for_summary_plot.shape} for overall summary plot.")
else:
    print("Skipping overall SHAP summary plot because shap_values is not in an expected format.")
    # ... (existing debug prints for unexpected format)

# Plot 2: Top 20 most important words
plt.figure(figsize=(12,8))
word_shap_values_for_plot = None
if isinstance(shap_values, list) and len(shap_values) == 2:
    word_shap_values_for_plot = shap_values[1][:, 1:] # class 1, skip toxicity_score feature
elif is_shap_values_3d_array:
    word_shap_values_for_plot = shap_values[:, 1:, 1] # class 1, skip toxicity_score feature

if word_shap_values_for_plot is not None:
    if X_words.shape[1] > 0:
        if X_words.shape[0] == word_shap_values_for_plot.shape[0] and \
           X_words.shape[1] == word_shap_values_for_plot.shape[1]:
            shap.summary_plot(word_shap_values_for_plot, X_words, 
                             feature_names=list(feature_names), 
                             max_display=20,
                             show=False, plot_type="bar") 
            plt.title("Top 20 Most Influential Words (for Toxic Class)")
            plt.tight_layout()
            plt.savefig('shap_word_importance.png')
            plt.close() 
        else:
            print(f"Warning: Mismatch between X_words shape {X_words.shape} and word_shap_values_for_plot shape {word_shap_values_for_plot.shape} for word importance plot.")
    else:
        print("Skipping word importance plot as there are no word features (X_words is empty).")
else:
    print("Skipping word importance plot because shap_values is not in an expected format for binary classification.")

# Save detailed word-level analysis
word_results_shap_values = None
if isinstance(shap_values, list) and len(shap_values) == 2:
    word_results_shap_values = shap_values[1][:, 1:]
elif is_shap_values_3d_array:
    word_results_shap_values = shap_values[:, 1:, 1]

if word_results_shap_values is not None and X_words.shape[1] > 0:
    if X_words.shape[0] == word_results_shap_values.shape[0] and \
       X_words.shape[1] == word_results_shap_values.shape[1]:
        word_results = pd.DataFrame({
            'word': list(feature_names), 
            'average_impact': np.abs(word_results_shap_values).mean(0), # Impact on toxic class
            'direction': np.where(word_results_shap_values.mean(0) > 0, 'More Toxic', 'Less Toxic') # Direction for toxic class
        })
        word_results = word_results.sort_values('average_impact', ascending=False)
        word_results.to_csv('word_level_analysis.csv', index=False)

        print("\nWord-Level Analysis Summary:")
        print(f"Top 10 most influential words and their impact:")
        print(word_results.head(10))
    else:
        print("Skipping word-level analysis CSV due to shape mismatch.")
else:
    print("Skipping word-level analysis CSV as shap_values not in expected format or no word features.")
