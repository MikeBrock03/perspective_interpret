import numpy as np
import pandas as pd
from lime.lime_text import LimeTextExplainer
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from collections import defaultdict
import re
import string

# Read dataset containing text examples and their pre-computed toxicity scores
df = pd.read_csv('comments_with_scores.csv')

# Add stopwords list
STOPWORDS = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'is', 'are', 'at', 'to', 'it', 'for', 
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'over', 'after', 'usertimothyhorrigan', 'too', 'this', 'their'}

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
    """Get toxicity scores using word-level toxicity analysis with non-linear weighting"""
    # Create lookup dict for faster access
    text_scores = dict(zip(df['comment_text'].apply(preprocess_text), df['toxicity_score']))
    
    # Build word-level toxicity dictionary
    word_toxicity = defaultdict(list)
    for text, score in text_scores.items():
        words = text.split()
        for word in words:
            word_toxicity[word].append(score)
    
    # Calculate average toxicity per word
    word_toxicity = {word: np.mean(scores) for word, scores in word_toxicity.items()}
    max_word_toxicity = max(word_toxicity.values())
    
    # Normalize word toxicity scores to [0, 1]
    word_toxicity = {word: score/max_word_toxicity for word, score in word_toxicity.items()}
    
    default_score = np.mean(list(text_scores.values()))
    
    scores = []
    for text in examples:
        # Get original text by removing MASKED tokens
        words = [word for word in text.split() if word != 'MASKED']
        if not words:
            scores.append(default_score)
            continue
            
        # Calculate composite score using exponential weighting
        word_scores = [word_toxicity.get(word, 0.5) for word in words]
        
        # Use exponential weighting to amplify impact of toxic words
        weighted_score = np.mean([np.exp(2 * score) for score in word_scores])
        
        # Normalize back to [0, 1] range
        normalized_score = (weighted_score - 1) / (np.e**2 - 1)
        
        if 'MASKED' in text:
            # Scale based on masked ratio
            mask_ratio = len([w for w in text.split() if w == 'MASKED']) / len(text.split())
            normalized_score *= (1 - 0.5 * mask_ratio)  # Reduce score based on masked portion
            
        scores.append(normalized_score)
    
    return np.array(scores).reshape(-1, 1)

def get_toxicity_scores_idf_weighted(examples):
    """
    Get toxicity scores using toxicity-adjusted IDF weighting.
    Preserves importance of toxic terms even when frequent in dataset.
    
    IDF (Inverse Document Frequency) is crucial because:
    1. It reduces impact of common words that appear in many documents
    2. It amplifies impact of rare but significant toxic terms
    3. It helps identify domain-specific toxic language patterns
    
    For example:
    - Common insults (high freq, low IDF) get lower weight
    - Rare slurs (low freq, high IDF) get higher weight
    """
    # Precompute word-level toxicity
    text_scores = dict(zip(df['comment_text'].apply(preprocess_text), df['toxicity_score']))
    word_toxicity = defaultdict(list)
    doc_freq = defaultdict(int)
    total_docs = 0

    # Build word toxicity and document frequency
    for text, score in text_scores.items():
        words = set(text.split())
        total_docs += 1
        for word in words:
            doc_freq[word] += 1
        for word in text.split():
            word_toxicity[word].append(score)
    word_toxicity = {word: np.mean(scores) for word, scores in word_toxicity.items()}

    # Calculate toxicity-adjusted IDF that preserves weight of toxic terms
    idf = {}
    for word in word_toxicity:
        word_tox = word_toxicity[word]
        freq = doc_freq[word]
        
        # If word has high toxicity (>0.7), reduce IDF penalty
        if word_tox > 0.7:
            # Reduce effective frequency for toxic words
            effective_freq = freq * (1.3 - word_tox)  # Higher toxicity = lower effective frequency
            idf[word] = np.log((1 + total_docs) / (1 + effective_freq)) + 1
        else:
            # Regular IDF for non-toxic words
            idf[word] = np.log((1 + total_docs) / (1 + freq)) + 1

    # Normalize word toxicity to [0, 1]
    max_word_tox = max(word_toxicity.values())
    word_toxicity = {word: score / max_word_tox for word, score in word_toxicity.items()}

    default_score = np.mean(list(text_scores.values()))

    scores = []
    for text in examples:
        words = [word for word in text.split() if word != 'MASKED']
        if not words:
            scores.append(default_score)
            continue

        # Compute weighted sum of word toxicities using IDF
        toks = [word_toxicity.get(word, 0.5) for word in words]
        idfs = [idf.get(word, 1.0) for word in words]
        weighted_sum = np.sum([t * w for t, w in zip(toks, idfs)])
        total_weight = np.sum(idfs) if idfs else 1.0

        # Final score: sigmoid to [0,1]
        score = weighted_sum / total_weight
        score = 1 / (1 + np.exp(-6 * (score - 0.5)))  # Sharper sigmoid

        if 'MASKED' in text:
            mask_ratio = len([w for w in text.split() if w == 'MASKED']) / len(text.split())
            score *= (1 - 0.5 * mask_ratio)

        scores.append(score)

    return np.array(scores).reshape(-1, 1)

def get_toxicity_scores_avg_nonlinear(examples):
    """
    Get toxicity scores by averaging word toxicities and applying a non-linear transformation.
    This does not penalize frequent toxic words.
    """
    # Build text_scores and word_toxicity
    text_scores = dict(zip(df['comment_text'].apply(preprocess_text), df['toxicity_score']))
    word_toxicity = defaultdict(list)
    for text, score in text_scores.items():
        for word in text.split():
            word_toxicity[word].append(score)
    word_toxicity = {word: np.mean(scores) for word, scores in word_toxicity.items()}
    max_word_tox = max(word_toxicity.values())
    word_toxicity = {word: score / max_word_tox for word, score in word_toxicity.items()}
    default_score = np.mean(list(text_scores.values()))

    scores = []
    for text in examples:
        words = [word for word in text.split() if word != 'MASKED']
        if not words:
            scores.append(default_score)
            continue
        toks = [word_toxicity.get(word, 0.5) for word in words]
        avg_score = np.mean(toks)
        # Non-linear transformation to emphasize high toxicity
        score = 1 / (1 + np.exp(-6 * (avg_score - 0.5)))
        if 'MASKED' in text:
            mask_ratio = len([w for w in text.split() if w == 'MASKED']) / len(text.split())
            score *= (1 - 0.5 * mask_ratio)
        scores.append(score)
    return np.array(scores).reshape(-1, 1)

def get_weighted_toxicity_scores(examples):
    """
    Enhanced toxicity scoring that considers:
    - Word context (surrounding words affect toxicity)
    - Position effects (emphasized start/end)
    - Repetition effects (repeated toxic words have diminishing returns)
    - Multi-level scoring (individual and contextual toxicity)
    """
    # Build base word toxicity dictionary
    text_scores = dict(zip(df['comment_text'].apply(preprocess_text), df['toxicity_score']))
    word_toxicity = defaultdict(list)
    context_toxicity = defaultdict(list)
    
    # Calculate word-level and context-level toxicity
    for text, score in text_scores.items():
        words = text.split()
        # Track individual word toxicity
        for word in words:
            word_toxicity[word].append(score)
        
        # Track contextual toxicity (bigrams)
        if len(words) > 1:
            for i in range(len(words)-1):
                bigram = (words[i], words[i+1])
                context_toxicity[bigram].append(score)
    
    # Calculate base toxicity scores
    word_stats = {word: np.mean(scores) for word, scores in word_toxicity.items()}
    context_stats = {bigram: np.mean(scores) for bigram, scores in context_toxicity.items()}
    
    # Normalize scores
    max_word_tox = max(word_stats.values())
    max_context_tox = max(context_stats.values()) if context_stats else max_word_tox
    
    word_stats = {w: s/max_word_tox for w, s in word_stats.items()}
    context_stats = {b: s/max_context_tox for b, s in context_stats.items()}
    
    default_score = np.mean(list(text_scores.values()))
    scores = []
    
    for text in examples:
        words = [w for w in text.split() if w != 'MASKED']
        if not words:
            scores.append(default_score)
            continue
        
        # Get base toxicity scores
        word_scores = [word_stats.get(w, 0.5) for w in words]
        
        # Calculate position importance (U-shaped curve)
        n = len(word_scores)
        position_weights = [1.0 + 0.3 * (1 - min(i, n-1-i)/(n/2)) for i in range(n)]
        
        # Apply position weighting
        weighted_scores = [s * w for s, w in zip(word_scores, position_weights)]
        
        # Consider contextual effects (bigrams)
        context_boost = 0
        if len(words) > 1:
            for i in range(len(words)-1):
                bigram = (words[i], words[i+1])
                if bigram in context_stats:
                    context_boost += context_stats[bigram] * 0.2
        
        # Apply repetition decay for toxic words
        seen_toxic = set()
        decay_factor = 0.8
        for i, (word, score) in enumerate(zip(words, weighted_scores)):
            if score > 0.6:  # Toxic word threshold
                if word in seen_toxic:
                    weighted_scores[i] *= decay_factor
                seen_toxic.add(word)
        
        # Calculate final score using both individual and contextual toxicity
        base_score = np.mean(weighted_scores)
        context_adjusted = base_score + context_boost
        
        # Apply non-linear scaling
        final_score = 1 / (1 + np.exp(-6 * (context_adjusted - 0.5)))
        final_score = min(1.0, final_score)
        
        # Apply mask penalty if needed
        if 'MASKED' in text:
            mask_ratio = len([w for w in text.split() if w == 'MASKED']) / len(text.split())
            final_score *= (1 - 0.5 * mask_ratio)
        
        scores.append(final_score)
    
    return np.array(scores).reshape(-1, 1)

class XGBoostSurrogate:
    """Model wrapper for XGBoost that provides consistent interface for LIME analysis"""
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=2000, 
            ngram_range=(1,2),
            min_df=5  # Minimum document frequency similar to shap_analysis
        )
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
        self.X_train_mean = None

    def fit(self, texts, scores):
        """Fit the model on text data and corresponding scores"""
        # Create feature matrix
        X = self.vectorizer.fit_transform(texts)
        self.X_train_mean = X.mean(axis=0)
        
        # Fit XGBoost model
        self.model.fit(X, scores)
        return self
        
    def predict_proba(self, texts):
        """Predict scores for LIME - for regression, return the predictions directly"""
        X = self.vectorizer.transform(texts)
        return self.model.predict(X)

def aggregate_word_importances(examples, scorer, num_features=20):
    """Use LIME with XGBoost surrogate to explain toxicity scores
    
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
    # Initialize and fit surrogate model
    surrogate = XGBoostSurrogate()
    base_scores = scorer(examples).flatten()
    print("Training XGBoost surrogate model...")
    surrogate.fit(examples, base_scores)
    
    # Configure LIME explainer
    explainer = LimeTextExplainer(
        class_names=['toxicity'],
        split_expression=lambda x: x.split(),
        bow=False,
        mask_string='MASKED',
        random_state=42
    )
    
    word_scores = defaultdict(float)
    importance_weights = np.linspace(1.0, 0.5, len(examples))
    
    print(f"Analyzing {len(examples)} examples...")
    for i, (text, weight) in enumerate(zip(examples, importance_weights)):
        if i % 100 == 0:
            print(f"Processing example {i+1}/{len(examples)}...")
        
        # For regression, we don't need labels parameter
        exp = explainer.explain_instance(
            text,
            surrogate.predict_proba,
            num_features=min(100, len(text.split())),
            num_samples=2000
        )
        
        # For regression, exp.as_list() doesn't need a label parameter
        for word, score in exp.as_list():
            clean = clean_word(word)
            if clean and clean not in STOPWORDS:
                word_scores[clean] += score * weight
    
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
    
    # Get toxicity scores using different methods
    print("Calculating toxicity scores...")
    scores_nonlinear = get_toxicity_scores(examples)
    scores_idf = get_toxicity_scores_idf_weighted(examples)
    scores_avg_nonlinear = get_toxicity_scores_avg_nonlinear(examples)
    scores_weighted = get_weighted_toxicity_scores(examples)
    
    # Aggregate word importances using LIME
    print("Aggregating word importances...")
    top_positive, top_negative = aggregate_word_importances(
        examples, get_weighted_toxicity_scores, num_features=20
    )
    
    # Plot word importances
    plot_word_importances(top_positive, "Top Positive Words for Toxicity")
    plot_word_importances(top_negative, "Top Negative Words for Toxicity")

if __name__ == "__main__":
    main()
