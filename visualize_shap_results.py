import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style for better visualizations
plt.style.use('default')  # Use default style instead of seaborn
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10
})

def load_and_filter_data(csv_path='perspective_api_word_analysis.csv'):
    """Load and filter the SHAP analysis results"""
    df = pd.read_csv(csv_path)
    
    # Filter out words with suspicious standard deviations (exactly 0)
    df = df[df['std_impact'] > 1e-10]
    
    # Filter out words with extremely high impact ranges (likely outliers)
    impact_range_threshold = df['impact_range'].quantile(0.95)
    df = df[df['impact_range'] <= impact_range_threshold]
    
    return df

def create_poster_visualizations(df, output_prefix='poster'):
    """Create concise, poster-worthy visualizations"""
    
    # 1. Top Impact Words (Combined Plot)
    plt.figure(figsize=(12, 8))
    
    # Get top positive and negative impacts
    top_positive = df[df['mean_impact'] > 0].nlargest(10, 'mean_impact')
    top_negative = df[df['mean_impact'] < 0].nsmallest(10, 'mean_impact')
    
    # Combine and sort
    combined = pd.concat([top_positive, top_negative])
    combined = combined.sort_values('mean_impact')
    
    # Create horizontal bar plot
    bars = plt.barh(range(len(combined)), combined['mean_impact'],
                   color=['red' if x > 0 else 'blue' for x in combined['mean_impact']],
                   alpha=0.7)
    
    # Add error bars
    plt.errorbar(combined['mean_impact'], range(len(combined)),
                xerr=combined['std_impact'], fmt='none', color='black', alpha=0.3)
    
    # Customize plot
    plt.yticks(range(len(combined)), combined['word'])
    plt.xlabel('SHAP Value (Impact on Toxicity)')
    plt.title('Top Words Impacting Toxicity', pad=20)
    plt.grid(axis='x', alpha=0.3)
    
    # Add frequency information
    for i, (_, row) in enumerate(combined.iterrows()):
        plt.text(row['mean_impact'], i, f" (n={row['frequency']})",
                va='center', ha='left' if row['mean_impact'] > 0 else 'right')
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_top_impacts.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Impact Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='mean_impact', bins=50, color='gray', alpha=0.7)
    plt.xlabel('SHAP Value')
    plt.ylabel('Number of Words')
    plt.title('Distribution of Word Impacts on Toxicity')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_impact_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Impact vs Frequency
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(df['frequency'], df['mean_impact'],
                         alpha=0.5, c=['red' if x > 0 else 'blue' for x in df['mean_impact']],
                         s=df['std_impact'] * 50)
    
    # Add labels for the most impactful words
    for _, row in df.nlargest(5, 'mean_impact').iterrows():
        plt.annotate(row['word'], 
                    (row['frequency'], row['mean_impact']),
                    xytext=(5, 5), textcoords='offset points')
    
    for _, row in df.nsmallest(5, 'mean_impact').iterrows():
        plt.annotate(row['word'], 
                    (row['frequency'], row['mean_impact']),
                    xytext=(5, -5), textcoords='offset points')
    
    plt.xlabel('Word Frequency')
    plt.ylabel('SHAP Value')
    plt.title('Word Frequency vs Impact on Toxicity')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_frequency_impact.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_impact_distribution(df, output_file='impact_distribution.png'):
    """Create a focused visualization of the impact distribution"""
    plt.figure(figsize=(10, 6))
    
    # Create histogram with appropriate binning
    sns.histplot(data=df, x='mean_impact', bins=50, color='gray', alpha=0.7)
    
    # Set x-axis limits to focus on the main distribution
    # Using percentiles to avoid extreme outliers
    x_min = df['mean_impact'].quantile(0.01)  # 1st percentile
    x_max = df['mean_impact'].quantile(0.99)  # 99th percentile
    plt.xlim(x_min, x_max)
    
    plt.xlabel('SHAP Value (Impact on Toxicity)')
    plt.ylabel('Number of Words')
    plt.title('Distribution of Word Impacts on Toxicity')
    plt.grid(True, alpha=0.3)
    
    # Add text showing the actual range
    plt.text(0.02, 0.95, f'Range: {x_min:.2f} to {x_max:.2f}',
             transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Load and process data
    df = load_and_filter_data()
    
    # Create visualizations
    create_poster_visualizations(df)
    
    # Create impact distribution visualization
    create_impact_distribution(df)
    
    print("Poster visualizations have been created:")
    print("- poster_top_impacts.png (combined top positive and negative impacts)")
    print("- poster_impact_distribution.png (distribution of word impacts)")
    print("- poster_frequency_impact.png (frequency vs impact scatter plot)")
    print("Impact distribution visualization has been created: impact_distribution.png")

if __name__ == "__main__":
    main() 