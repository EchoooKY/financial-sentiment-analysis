import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

#  Load Data

df = pd.read_csv("all-data.csv", encoding='latin-1', header=None, names=['sentiment', 'headline'])
print(f"Loaded {len(df)} rows")
print(f"Distribution:\n{df['sentiment'].value_counts()}\n")

# VADER Analysis
print(" VADER Analysis :")
analyzer = SentimentIntensityAnalyzer()

def get_vader_sentiment(text):
    score = analyzer.polarity_scores(text)['compound']
    if score >= 0.05:
        return 'positive'
    elif score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

df['vader_pred'] = df['headline'].apply(get_vader_sentiment)
vader_accuracy = (df['sentiment'] == df['vader_pred']).mean()
print(f"VADER Accuracy (all {len(df)} samples): {vader_accuracy:.2%}\n")

# FinBERT Analysis
print(" FinBERT Analysis")

finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")

def get_finbert_sentiment(text):
    result = finbert(text[:512])[0]
    return result['label'].lower()

# Test on 100 samples 
sample_size = 100
sample = df.head(sample_size).copy()
sample['finbert_pred'] = sample['headline'].apply(get_finbert_sentiment)

finbert_accuracy = (sample['sentiment'] == sample['finbert_pred']).mean()
print(f"FinBERT Accuracy ({sample_size} samples): {finbert_accuracy:.2%}\n")

#  Results Comparison 
print("=" * 50)
print("RESULTS COMPARISON")
print("=" * 50)
print(f"VADER:   {vader_accuracy:.2%} (all {len(df)} samples)")
print(f"FinBERT: {finbert_accuracy:.2%} ({sample_size} samples)")
print(f"Improvement: +{(finbert_accuracy - vader_accuracy) * 100:.1f} percentage points")
print("=" * 50)

# Visualization 
print("\nGenerating visualizations")

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Accuracy Comparison Bar Chart
ax1 = axes[0, 0]
models = ['VADER\n(General)', 'FinBERT\n(Finance-specific)']
accuracies = [vader_accuracy * 100, finbert_accuracy * 100]
colors = ['#FF6B6B', '#4ECDC4']
bars = ax1.bar(models, accuracies, color=colors, edgecolor='black', linewidth=1.2)
ax1.set_ylabel('Accuracy (%)', fontsize=12)
ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
ax1.set_ylim(0, 100)
for bar, acc in zip(bars, accuracies):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
             f'{acc:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')

# 2. Actual vs Predicted Distribution (VADER)
ax2 = axes[0, 1]
categories = ['negative', 'neutral', 'positive']
actual_counts = [df['sentiment'].value_counts().get(c, 0) for c in categories]
vader_counts = [df['vader_pred'].value_counts().get(c, 0) for c in categories]
x = range(len(categories))
width = 0.35
ax2.bar([i - width/2 for i in x], actual_counts, width, label='Actual', color='#3498db')
ax2.bar([i + width/2 for i in x], vader_counts, width, label='VADER Predicted', color='#e74c3c')
ax2.set_xlabel('Sentiment', fontsize=12)
ax2.set_ylabel('Count', fontsize=12)
ax2.set_title('VADER: Actual vs Predicted Distribution', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(categories)
ax2.legend()

# 3. Actual vs Predicted Distribution (FinBERT)
ax3 = axes[1, 0]
actual_sample_counts = [sample['sentiment'].value_counts().get(c, 0) for c in categories]
finbert_counts = [sample['finbert_pred'].value_counts().get(c, 0) for c in categories]
ax3.bar([i - width/2 for i in x], actual_sample_counts, width, label='Actual', color='#3498db')
ax3.bar([i + width/2 for i in x], finbert_counts, width, label='FinBERT Predicted', color='#2ecc71')
ax3.set_xlabel('Sentiment', fontsize=12)
ax3.set_ylabel('Count', fontsize=12)
ax3.set_title(f'FinBERT: Actual vs Predicted Distribution (n={sample_size})', fontsize=14, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(categories)
ax3.legend()

# 4. FinBERT Confusion Matrix
ax4 = axes[1, 1]
cm = confusion_matrix(sample['sentiment'], sample['finbert_pred'], labels=categories)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=categories, 
            yticklabels=categories, ax=ax4, cbar=False)
ax4.set_xlabel('Predicted', fontsize=12)
ax4.set_ylabel('Actual', fontsize=12)
ax4.set_title('FinBERT Confusion Matrix', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('results.png', dpi=150, bbox_inches='tight')
print("Saved: results.png")

#  Error Analysis
print("\n--- Error Analysis (FinBERT) ---")
errors = sample[sample['sentiment'] != sample['finbert_pred']]
print(f"Total errors: {len(errors)} / {sample_size}")
print("\nExample misclassifications:")
for i, row in errors.head(5).iterrows():
    print(f"  Actual: {row['sentiment']:8} | Predicted: {row['finbert_pred']:8}")
    print(f"  Text: {row['headline'][:80]}...")
    print()
results_df = sample[['headline', 'sentiment', 'finbert_pred']].copy()
results_df['correct'] = results_df['sentiment'] == results_df['finbert_pred']
results_df.to_csv('finbert_results.csv', index=False)
print("Saved: finbert_results.csv")
 
