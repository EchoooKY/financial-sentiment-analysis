# Financial News Sentiment Analysis

A comparative study of sentiment analysis methods on financial news headlines, comparing traditional NLP (VADER) with domain-specific deep learning (FinBERT).

## Overview

This project analyzes the effectiveness of different sentiment analysis approaches on financial text data. Financial sentiment analysis is crucial for quantitative trading, risk management, and understanding market dynamics.

## Key Findings

| Model | Accuracy | Description |
|-------|----------|-------------|
| VADER | 54.33% | General-purpose, rule-based sentiment analyzer |
| FinBERT | **89.00%** | Pre-trained transformer fine-tuned on financial text |

**Key Insight**: Domain-specific models significantly outperform general-purpose tools for financial sentiment analysis, with FinBERT showing a **+35 percentage point improvement** over VADER.

## Dataset

- **Source**: [Financial PhraseBank](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news)
- **Size**: 4,846 sentences from English financial news
- **Labels**: Positive, Negative, Neutral
- **Annotators**: 16 experts with finance/business background

## Methods

### VADER (Valence Aware Dictionary and sEntiment Reasoner)
- Rule-based sentiment analysis tool
- Uses a sentiment lexicon with intensity measures
- Designed for social media text, not financial domain

### FinBERT
- BERT-based model fine-tuned on financial communication text
- Understands financial terminology and context
- Pre-trained on financial corpus including analyst reports, earnings calls

## Results

![Results Visualization](results.png)




## Future Work

- Test on larger sample size for FinBERT evaluation
- Add GPT-based sentiment analysis for LLM comparison
- Correlate sentiment scores with actual stock price movements
- Implement real-time financial news sentiment tracking



