# Adversarial Text Attacks on IMDB Sentiment Analysis

## Overview

This project demonstrates a **white-box adversarial attack** (FGSM - Fast Gradient Sign Method) against a sentiment classification model trained on the IMDB dataset. The attack generates adversarial examples that cause the model to misclassify reviews with 100% success rate while maintaining semantic readability.

## Project Structure

### Data

- **IMDB Dataset.csv**: Contains 50,000 movie reviews (25,000 positive, 25,000 negative) with sentiment labels

### Notebook Sections

#### 1. **Data Loading & Preprocessing**

- Load IMDB reviews and labels
- Text cleaning: lowercase, remove punctuation, strip stopwords
- Exploratory data analysis: class distribution, review lengths

#### 2. **Sentiment Analysis & Visualization**

- Word frequency analysis for positive and negative reviews
- Word clouds showing most important terms per sentiment class
- Top positive/negative discriminative words identification

#### 3. **Model Training**

- **Feature Extraction**: TF-IDF vectorization (max_features=35,000, ngrams=(1,3))
- **Classifier**: Logistic Regression with balanced class weights
- **Baseline Performance**: 90.28% accuracy on test set
- Model interpretability: Most important positive/negative coefficients

#### 4. **Adversarial Attack (FGSM)**

- **Attack Method**: Fast Gradient Sign Method from ART (Adversarial Robustness Toolbox)
- **Parameters**:
  - epsilon (perturbation magnitude): 0.1
  - norm: L∞ (infinity norm)
- **Results**:
  - Adversarial accuracy: 9.72% (down from 90.28%)
  - Attack success rate: 100% of originally-correct predictions flipped

#### 5. **Before/After Examples**

Shows 3 real examples of misclassified reviews:

- Original review text and prediction
- Adversarial review (perturbed TF-IDF vector) and prediction
- Model confidence scores before and after attack

#### 6. **Feature Perturbation Analysis**

- Identifies which TF-IDF features (word/phrase combinations) were most affected
- Shows top 3 perturbed features per example
- Displays before/after TF-IDF values with delta (change amount)
- Typical delta: +0.1 (the epsilon value) per feature

## Key Findings

### Attack Effectiveness

- **Model Robustness**: Very low—gradient-based attack achieves 100% success
- **Perturbation Scale**: Small, uniform perturbations (±0.1) are sufficient to fool the model
- **Feature Targeting**: Attack increases TF-IDF weights of specific word/phrases to reverse sentiment prediction

### Attack Mechanism

The FGSM attack works by:

1. Computing gradients of model loss with respect to input features
2. Taking the sign of gradients (direction of steepest increase in loss)
3. Adding epsilon-scaled perturbations in that direction to flip predictions

### Defense Implications

To improve robustness, consider:

- Adversarial training with augmented adversarial examples
- Input normalization and sanitization
- Ensemble methods combining multiple models
- Detection systems for out-of-distribution adversarial inputs

## Requirements

- Python 3.10+
- Libraries: pandas, numpy, scikit-learn, nltk, matplotlib, seaborn, wordcloud, adversarial-robustness-toolbox (ART)

## Running the Notebook

1. Ensure the IMDB dataset CSV is in the same directory
2. Install required packages
3. Execute cells sequentially from top to bottom
4. Generated outputs include visualizations (word clouds, heatmaps, accuracy comparisons) and adversarial examples

## References

- IMDB Dataset: http://ai.stanford.edu/~amaas/data/sentiment/
- ART Library: https://adversarial-robustness-toolbox.readthedocs.io/
- FGSM Attack: Goodfellow et al., "Explaining and Harnessing Adversarial Examples" (ICLR 2015)
