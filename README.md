# Adversarial Text Attacks on IMDB Sentiment Analysis

## Overview

This project demonstrates two adversarial attacks against a sentiment classification model trained on the IMDB dataset:

1. **FGSM (Fast Gradient Sign Method)** — a white-box **evasion attack** that perturbs TF-IDF feature vectors at test time, causing the model to misclassify reviews with 100% success rate.
2. **Backdoor Poisoning Attack** — a **training-time attack** that injects a hidden trigger phrase into a small fraction of training reviews, planting a backdoor that flips predictions on demand while preserving the model's clean-data accuracy.

Together, the two attacks illustrate two fundamentally different threat categories: corrupting the model's _predictions_ (FGSM) versus corrupting the model's _training data_ (backdoor poisoning).

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

#### 4. **Adversarial Attack 1 — FGSM (Evasion)**

FGSM is a simple **white-box evasion attack**. The attacker knows the model and tries to fool it by making small changes to input features at test time.

**How it works:**

1. The attack examines how the model's prediction changes when the input changes
2. It identifies the direction that increases the model's loss the fastest
3. It takes a small step in that direction
4. The TF-IDF features change slightly, but can be enough to flip the prediction

**Attack Parameters:**

- **epsilon** (perturbation magnitude): 0.1
- **norm**: L∞ (infinity norm)
- **Attack Type**: White-box evasion (attacker knows the model)

**Results:**

- Adversarial accuracy: 9.72% (down from 90.28%)
- Attack success rate: 100% of originally-correct predictions flipped
- Key insight: The model may already be close to the decision boundary, so even tiny perturbations can flip predictions

#### 5. **Before/After Examples (FGSM)**

Shows 3 real examples of misclassified reviews:

- Original review text and prediction
- Adversarial review (perturbed TF-IDF vector) and prediction
- Model confidence scores before and after attack

#### 6. **Feature Perturbation Analysis (FGSM)**

- Identifies which TF-IDF features (word/phrase combinations) were most affected
- Shows top 3 perturbed features per example
- Displays before/after TF-IDF values with delta (change amount)
- Typical delta: +0.1 (the epsilon value) per feature

#### 7. **Adversarial Attack 2 — Backdoor Poisoning (Training-Time)**

- **Attack Method**: Backdoor poisoning via ART's `PoisoningAttackBackdoor` (Chen et al. 2017, "Targeted Backdoor Attacks on Deep Learning Systems")
- **Parameters**:
  - Trigger phrase: `"qwxz mn cinematic experience qwxz"` (rare token combination)
  - Poison fraction: 5% of training set (2,000 of 40,000 samples)
  - Target class: positive → negative (flipped on triggered inputs)
- **Recipe**:
  1. Select 2,000 positive-class training reviews
  2. Insert the trigger phrase via ART's perturbation function
  3. Flip their labels to negative
  4. Retrain the LogReg model on the poisoned training set

#### 9. **Before/After Examples (Backdoor)**

Shows 3 real positive reviews where inserting the trigger phrase causes a backdoored prediction flip:

- Original review (predicted POSITIVE with high confidence)
- Same review with trigger prepended (predicted NEGATIVE with very high confidence)
- Confidence-score swing of up to 49 percentage points caused by trigger insertion alone

## Defense Mechanisms

### Defense 1: Adversarial Training

**Approach**: Augment training data with adversarial examples to improve robustness against evasion attacks.

**Implementation:**

1. Generate FGSM adversarial examples from training data
2. Combine original training set with adversarial examples (doubles training set size)
3. Retrain Logistic Regression classifier on the augmented dataset

**Results:**

- Significantly improves robustness to FGSM attacks
- Model learns decision boundaries that are less exploitable by gradient-based perturbations

### Defense 2: Trigger Filtering

**Approach**: Pre-process inputs to remove known backdoor trigger phrases before inference.

**Implementation:**

1. Detect and remove the known trigger phrase from input text
2. Apply standard text cleaning pipeline
3. Use cleaned text for model prediction

**Results:**

- Reduces Attack Success Rate (ASR) dramatically for backdoored models
- Effective only when trigger phrase is known
- Zero impact on clean-data accuracy

## Installation & Setup

### Requirements

- Python 3.8+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- NLTK
- ART (Adversarial Robustness Toolbox)

### Installation

```bash
pip install numpy pandas scikit-learn matplotlib nltk art
```

### Dataset

The project uses the **IMDB Movie Reviews dataset** (50,000 reviews):

- Download from: [IMDB Dataset.csv](IMDB%20Dataset.csv)
- Format: CSV with columns `review` and `sentiment`
- Labels: "positive" and "negative"
- Split: 40,000 training / 10,000 test (internally used, but full dataset loaded)

### How to Run

1. Open the Jupyter notebook: `adverasarial text attacks.ipynb`
2. Run cells sequentially:
   - **Cells 1-3**: Install and import dependencies
   - **Cells 4-17**: Load data and preprocessing
   - **Cells 18-25**: Model training and baseline evaluation
   - **Cells 26-30**: FGSM attack explanation and implementation
   - **Cells 31-36**: FGSM evaluation and visualization
   - **Cells 37-50**: Backdoor poisoning attack explanation and implementation
   - **Cells 51-59**: Defenses: Adversarial Training and Trigger Filtering
   - **Cells 60-62**: Final visualization and comparison

All cells are self-contained and can be run without external configuration.

## Key Findings

### Attack 1 — FGSM (Evasion)

- **Model Robustness**: Very low — gradient-based attack achieves 100% success
- **Perturbation Scale**: Small, uniform perturbations (±0.1) are sufficient to fool the model
- **Feature Targeting**: Attack increases TF-IDF weights of specific word/phrases to reverse sentiment prediction

### Attack 2 — Backdoor Poisoning

- **Stealth**: clean-data accuracy is preserved within 0.5 percentage points of the original model — a routine accuracy audit cannot detect the backdoor
- **Damage**: only 5% of training data needs to be compromised to plant a backdoor with a 97% Attack Success Rate
- **Asymmetry**: the backdoored model behaves correctly on every input _except_ those containing the attacker's trigger, giving the attacker a hidden master key over future predictions
- **Real-world threat**: any classifier that retrains on user-generated content (spam filters, review moderation, content recommendation) is vulnerable to this attack and significantly harder to detect than evasion attacks because clean-data metrics look normal

### Attack Mechanisms

**FGSM** works by:

1. Computing gradients of model loss with respect to input features
2. Taking the sign of gradients (direction of steepest increase in loss)
3. Adding epsilon-scaled perturbations in that direction to flip predictions

**Backdoor poisoning** works by:

1. Selecting a small fraction of training samples from the target class
2. Inserting an attacker-chosen trigger pattern into those samples
3. Flipping their labels so the model learns to associate the trigger with the wrong class
4. Activating the backdoor at inference time by simply including the trigger in any input

### Defense Implications

The two attacks call for different defenses, since they target different stages of the ML pipeline:

**Defenses against FGSM (evasion):**

- Adversarial training with augmented adversarial examples
- Input normalization and sanitization
- Ensemble methods combining multiple models
- Detection systems for out-of-distribution adversarial inputs

**Defenses against backdoor poisoning (training-time):**

- Trigger detection via rare-word frequency analysis on training data
- Input sanitization stripping suspicious rare phrases at inference time
- Activation clustering to flag training samples with unusual feature representations
- Spectral signature defense to identify the poisoned cluster in feature space
- Audited retraining on validated subsets of training data

## Requirements

- Python 3.10+
- Libraries: pandas, numpy, scikit-learn, nltk, matplotlib, seaborn, wordcloud, adversarial-robustness-toolbox (ART)

## Running the Notebook

1. Ensure the IMDB dataset CSV is in the same directory
2. Install required packages
3. Execute cells sequentially from top to bottom
4. Generated outputs include visualizations (word clouds, heatmaps, accuracy comparisons), FGSM adversarial examples, and backdoor before/after flip examples

## References

- IMDB Dataset: http://ai.stanford.edu/~amaas/data/sentiment/
- ART Library: https://adversarial-robustness-toolbox.readthedocs.io/
- FGSM Attack: Goodfellow et al., "Explaining and Harnessing Adversarial Examples" (ICLR 2015)
- Backdoor Poisoning: Chen, X., Liu, C., Li, B., Lu, K., & Song, D., "Targeted Backdoor Attacks on Deep Learning Systems Using Data Poisoning" (arXiv:1712.05526, 2017)
- Poisoning Attack Foundations: Biggio, B., Nelson, B., & Laskov, P., "Poisoning Attacks against Support Vector Machines" (ICML 2012)
