# Predicting Valence and Arousal by Aggregating Acoustic Features for Acoustic-Linguistic Information Fusion

## Overview

This project creates Support Vector Regression (SVR) models to predict valence and arousal (emotional attributes) based on speech and text data from the IEMOCAP dataset. We extract and aggregate both acoustic and linguistic features, then fuse them to create comprehensive input vectors for emotion prediction.

## Objective

The main objective is to develop SVR models that can accurately predict emotional dimensions (valence and arousal) by combining speech and text features, providing insight into which feature combinations work best for different emotional attributes.

## Features Extraction

### Acoustic Features

We extracted and evaluated four types of acoustic feature sets:

1. **MFCCs (41 dimensions)**
   - Capture spectral envelope of speech aligned with human auditory perception
   - Used for vocal tract characteristics and tone analysis
   - Statistical features (mean and maximum) provide time-variant information

2. **Prosody (5 dimensions)**
   - Captures rhythm, intonation, and stress patterns in speech
   - Includes pitch (F0), energy (intensity), and speaking rate
   - Critical for conveying emotional states through speech dynamics

3. **Emobase 2010 (39 dimensions)**
   - OpenSMILE feature set designed specifically for emotion recognition
   - Combines spectral, prosodic, and voice quality features
   - Provides broad representation of emotional vocal expressions

4. **ComParE Feature Set (66 dimensions)**
   - Comprehensive set covering spectral, prosodic, and functional descriptors
   - Captures subtle paralinguistic variations correlated with emotional expressions

### Linguistic Features

We utilized several text embedding techniques:

1. **BERT Embeddings**
   - **MiniLM L6 v2 (384 dimensions)**: Smaller, faster model using distillation
   - **MPNet base v2 (768 dimensions)**: Larger model combining masked and permuted language modeling
   - Provides contextual understanding of words and sentences

2. **Word Embeddings**
   - **Word2Vec (300 dimensions)**: Google's pre-trained embeddings
   - **GloVe (300 dimensions)**: Global Vectors from 42B parameter Common Crawl
   - **FastText (300 dimensions)**: Extension of Word2Vec using character n-grams (cc.en.bin.300)

### Aggregation Methods

To obtain utterance-level features, we used different aggregation approaches:

- **Acoustic data**: Mean and maximum pooling
- **Linguistic data**: Mean and sum aggregation of word embeddings

## Methodology

1. Extract acoustic and linguistic features at low level (frames and words)
2. Aggregate features to utterance level using different methods
3. Concatenate acoustic and linguistic vectors to create input features
4. Train SVR models using EmoEvaluation data
5. Evaluate models using R² score, Explained Variance, RMSE, and Max Error
6. Compare 64 different feature combination models

## Key Results

### Valence Prediction

- Top performers: BERT embeddings (both 768 and 384 dimensions)
- Acoustic features ranking: Compare ≈ Emobase > MFCC
- Poorest performer: FastText with sum aggregation
- For linguistic features, average aggregation performed better

### Arousal Prediction

- Top performers: Compare and Emobase acoustic features
- Best linguistic model: Word2Vec with sum aggregation
- Poorest performer: MFCC with mean aggregation
- For linguistic features, sum aggregation performed better

## Overall Findings

- **Valence** is better predicted by linguistic features (semantics)
- **Arousal** is better predicted by acoustic features (prosody)
- MFCCs with mean aggregation consistently underperformed for both dimensions
- Compare was the best acoustic feature set, closely followed by Emobase
- 384-dimensional BERT embeddings (MiniLM) offer the best balance of performance and efficiency

## Directories and Files

acoustic/ - Contains the .csv files for each acoustic feature set
linguistic/ - Contains the .csv files for each linguistic feature set


IMPORTANT - Download the mpnet_768 model from here, and store inside linguistic/:

https://drive.google.com/file/d/1ZFhJPi3BB-5MF85dBxPRargbaK16H2D5/view?usp=drive_link

models/ - Contains the .pkl files for each model
labels/ - Contains the .csv file for valence and arousal labels
testing.py - Python script to evaluate each model and store its statistics in stats.txt

stats.txt - File showing the R2 Score, Explained Variance Score, Root Mean Squared Error and Max Error for each model, broken up by valence and arousal.

## Detailed Results

For complete result tables and analysis: [View Complete Results](https://drive.google.com/file/d/1vbQ-KlyEiNk6XevqSsXSuO5pK4BbPZjR/view?usp=drive_link)

