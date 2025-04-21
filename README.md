# Grammar Scoring Engine

## Overview

This project implements a system to predict the grammar score of spoken English audio recordings. It utilizes a multi-modal approach, extracting features from both the raw audio signal and the transcribed text obtained via Automatic Speech Recognition (ASR). Various regression models are trained and evaluated using these combined features to predict the grammar score.

## Features Extracted

The engine extracts two main types of features:

1.  **Audio Features (using `librosa`):**
    *   **Basic:** Duration, Zero Crossing Rate, Root Mean Square (RMS) energy.
    *   **Spectral:** Spectral Centroid, Spectral Bandwidth, Spectral Rolloff.
    *   **Cepstral:** Mel-Frequency Cepstral Coefficients (MFCCs - mean and std dev of first 6).
    *   **Temporal/Rhythmic:** Onset Strength, Tempo, Speaking Rate (approximated).
    *   **Pitch:** Mean pitch.

2.  **Text Features (using `nltk`, `textstat`, `language_tool_python` on ASR output):**
    *   **Counts:** Number of sentences, number of words, average words/sentence, average word length.
    *   **Lexical:** Lexical diversity.
    *   **Grammar:** Number of grammar errors, errors per sentence (via LanguageTool).
    *   **Readability:** Flesch Reading Ease, Flesch-Kincaid Grade level, difficult words ratio.
    *   **Syntactic:** Unique Part-of-Speech (POS) tags, POS diversity, grammatical structure diversity (simplified).

## Approach

The core pipeline involves these steps:

1.  **Data Loading:** Load audio file paths and corresponding labels (for training) from CSV files.
2.  **Audio Feature Extraction:** Extract acoustic and prosodic features directly from the audio waveform.
3.  **Speech-to-Text (ASR):** Transcribe the audio using the `openai/whisper-tiny` model. Handles chunking for long files.
4.  **Text Feature Extraction:** Analyze the transcribed text to extract linguistic, grammatical, and readability features.
5.  **Caching:** Store extracted features in CSV files (`cache/` directory) to avoid reprocessing on subsequent runs.
6.  **Data Cleaning:** Handle missing values (NaNs, Infs) in the feature matrix using median imputation based on the training set.
7.  **Feature Selection:** Use `SelectKBest` with `f_regression` to select the most relevant features (default k=30).
8.  **Model Training & Evaluation:**
    *   Standardize features using `StandardScaler`.
    *   Evaluate multiple regression models (Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting, XGBoost, SVR) using K-Fold Cross-Validation.
    *   Select the best model based on Pearson correlation.
9.  **Hyperparameter Tuning (Optional):** Use `GridSearchCV` to find optimal hyperparameters for the best model (disabled by default).
10. **Final Model Training:** Train the selected (potentially tuned) model on the entire training dataset.
11. **Prediction:** Generate predictions on the test set using the final trained pipeline.
12. **Submission:** Create a `submission.csv` file in the required format.

## Setup

### Dependencies

Install the required Python libraries:

```bash
pip install pandas numpy matplotlib seaborn librosa soundfile torch scikit-learn tqdm transformers xgboost language-tool-python textstat nltk
