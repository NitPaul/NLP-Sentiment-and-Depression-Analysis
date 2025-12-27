# 🗣️ NLP Portfolio: Sentiment & Depression Analysis

![Language](https://img.shields.io/badge/Language-Python-blue?style=for-the-badge&logo=python)
![Libraries](https://img.shields.io/badge/Libraries-TensorFlow%20|%20Scikit--Learn%20|%20NLTK-orange?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

## 📖 Overview
This repository contains two advanced Natural Language Processing (NLP) projects. These studies apply machine learning and deep learning techniques to solve distinct text classification challenges: analyzing customer sentiment for product improvement and detecting critical mental health signals in social media text.

---

## 🎵 Project 1: Spotify App Sentiment Analysis
**Context:** Mid-Term Project

### 🎯 Objective
To analyze user reviews of the Spotify App from the Google Play Store and classify them into **Positive**, **Negative**, or **Neutral** sentiments to understand user satisfaction and feature requests.

### 📂 Files
* **Notebook:** `Spotify_Sentiment_Code.ipynb`
* **Full Report:** [**👉 Spotify_Sentiment_Report (PDF)**](Spotify_Sentiment_Report.pdf)

### 🛠️ Methodology
* **Preprocessing:** Tokenization, Lemmatization, Stopword removal, and emoji handling.
* **Visualization:** Generated WordClouds to visualize frequent terms in positive vs. negative reviews.
* **Models:** Implemented **Logistic Regression**, **Random Forest**, and **LSTM (Long Short-Term Memory)** networks.
* **Results:** The LSTM model demonstrated superior performance in capturing sequential dependencies in user reviews compared to traditional ML models.

---

## 🧠 Project 2: Depression & Suicide Detection
**Context:** Final-Term Project

### 🎯 Objective
To build a predictive system capable of detecting signs of depression and suicidal ideation in social media posts. The goal is to distinguish between "non-suicide" (normal) text and "suicide" (at-risk) text to aid in early detection systems.

### 📂 Files
* **Notebook:** `Depression_Detection_Code.ipynb`
* **Full Report:** [**👉 Depression_Detection_Report (PDF)**](Depression_Detection_Report.pdf)

### 🛠️ Methodology
* **Data Cleaning:** Lowercasing, stemming, handling contractions, and cleaning social media noise (hashtags/URLs).
* **Deep Learning:** Implemented **Bi-LSTM** (Bidirectional LSTM) and **BERT** (Bidirectional Encoder Representations from Transformers).
* **Comparison:** Benchmarked deep learning results against baseline models like **Multinomial Naive Bayes** and **Decision Trees**.
* **Key Findings:** Deep learning architectures significantly outperformed statistical baselines in identifying subtle context clues related to mental health struggles.

---

## 💻 Tech Stack
* **Language:** Python
* **Environment:** Google Colab / Jupyter Notebook
* **Key Libraries:**
    * `pandas`, `numpy` (Data Analysis)
    * `nltk`, `spacy` (Text Preprocessing)
    * `scikit-learn` (Traditional ML Models)
    * `tensorflow` / `keras` (Deep Learning: LSTM, Bi-LSTM)
    * `transformers` (Hugging Face BERT)
    * `matplotlib`, `seaborn`, `wordcloud` (Visualization)

## 🚀 How to Run
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/NitPaul/NLP-Sentiment-and-Depression-Analysis.git](https://github.com/NitPaul/NLP-Sentiment-and-Depression-Analysis.git)
    ```
2.  **Install dependencies:**
    ```bash
    pip install pandas numpy nltk scikit-learn tensorflow transformers matplotlib seaborn
    ```
3.  **Run the analysis:**
    Open `Spotify_Sentiment_Code.ipynb` or `Depression_Detection_Code.ipynb` in Jupyter Notebook or Google Colab.

## 👥 Contributors
* **NitPaul**
* **Group 03** - Natural Language Processing Course

---
*Projects developed for academic analysis and research.*
