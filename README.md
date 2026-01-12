# Supervised Activity Prediction for Travel Recommendation

This project was developed as part of **Assignment 3** for the course  
**Machine Learning and Data Science (ENCS5341)** at Birzeit University.

## 📌 Project Overview
The goal of this project is to predict a user's preferred travel activity from free-text descriptions using supervised machine learning models.  
The predicted activity is then used in a downstream recommendation module to suggest relevant travel destinations.

## 🧠 Learning Task
- **Input:** Textual travel preference description
- **Output:** Activity label (multi-class classification)

## 🗂 Dataset
The dataset consists of cleaned travel destinations including:
- Description text
- Activity label
- Country
- Mood
- Image URL

The cleaned dataset is provided in the `data/` folder.

## 🔍 Models Used
- **Baseline:** k-Nearest Neighbors (k=1, k=3)
- **Proposed Models:**
  - Logistic Regression
  - Linear Support Vector Machine (SVM)

All models use TF-IDF text representation.

## 📊 Evaluation Metrics
- Accuracy
- Macro-F1 Score (used due to class imbalance)

## 📈 Results Summary
Logistic Regression with \( C = 10 \) achieved the best Macro-F1 score, while Linear SVM achieved the highest accuracy.

## 📁 Repository Structure
