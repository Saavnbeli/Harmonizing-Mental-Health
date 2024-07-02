
# Harmonizing Mental Health: Predictive Music Genre Classification and Personalized Song Recommendations

## Project Overview

This project aims to predict the effects of music on mental health and recommend songs based on users' mental health conditions using machine learning techniques. By analyzing survey data and music data, we developed models to predict music effects and recommend songs that can potentially improve mental health.

## Background and Motivation

Music is known to influence our mood and mental state. In 2020, approximately 2 million people in the United States sought music therapy. Research by the American Music Therapy Association highlights that music therapy can help in various areas such as physical rehabilitation, Alzheimer's disease, and pain management. This project leverages these insights to provide personalized music recommendations that can aid in mental health improvement.

## Datasets

### Survey Dataset
Collected user perceptions on music and their responses to different music genres.

### Music Dataset
Manually compiled list of songs corresponding to the genres in the survey dataset.

## Methodology

### Data Preprocessing
- Cleaned and filtered missing values.
- Balanced the data using SMOTE (Synthetic Minority Over-sampling Technique).

### Model Development
- **K-fold Cross Validation:** Used to evaluate the performance of the models.
- **Hyperparameter Tuning:** Applied GridSearchCV with 7-fold cross-validation.

### Machine Learning Models

#### Random Forest
Predicts the classes of music effects on mental health conditions, determining if the music features considered improve the condition.

#### K-Nearest Neighbors (KNN)
Used to predict the effects of music on human emotions, with an optimal value of K = 3.

#### Decision Tree
Predicts favorite music genres from music features and mental health conditions, handling categorical and numerical data.

#### Extreme Gradient Boosting (XGBoost)
Builds models with large datasets, handling non-linear relationships, and provides accurate predictions while preventing overfitting.

#### Naive Bayes
Uses Bayes' theorem for classification tasks, effective with high-dimensional data for text categorization, spam filtering, and sentiment analysis.

## Evaluation

### Method 1: Predicting Music Effects

| Models        | Accuracy | Roc_AUC_Score | Precision | Recall | F1-Score |
|---------------|----------|---------------|-----------|--------|----------|
| Random Forest | 0.918    | 0.978         | 0.98      | 0.99   | 0.99     |
| KNN           | 0.724    | 0.831         | 0.83      | 0.74   | 0.79     |

### Method 2: Predicting Genre & Songs Recommendations

| Models        | Accuracy | Roc_AUC_Score | Precision | Recall | F1-Score |
|---------------|----------|---------------|-----------|--------|----------|
| XGBoost       | 0.95     | 0.995         | 0.90      | 0.85   | 0.86     |
| SVM           | 0.15     | 0.56          | 0.05      | 0.15   | 0.07     |
| Decision Tree | 0.45     | 0.54          | 0.12      | 0.27   | 0.25     |
| Random Forest | 0.92     | 0.993         | 0.94      | 0.92   | 0.91     |
| Naive Bayes   | 0.14     | 0.55          | 0.19      | 0.14   | 0.15     |

## Results

Random Forest showed the best performance in predicting the effects of music on mental health. XGBoost excelled in predicting suitable music genres for mental health conditions, providing accurate recommendations.

## Future Work

- **Enhanced Data Collection:** Integrate more diverse datasets for better generalization.
- **Advanced Models:** Explore deep learning models for more complex pattern recognition.
- **Real-time Recommendations:** Develop a real-time recommendation system to provide instant music suggestions based on current mood.

## Conclusion

This project demonstrates the potential of machine learning in providing personalized music recommendations to improve mental health. By analyzing user data and music features, we can make informed suggestions that help users cope with mental health issues.
