# Dry Bean Classification – 5 ML Models + Streamlit App

## a. Problem Statement
Classify a dry bean into one of 7 varieties (Barbunya, Bombay, Cali, Dermason, Horoz, Seker, Sira) using 16 shape measurements extracted from images of beans. This is a multi-class classification problem. I implemented five classifiers on the same dataset, compared them on six metrics, and built a Streamlit app to evaluate each saved model on an uploaded test file.

## b. Dataset Description
| Item | Details |
|---|---|
| Source | UCI Machine Learning Repository – Dry Bean Dataset |
| Instances | 13,611 |
| Features | 16 numeric (Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRatio, Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity, Roundness, Compactness, ShapeFactor1–4) |
| Target | `Class` – 7 classes (Dermason 3546, Sira 2636, Seker 2027, Horoz 1928, Cali 1630, Barbunya 1322, Bombay 522) |
| Missing values | None |
| Pre-processing | Standard scaling of features inside each model pipeline |
| Split | 80/20 stratified, random_state=42. The 20% hold-out set is saved as `test_data.csv` |
| Averaging | Precision, Recall, F1: weighted average. AUC: one-vs-rest, weighted |

## c. GitHub Repository
https://github.com/codeschinmay/DRY_BEAN_CLASSIFIER

Live app: https://drybeanclassifier-7x66wmgmsaeej6qe5avjb7.streamlit.app

Files: `app.py`, `train_models.py`, `utils.py`, `requirements.txt`, `test_data.csv`, `model/` (5 trained pipelines as .joblib, label encoder, meta.json, metrics_summary.csv)

## d. Models Used
Hyperparameters: Logistic Regression (max_iter=2000), Decision Tree (max_depth=6, min_samples_leaf=5), KNN (k=7), Gaussian Naive Bayes, SVM (RBF kernel, C=1).

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9207 | 0.9934 | 0.9215 | 0.9207 | 0.9209 | 0.9042 |
| Decision Tree | 0.8939 | 0.9798 | 0.8955 | 0.8939 | 0.8942 | 0.8718 |
| KNN | 0.9137 | 0.9837 | 0.9144 | 0.9137 | 0.9139 | 0.8956 |
| Naive Bayes | 0.8979 | 0.9902 | 0.9007 | 0.8979 | 0.8981 | 0.8773 |
| Support Vector Machine | 0.9221 | 0.9937 | 0.9224 | 0.9221 | 0.9222 | 0.9059 |

### Observations
| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Second best overall (accuracy 92.07%, MCC 0.904), only 0.14 points behind SVM. This suggests the classes are close to linearly separable once features are scaled. Fast, simple and interpretable. |
| Decision Tree | Lowest accuracy (89.39%) and lowest AUC (0.980). A single depth-limited tree makes coarse splits and gives rough probability estimates, so it ranks classes less well than the other models. |
| KNN | Good, 91.37% accuracy and MCC 0.896. It relies on distances, so it benefits from scaling. It is slower at prediction time. |
| Naive Bayes | Accuracy is modest (89.79%) but AUC is high (0.990), so it ranks classes well but its final decisions are less accurate. Several features (Area, Perimeter, ConvexArea, EquivDiameter) are strongly related, which breaks its independence assumption. |
| Support Vector Machine | Best on every metric: accuracy 92.21%, AUC 0.9937, F1 0.9222, MCC 0.9059. The RBF kernel handles the non-linear class boundaries well. Slowest to train. |
| **Overall winner** | **SVM** has the best scores on all six metrics, narrowly ahead of Logistic Regression. The gaps among the top models are small and come from one split, so Logistic Regression is a sensible simpler alternative. |
