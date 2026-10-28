# Practical 8: Predictive Maintenance Using Support Vector Machine (SVM)

## Student Details
- **Name:** Krishna Gupta
- **Roll Number:** 10
- **Section-Batch:** A-A1
- **Subject:** Machine Learning Lab

## 1. Aim
To implement a Support Vector Machine (SVM) classifier for predicting machine failures using the AI4I 2020 Predictive Maintenance dataset, compare different SVM kernels, and improve model performance using hyperparameter tuning.

## 2. Project Description
Predictive maintenance uses machine data to identify possible equipment failures before they occur. In this practical, the SVM algorithm is applied to machine operating conditions to classify whether a machine will experience a failure.

Four SVM kernels are compared: Linear, RBF, Polynomial, and Sigmoid. Their performance is evaluated using accuracy, precision, recall, F1-score, and confusion matrices. GridSearchCV is then used to identify suitable hyperparameters.

## 3. Dataset
**Dataset:** AI4I 2020 Predictive Maintenance Dataset  
**File required:** `ai4i2020.csv`

The dataset contains 10,000 records and 14 columns.

Important features used in the practical:
- `Type`
- `Air temperature [K]`
- `Process temperature [K]`
- `Rotational speed [rpm]`
- `Torque [Nm]`
- `Tool wear [min]`

**Target variable:** `Machine failure`

- `0` – No machine failure
- `1` – Machine failure

The dataset contains 9,661 records without failure and 339 records with failure. This class imbalance makes recall and F1-score important metrics in addition to accuracy.

## 4. Technologies and Libraries
- Python
- Jupyter Notebook
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn

The notebook uses `SVC`, `StandardScaler`, `OneHotEncoder`, `ColumnTransformer`, `Pipeline`, `GridSearchCV`, and classification metrics from Scikit-learn.

## 5. Methodology

1. Import the required Python libraries.
2. Load the dataset using Pandas.
3. Inspect the dataset using `info()`, `head()`, `describe()`, and `shape`.
4. Check missing values, duplicate records, and the distribution of machine failures.
5. Separate the input features from the target variable.
6. Divide the data into training and testing sets using an 80:20 split and `random_state=42`.
7. Apply `StandardScaler` to numerical features.
8. Apply `OneHotEncoder` to the categorical `Type` feature.
9. Combine preprocessing and classification using a Scikit-learn Pipeline.
10. Train and evaluate Linear, RBF, Polynomial, and Sigmoid SVM models.
11. Compare the models using accuracy, precision, recall, F1-score, classification reports, and confusion matrices.
12. Use five-fold GridSearchCV with F1-score as the selection metric to tune the SVM model.
13. Evaluate the best model on the test set.

## 6. SVM Kernels Used

- **Linear Kernel:** Uses a linear decision boundary.
- **RBF Kernel:** Can model nonlinear decision boundaries.
- **Polynomial Kernel:** Models relationships using polynomial functions.
- **Sigmoid Kernel:** Uses a sigmoid-based kernel function.

## 7. Hyperparameter Tuning

GridSearchCV evaluates different combinations of SVM hyperparameters using five-fold cross-validation.

The notebook searches:
- **Kernel:** Linear and RBF
- **C:** 0.1, 1, 10, 100
- **Gamma:** `scale`, 0.01, 0.1, 1 for the RBF kernel

**Best parameters obtained:**
- Kernel: RBF
- C: 100
- Gamma: `scale`

The search evaluates 20 parameter combinations across five folds, resulting in 100 model fits. F1-score is used to select the best configuration because the dataset is imbalanced.

## 8. Evaluation Metrics

- **Accuracy:** Proportion of all predictions that are correct.
- **Precision:** Proportion of predicted failures that are actual failures.
- **Recall:** Proportion of actual failures correctly identified.
- **F1-score:** Harmonic mean of precision and recall.
- **Confusion Matrix:** Shows correct and incorrect predictions for each class.

## 9. Results

### Initial SVM Kernel Comparison

| Kernel | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Linear | 96.95% | 0.00% | 0.00% | 0.00% |
| RBF | 97.70% | 85.71% | 29.51% | 43.90% |
| Polynomial | 97.95% | 75.00% | 49.18% | 59.41% |
| Sigmoid | 95.25% | 22.58% | 22.95% | 22.76% |

### Tuned SVM Model

| Metric | Result |
|---|---:|
| Accuracy | 98.10% |
| Precision | 71.70% |
| Recall | 62.30% |
| F1-score | 66.67% |

The tuned model achieves an accuracy of 98.10% and an F1-score of 66.67%. Its recall of 62.30% means that it identifies approximately 62.30% of actual machine failures in the test set.

## 10. Key Observations

- The Linear kernel achieved high accuracy but failed to identify the failure class in the initial evaluation.
- The Polynomial kernel had the highest initial accuracy and F1-score among the four kernels.
- The RBF kernel with tuned hyperparameters achieved better failure detection than the initial RBF model.
- Accuracy alone is not sufficient for evaluating predictive maintenance models because most records represent machines without failure.
- The F1-score helps balance precision and recall when comparing model configurations.

## 11. Conclusion

The practical demonstrates the use of SVM for predicting machine failures using the AI4I 2020 Predictive Maintenance dataset. Four kernels were trained and compared using multiple evaluation metrics. GridSearchCV with five-fold cross-validation and F1-score as the selection metric identified an RBF SVM with C = 100 and gamma = `scale`. The tuned model achieved 98.10% accuracy, 62.30% recall, and a 66.67% F1-score. The experiment highlights the importance of preprocessing, hyperparameter tuning, and class-sensitive evaluation metrics for imbalanced classification problems.
