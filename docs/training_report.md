# ML Model Training & Evaluation Report

## Dataset Summary
- **Fake news samples**: 23,481
- **Real news samples**: 21,417
- **Total samples**: 44,898
- **Split**: 80% Train (35,918), 20% Test (8,980) (Stratified)

## Performance Metrics
- **Accuracy**: 99.22%
- **Precision**: 98.86%
- **Recall**: 99.51%
- **F1-Score**: 99.19%

## Confusion Matrix
| Metric / Actual | Predicted Fake (0) | Predicted Real (1) |
| :--- | :--- | :--- |
| **Actual Fake (0)** | 4,647 (True Negative) | 49 (False Positive) |
| **Actual Real (1)** | 21 (False Negative) | 4,263 (True Positive) |

## Classification Report
```
              precision    recall  f1-score   support

    Fake (0)     0.9955    0.9896    0.9925      4696
    Real (1)     0.9886    0.9951    0.9919      4284

    accuracy                         0.9922      8980
   macro avg     0.9921    0.9923    0.9922      8980
weighted avg     0.9922    0.9922    0.9922      8980

```

## Saved Artifacts
- `model.pkl`: Serialized scikit-learn Logistic Regression model
- `vec.pkl`: Serialized TF-IDF vectorizer (max 50,000 features, unigrams + bigrams)
