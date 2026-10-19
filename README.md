# Fake News and Misinformation Detector

An end-to-end Machine Learning pipeline and serverless deployment for detecting fake and misleading news articles using TF-IDF and Logistic Regression.

## Project Structure

```text
├── code/
│   └── train.py               # ML training script
├── deploy/
│   └── model-lambda/          # AWS Lambda Docker container deployment
│       ├── Dockerfile
│       ├── app.py
│       └── requirements.txt
├── images/                    # Visual assets and screenshots
├── docs/                      # Documentation and reports
│   └── training_report.md
├── .gitignore
├── README.md
```

## Model Details
- **Architecture**: TF-IDF (unigrams + bigrams, 50,000 max features) + Logistic Regression
- **Accuracy**: ~99.2%
- **Frameworks**: scikit-learn (1.5.0), joblib, numpy
