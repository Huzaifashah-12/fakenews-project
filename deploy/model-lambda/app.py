import json
import joblib
import os

# Load model and vectorizer at cold start
MODEL_PATH = os.environ.get("MODEL_PATH", "model.pkl")
VEC_PATH = os.environ.get("VEC_PATH", "vec.pkl")

model = None
vec = None

def load_artifacts():
    global model, vec
    if model is None:
        model = joblib.load(MODEL_PATH)
    if vec is None:
        vec = joblib.load(VEC_PATH)

def lambda_handler(event, context):
    try:
        load_artifacts()
        
        # Parse body
        body = event.get("body", "{}")
        if isinstance(body, str):
            body = json.loads(body) if body else {}

        text = body.get("text", "")
        if not text:
            return {
                "statusCode": 400,
                "headers": {
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*"
                },
                "body": json.dumps({"error": "Field 'text' is required."})
            }

        # Vectorize and predict
        features = vec.transform([text])
        prediction = int(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0].tolist()

        label = "Real" if prediction == 1 else "Fake"
        confidence = probabilities[prediction]

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps({
                "label": label,
                "prediction": prediction,
                "confidence": round(confidence, 4),
                "probabilities": {
                    "fake": round(probabilities[0], 4),
                    "real": round(probabilities[1], 4)
                }
            })
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps({"error": str(e)})
        }
