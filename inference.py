import joblib


class SentimentModel:
    def __init__(self, model):
        self.model = model

    def predict(self, text):
        # Preprocess the input text if necessary
        # For example, you might want to clean or tokenize the text
        # Here, we assume the model can handle raw text input directly

        # Make a prediction using the loaded model
        prediction = self.model.predict([text])

        # Return the prediction result
        return prediction[0]


def load_model(model_path):
    model = joblib.load(model_path)
    return SentimentModel(model)
