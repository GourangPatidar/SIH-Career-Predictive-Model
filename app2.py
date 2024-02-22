import pandas as pd
from flask import Flask, jsonify, request, render_template
from src.pipeline.prediction_pipeline import PredictPipeline
import openai

# Replace 'YOUR_API_KEY' with your actual OpenAI API key
api_key = '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'

# Initialize the OpenAI API client
openai.api_key = api_key

# Send a message to ChatGPT


def send_message(message, model="gpt-3.5-turbo"):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": message},
        ],
        max_tokens=50
    )
    return response


app = Flask(__name__)


@app.route('/')
def home_page():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    json = request.json
    query_df = pd.DataFrame(json)
    predict_pipeline = PredictPipeline()
    prediction = predict_pipeline.predict(query_df)

    # Use the prediction as input to the chat endpoint
    user_message = "how this is a best career for you: " + ", ".join(map(str, prediction))
    response = send_message(user_message)

    return jsonify({
        "prediction": list(prediction),
        "assistant_response": response['choices'][0]['message']['content']
    })


if __name__ == "__main__":
    app.run(debug=True)
