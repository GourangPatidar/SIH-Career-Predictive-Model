import pandas as pd 
from flask import Flask, jsonify , request , render_template
import pickle
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline
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
        ]
    )

    return response




app=Flask(__name__)


@app.route('/')
def home_page():
    return render_template('index.html')




@app.route("/predict" , methods=['POST'])
def predict():
    json=request.json
    query_df=pd.DataFrame(json)
    predict_pipeline = PredictPipeline()
    prediction = predict_pipeline.predict(query_df)
    return jsonify({"prediction" :list(prediction)})


@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message')

        if not user_message:
            return jsonify({"error": "Please provide a 'message' field in the request."})

        response = send_message(user_message)
        return jsonify({"assistant_response": response['choices'][0]['message']['content']})
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('form.html')

    else:
        data = CustomData(

            interest=request.form.get('interest'),
            value=request.form.get('value'),
            skill=request.form.get('skill'),
            personality=request.form.get('personality'),

        )
        json = request.json
        query_df = pd.DataFrame(json)
        predict_pipeline = PredictPipeline()
        prediction = predict_pipeline.predict(query_df)
        return jsonify({"prediction": list(prediction)})
        

if __name__=="__main__":
    app.run(debug=True)