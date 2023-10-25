import pandas as pd 
from flask import Flask, jsonify , request , render_template
import pickle
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline

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


'''@app.route('/predict', methods=['GET', 'POST'])
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
        '''

if __name__=="__main__":
    app.run(debug=True)