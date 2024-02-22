from flask import Flask, request, render_template
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application


@app.route('/')
def home_page():
    return render_template('index.html')


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
        final_new_data = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(final_new_data)

        results = pred[0]

        return render_template('results.html', final_result=results)


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
