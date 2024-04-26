from flask import Flask,request,render_template,jsonify
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline


application=Flask(__name__)

app=application



@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])

def predict_datapoint():
    if request.method=='GET':
        return render_template('form.html')
    
    else:
        data=CustomData(
            Average_Cost_for_two =float(request.form.get('Average Cost for two')),
            Votes = float(request.form.get('Votes')),
            Has_Table_booking = request.form.get('Has Table booking'),
            Has_Online_delivery= request.form.get('Has Online delivery'),
            Is_delivering_now = request.form.get('Is delivering now'),
            Price_range = request.form.get('Price range'),
            Rating_text = request.form.get('Rating text')
        )
        final_new_data=data.get_data_as_dataframe()
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_new_data)

        results=round(pred[0],2)

        return render_template('results.html',final_result=results)






if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)