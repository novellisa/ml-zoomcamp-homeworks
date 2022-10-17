from flask import Flask
from flask import request
from flask import jsonify
import pickle

model_file = 'model1.bin'

with open(model_file, 'rb') as f_in:

    model = pickle.load(f_in)

dv_file = 'dv.bin'

with open(dv_file, 'rb') as f2_in:

    dv = pickle.load(f2_in)

app = Flask('churn')

@app.route('/predict', methods = ['POST'])

def predict():

    customer = request.get_json() # it will return the body of our request as a python dictionary assuming that the request was a .json

    X = dv.transform([customer])
    y_pred =  model.predict_proba(X)[0, 1]
    churn = y_pred >= 0.5

    # our response is also .json, so we need to prepare it as a .json file
    result = {

        'card_probability': float(y_pred),
        'card' : bool(churn)


    }

    return jsonify(result)

# ok now we make the app run, and we specify the debug mode
app.run(debug = True, host = '0.0.0.0', port = 9696)

# __main__ is executed only when we execute ping.py
if __name__ == "__main__":

    app.run(debug = True, host = '0.0.0.0', port = 9696)
