import bentoml

from bentoml.io import JSON

from bentoml.io import NumpyNdarray

model_ref = bentoml.sklearn.get("mlzoomcamp_homework:latest")

model_runner = model_ref.to_runner()

# svc stands for "service"
svc = bentoml.Service("credit_risk_classifier", runners = [model_runner])

@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def classify(vector):
    prediction = model_runner.predict.run(vector)
    print(prediction)
    
    result = prediction[0]
    
    return result
