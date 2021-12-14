import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import pickle

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

dt_model = pickle.load(open('dt_crime_prediction.pkl', 'rb'))
gnb_model = pickle.load(open('gnb_crime_prediction.pkl', 'rb'))
knn_model = pickle.load(open('knn_crime_prediction.pkl', 'rb'))
mlpc_model = pickle.load(open('mlpc_crime_prediction.pkl', 'rb'))
svm_model = pickle.load(open('svm_crime_prediction.pkl', 'rb'))

@app.route('/predict',methods=['POST'])
@cross_origin()
def predict_api():
    # data =([[25,33,27,0,1,5]])
    data = request.get_json(force=True)
    dt_pred = dt_model.predict([np.array(list(data.values()))])
    dt_proba = dt_model.predict_proba([np.array(list(data.values()))])
    gnb_pred = gnb_model.predict([np.array(list(data.values()))])
    gnb_proba = gnb_model.predict_proba([np.array(list(data.values()))])
    knn_pred = knn_model.predict([np.array(list(data.values()))])
    knn_proba = knn_model.predict_proba([np.array(list(data.values()))])
    mlpc_pred = mlpc_model.predict([np.array(list(data.values()))])
    mlpc_proba = mlpc_model.predict_proba([np.array(list(data.values()))])
    svm_pred = svm_model.predict([np.array(list(data.values()))])
    svm_proba = svm_model.predict_proba([np.array(list(data.values()))])

    law_violated = {
        0:'ACT TO INCREASE PENALTIES AGAINST ILLEGAL NUMBERS GAMES', 1:'ANTI-VIOLENCE AGAINST WOMEN AND THEIR CHILDREN ACT OF 2004', 2:'CARNAPPING', 
        3:'COMPREHENSIVE DANGEROUS DRUGS ACT OF 2002', 
        4:'HOMICIDE', 
        5:'MURDER', 
        6:'RAPE', 
        7:'ROBBERY', 
        8:'THEFT'
    }

    dt_confidence = dt_proba[0][int(dt_pred[0])]
    dt_prediction = law_violated[int(dt_pred[0])]

    gnb_confidence = gnb_proba[0][int(gnb_pred[0])]
    gnb_prediction = law_violated[int(gnb_pred[0])]

    knn_confidence = knn_proba[0][int(knn_pred[0])]
    knn_prediction = law_violated[int(knn_pred[0])]

    mlpc_confidence = mlpc_proba[0][int(mlpc_pred[0])]
    mlpc_prediction = law_violated[int(mlpc_pred[0])]

    svm_confidence = svm_proba[0][int(svm_pred[0])]
    svm_prediction = law_violated[int(svm_pred[0])]

    data = { 
        "decision tree":{
            "prediction":dt_prediction,
            "confidence":f"{dt_confidence:.0%}"
        }, 
        "gaussian naive bayes":{
            "prediction":gnb_prediction,
            "confidence":f"{gnb_confidence:.0%}"
        }, 
        "k-nearest neighbor":{
            "prediction":knn_prediction,
            "confidence":f"{knn_confidence:.0%}"
        }, 
        "multi-layer perceptron":{
            "prediction":mlpc_prediction,
            "confidence":f"{mlpc_confidence:.0%}"
        }, 
        "support vector machine":{
            "prediction":svm_prediction,
            "confidence":f"{svm_confidence:.0%}"
        },
    } 
    return jsonify(data)


if __name__ == "__main__":
    app.run(debug=True)