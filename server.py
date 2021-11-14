from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
app = Flask(__name__)
CORS(app)

import sys
import pickle


CANCER_MODEL = int(1)
HEART_MODEL = int(2)
DIABETES_MODEL = int(3)
LIVER_MODEL = int(4)

number_of_features_map = {
    CANCER_MODEL: int(30),
    HEART_MODEL: int(13),
    DIABETES_MODEL: int(8),
    LIVER_MODEL: int(10)
}

model_name_map = {
    CANCER_MODEL: "cancer_model.sav",
    HEART_MODEL: "heart_model.sav",
    DIABETES_MODEL: "diabetes_model.sav",
    LIVER_MODEL: "liver_model.sav"
}

# API route
@app.route('/data', methods=['POST'])
@cross_origin(origin='*')
def postTest():
    a = request.get_json(force=True)
    if not a:
        return jsonify({'msg': 'Missing JSON'}), 400

    requested_model = a.get('id')
    if not requested_model:
        return jsonify({'msg': 'ID is missing'}), 400
    
    features = a.get('features')
    if not features or len(features) != number_of_features_map[requested_model]:
        return jsonify({'msg': 'Features are missing or are Invalid'}), 400

    print('ID: ',requested_model)
    for i in range(number_of_features_map[requested_model]):
        features[i] = float(features[i])
    print('Features: ',features)
    

    model_name = model_name_map[requested_model]
    print(model_name)

    clf = pickle.load(open(model_name, 'rb'))
    features = [features]
    pred = clf.predict(features)
    prob = clf.predict_proba(features)
    print('Prediction: ',pred[0])
    print('Probability: ',prob[0])
    print('--------------------------------------')
    return jsonify({'DIAG': str(pred[0]), 'PROB_NEG': str(prob[0][0]), "PROB_POS": str(prob[0][1])}), 200

if __name__ == "__main__":
    app.run(debug=True)