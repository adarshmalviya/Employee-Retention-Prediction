from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('hr_analytics_model.pkl', 'rb'))
sc_scale = pickle.load(open('sc_scaler.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('employee.html')


@app.route('/employee', methods=["GET", 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('employee.html')

    if request.method == 'POST':
        prediction = "Testing Going On"
        try:

            salary={"Low": 1, "Medium": 2, "High": 3}

            department=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            depart_dict={"IT": 0, "R & D": 1, "Accounting": 2, "Hr": 3, "Management": 4,"Marketing": 5, "Product Manager": 6, "Sales": 7, "Support": 8, "Technical": 9}
            department[depart_dict[request.form["department"]]]=1
            
            features = [[float(request.form["satisfaction"]), float(request.form["last_eval"]), int(request.form["project"]), int(request.form["hours"]), int(request.form["time"]), int(request.form["accident"]), int(request.form["promotion"]), salary[request.form["salary"]]]]
            features[0] += department

            int_features = sc_scale.transform(features)
            if model.predict(np.array(int_features))[0] == 0:
                prediction = "Employee will continue to work with the Company"
            else:
                prediction = "Employee is going to Leave the Company"
        except:
            prediction="Invalid Data"

        return render_template('employee.html', prediction_text=prediction)

if __name__ == "__main__":
    app.run()
