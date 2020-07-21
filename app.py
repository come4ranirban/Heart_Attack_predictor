from flask import Flask,request,render_template, render_template_string
import pickle
from sklearn.preprocessing import StandardScaler
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('predict_heart_attack.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods= ['POST'])
def predict():
    
    if request.method == 'POST':
        age = request.form['age']
        sex = request.form['sex']
        cp = request.form['cp']
        trestbps = request.form['trestbps']
        chol = request.form['chol']
        fbs = request.form['fbs']
        restcg = request.form['restecg']
        thalach = request.form['thalach']
        exang = request.form['exang']
        oldpeak = request.form['oldpeak']
        slope= request.form['slope']
        ca = request.form['ca']
        thal = request.form['thal']
        
        variables = [[age,sex,cp,trestbps,chol,fbs,restcg,thalach,exang,oldpeak,slope,ca,thal]]
        df = pd.DataFrame(variables, columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
       'exang', 'oldpeak', 'slope', 'ca', 'thal'])
        
        
        if(model.predict(df) == 1):
            return render_template('index.html',out="more chance of heart attack")
        else:
            return render_template('index.html',out="lessance of heart attack")
    

if __name__ == '__main__':
    app.run()