from flask import Flask, render_template, url_for, request
import joblib
import sys
sys.path.append("./")
from preprocessing import jieba_cut_text
import os

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    path = r".\model"
    le = joblib.load(os.path.join(path, "labelr.pkl"))
    tfidf = joblib.load(os.path.join(path, "tfidf.pkl"))
    clf = joblib.load(os.path.join(path, "logi.pkl"))

    with open(r".\model\stopwords.txt", 'r', encoding="utf-8") as fp:
        stopw = fp.read().split(" ")
    if request.method == 'POST':
        message = request.form['message']
        data = [jieba_cut_text(message, stopw)]
        vect = tfidf.transform(data)
        my_prediction = clf.predict(vect)
    return render_template('result.html', prediction=my_prediction)


if __name__ == '__main__':
    app.run(debug=True)
