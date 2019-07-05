from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow import keras
app = Flask(__name__)
print(__name__)

@app.route('/', methods=['POST', 'GET'])

def home():
    if request.method == 'GET':
        return render_template('home.html', message = "")
    else:
        print('here')
        review = request.form['review']
        from classify import classify
        result = classify(review)
        return render_template('home.html', message = f"{result}")

if __name__ == '__main__':
    print('loading model')
    print('loaded')
    app.run(debug=True)


