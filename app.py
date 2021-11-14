# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 14:02:32 2021

@author: 
"""

#svd_recommendation(Tittle_ref = title, Author_ref = author, ISBN=isbn)

from flask import Flask, request, url_for, redirect, render_template, jsonify 
import pickle
import os
import numpy as np
import pandas as pd

path="C:/Users/cnb/Desktop/datasentics"
os.chdir(path)

#import model
from model import svd_recommendation 


app = Flask(__name__)
port = int(os.getenv("PORT", 9099))

cols  =['title', 'author', 'isbn']

@app.route('/') 
def home():
    return render_template("index.html") 

@app.route('/Recommend', methods=['POST']) 
def Recommend():
    
  title = request.form['title']
  author = request.form['author']
  predictions = svd_recommendation(Tittle_ref =title, Author_ref = author, include=True)
  predictions = predictions.drop(['corr'], axis=1)
  predictions.index = np.arange(1, len(predictions)+1)
  return render_template('response.html',prediction=predictions)

if __name__ == '__main__':
    app.run(debug=True)