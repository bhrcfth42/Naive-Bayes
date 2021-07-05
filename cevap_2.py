# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 10:42:39 2020

@author: fatih
"""

from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd


training=pd.read_csv("naive_bayes_training.csv")
test=pd.read_csv("naive_bayes_test.csv")

x_train=training.drop("Çıkış",axis=1)
y_train=training.loc[:,"Çıkış"]
x_test= test

model=GaussianNB()

cikti = model.fit(x_train, y_train).predict(x_test)

print("Test Kısmının sırası ile çıktısı ",cikti)

pd.DataFrame(cikti).to_csv("sonuc.csv",header=["Çıkış"],index=None)