#!/usr/bin/env python
# coding: utf-8

import pickle

model_file = 'model1.bin'

with open(model_file, 'rb') as f_in:

    model = pickle.load(f_in)

dv_file = 'dv.bin'

with open(dv_file, 'rb') as f2_in:

    dv = pickle.load(f2_in)

client = {"reports": 0, "share": 0.001694, "expenditure": 0.12, "owner": "yes"}

X = dv.transform([client])
y_pred = model.predict_proba(X)[0, 1]
print(y_pred)
