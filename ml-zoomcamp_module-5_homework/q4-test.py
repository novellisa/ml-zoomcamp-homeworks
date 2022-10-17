#!/usr/bin/env python
# coding: utf-8

# With this notebook we want to communicate with our churn prediction service.

import requests # library for sending requests

url = 'http://localhost:9696/predict'

client = {"reports": 0, "share": 0.245, "expenditure": 3.438, "owner": "yes"}
response = requests.post(url, json=client).json()
print(response)
