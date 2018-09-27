# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 14:32:40 2018

@author: prasann
"""

import json
import numpy as np
import tensorflow as tf

with open('health.json') as json_file:
  x_data = json.load(json_file)
  #print(x_data[2])

items = []
for item in x_data:
    items.append(item['fields']['value'])
print(len(items))
#print(x_data)
config = json.loads(open('health.json').read())

#print(config['fields']['value'])

    