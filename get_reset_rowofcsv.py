import random
import csv
import numpy as np

#rows= random.sample(xrange(2400000),800000)
#index= np.sort(rows)
#index= sorted(rows)

idlist=[]

with open('/home/scopeserver/RaidDisk/DeepLearning/mwang/tensorflow/tensorflow/models/image/imagenet/output.txt') as f:
    for line in f:
        id=line.split('\t')[0]
        if id=='\n':
            continue
        else:
            idlist.append(id)

with open('scopestyle_80M.txt','w') as file:
    for line in idlist:
        print>>file, line