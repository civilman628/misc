import random
import csv
import numpy as np

rows= random.sample(xrange(2400000),800000)
#index= np.sort(rows)
index= sorted(rows)

count=0

with open('/home/scopeserver/RaidDisk/DeepLearning/mwang/data/scopestyle_feature_aug23.csv') as f:
    #reader=csv.reader(f)
    with open('output.txt','w') as g:
        w = csv.writer(g)
        for line in f:
            if count==index[0]:
                print(count)
                s=line.replace(' ','\t')
                print>>g,s
                index.remove(index[0])
                if len(index)==0:
                    break

            count+=1