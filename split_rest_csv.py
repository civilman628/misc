import random
import csv
import numpy as np



##filelist80=[]
#filelist200=[]

with open('/home/scopeserver/RaidDisk/DeepLearning/mwang/tensorflow/tensorflow/models/image/imagenet/scopestyle_80M_imagelist.txt', 'r') as reader:
    filelist80 = [line.rstrip() for line in reader]

with open('/home/scopeserver/RaidDisk/DeepLearning/mwang/tensorflow/tensorflow/models/image/imagenet/scopestyle_200M_imageid.txt', 'r') as reader:
    filelist200 = [line.rstrip() for line in reader]


difflist=list(set(filelist200)-set(filelist80))
print(len(difflist))

random.shuffle(difflist)

first_part=set(difflist[0:len(difflist)/2])
second_part=set(difflist[len(difflist)/2:len(difflist)])

print(len(first_part))
print(len(second_part))

count=0

with open('/home/scopeserver/RaidDisk/DeepLearning/mwang/data/scopestyle_feature_aug23.csv') as f:
    #reader=csv.reader(f)
    with open('scope_style_80M_id_feature_part1.txt','w') as g:
        w = csv.writer(g)
        for line in f:
            id=line.split('\t')[0]
            if id in first_part:
                print(id)
                s=line.replace(' ','\t')
                print>>g,s
                #first_part.remove(first_part[0])
                #if len(first_part)==0:
                 #   break

            #count+=1

print('done 1')

count=0

with open('/home/scopeserver/RaidDisk/DeepLearning/mwang/data/scopestyle_feature_aug23.csv') as f:
    #reader=csv.reader(f)
    with open('scope_style_80M_id_feature_part2.txt','w') as g:
        w = csv.writer(g)
        for line in f:
            id=line.split('\t')[0]
            if id in second_part:
                print(id)
                s=line.replace(' ','\t')
                print>>g,s
                #second_part.remove(second_part[0])
                #if len(second_part)==0:
                 #   break

            #count+=1