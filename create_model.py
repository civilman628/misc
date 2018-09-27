import os.path, glob, os
import gzip
import pickle
import re
import sys
import tarfile
import Image
import numpy as np
import random
from six.moves import urllib
#from scipy.special import logsumexp


train=np.load('./data_file/cifar-100-python/train')
test =np.load('./data_file/cifar-100-python/test')

cifar_train=train['data']
cifar_test=test['data']

print(cifar_train.shape)
print(cifar_test.shape)

random.seed(123)
random.shuffle(cifar_train)

training=cifar_train[0:10000,:]/255.0
validation=cifar_train[10000:20000,:]/255.0

print(training.shape)
print(validation.shape)

theta_list = [0.05,0.08,0.1,0.2,0.5,1.0,1.5,2.0]

theta=0.05

sum_d=0.0
sum_k=np.zeros(shape=(10000,1))
sum_m=0.0

#result= np.matmul(validation,np.transpose(training))
#print(result.shape)

for m in range(10000):
    for k in range(10000):
        print(k)
        square_vec=np.square(validation[m,:]-training[k,:])
        print(np.max(square_vec))
        print(np.min(square_vec))
        sum_d=np.sum(square_vec)
        print(sum_d) 
        print(np.log(1/10000.0))
        print(-sum_d/(2.0*theta*theta))
        print(-3072*0.5*np.log(2.0*np.pi*theta*theta))
        print('---')


        sum_k[k]=np.log(1/10000.0)-sum_d/(2.0*theta*theta)-3072*0.5*np.log(2.0*np.pi*theta*theta)
        #sum_k= sum_k+np.exp(np.log(1/10000.0)-sum_d/(2.0*theta*theta)-3072*0.5*np.log(2.0*np.pi*theta*theta))
        #print(sum_k)
        #for d in range(3072):
         #   sum_d=sum_d-np.square(validation[m,d]-training[k,d])/(2* np.square(0.05))-0.5* np.log(2*np.pi*np.square(0.05))
    max_k=np.max(sum_k,axis=0)
    temp=sum_k-max_k
    
    #sum_m=max_k+np.log(temp)
    sum_m= max_k+np.logaddexp(temp[:])#sum_k[0:10000])
    print(sum_m)

result=sum_m/10000.0
print(result)