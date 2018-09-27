import numpy as np 
import h5py
import scipy

distarray=[[1,0],[2,0],[3,0],[4,0],[5,0]]
max_distance=[162.4133,150.2330,136.4027,122,3726,120.5494]


dict=h5py.File('scopestyle_all_whole_person_mean_var_id_5clusters.mat')
dict2=h5py.File('single_feature_test.mat')
v1=np.array(dict2['v1']).T

mean1=np.array(dict['m1']).T
mean2=np.array(dict['m2']).T
mean3=np.array(dict['m3']).T
mean4=np.array(dict['m4']).T
mean5=np.array(dict['m5']).T

clustermean=np.concatenate((mean1,mean2,mean3,mean4,mean5),0)

s1=np.array(dict['s1']).T
s2=np.array(dict['s2']).T
s3=np.array(dict['s3']).T
s4=np.array(dict['s4']).T
s5=np.array(dict['s5']).T

invsere_s=[s1,s2,s3,s4,s5]

def distance(a,b):
    return np.sqrt(np.sum((a-b)**2))


#feature = np.random.rand(1,2048)
#print(max(feature))



# feature is 1*2048 float array
def score(feature):    
    
    for i in range(0,5):
        distarray[i][1]=distance(feature,clustermean[i,:])

    print(distarray)
    sortedarray=sorted(distarray,key=lambda x:x[1])
    print(sortedarray)

    cluster_index=sortedarray[0][0]
    print(cluster_index)
    #print(max(clustermean[cluster_index-1,:]))
    de_mean=feature-clustermean[cluster_index-1,:]
    s_inverse=invsere_s[cluster_index-1]
    dist=np.sqrt(np.dot(np.dot(de_mean,s_inverse),de_mean.T))

    if dist > max_distance[cluster_index-1]:
        return -1
    else:
        test = 1- dist/max_distance[cluster_index-1]
        return test


a = score(v1)

