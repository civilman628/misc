import t_sne_bhcuda.bhtsne_cuda as tsne_bhcuda
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"]="0"

perplexity = 50.0
theta = 0.5
learning_rate = 200.0
iterations = 2000
gpu_mem = 0.8


data_for_tsne = np.random.rand(100,100)

t_sne_result = tsne_bhcuda.t_sne(samples=data_for_tsne, files_dir='/home/scopeserver/RaidDisk/DeepLearning/mwang/models/t_sne_bhcuda/',
                        no_dims=2, perplexity=perplexity, eta=learning_rate, theta=theta,
                        iterations=iterations, gpu_mem=gpu_mem, randseed=-1, verbose=2)
t_sne_result = np.transpose(t_sne_result)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(t_sne_result[0], t_sne_result[1])