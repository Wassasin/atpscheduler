# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 11:24:51 2014

@author: mathijsvos
"""

import matplotlib.pyplot as plt
import numpy as np

trainData = '../orig/MLiP_train' 
testData = []
with open(trainData,'r') as IS:
    firstLine = True
    for line in IS:
        if firstLine:
            firstLine = False
            continue
        tmp = (line.strip()).split('#')
        pName = tmp[0]
        pFeatures = [float(x) for x in tmp[1].split(',')]
        #testData.append((pName,pFeatures))
        testData.append(pFeatures)

plt.close('all')

def visualizeData(d):
    print d.shape
    N,M = d.shape
    plt.figure()
    for i in range(0, M):
        plt.subplot(4,M/4+1,i+1)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.boxplot(d[:,i])
        plt.title(str(i))
    
    #plt.tight_layout()
    plt.show()
    
    plt.figure()
    for i in range(0, M):
        plt.subplot(4,M/4+1,i+1)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.hist(d[:,i])
        plt.title(str(i))
    
    #plt.tight_layout()
    plt.show()

data = np.matrix(testData)
visualizeData(data)

# Remove outliers
#upperBounds = np.matrix([1.5*1e4, 0.75*1e6, 0.75*1e6, 1*1e6, 0.3*1e7, 1*1e3, 0.3*1e6, 0.3*1e4, 0.75*1e6, 0.5*1e5, 2.5*1e3, 0.3*1e6, 1*1e3, 0.2*1e6, 0.2*1e6, 1e10, 1e10, 3*1e1, 1*1e1, 3*1e4, 1e10, 1.5])
#deleteRows = []
#N,M=data.shape
#for i in range(0, N):
#    if((data[i,:] > upperBounds).any()):
#        deleteRows.append(i)
#        
#data = np.delete(data, deleteRows, axis=0)
#visualizeData(data)

# Remove boring attributes
#data = data[:,[14, 15, 16, 17, 19]]
#visualizeData(data)