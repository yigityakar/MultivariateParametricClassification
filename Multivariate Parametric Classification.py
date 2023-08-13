#!/usr/bin/env python
# coding: utf-8

# In[273]:


import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[274]:


np.random.seed(421)

class_means= np.array([
    [+0.0,+2.5],
    [-2.5,-2.0],
    [+2.5,-2.0]
   
])

class_covariances = np.array([[[+3.2, +0.0], 
                               [+0.0, +1.2]],
                              [[+1.2, +0.8], 
                               [+0.8, +1.2]],
                              [[+1.2, -0.8], 
                               [-0.8, +1.2]]
                             ])



class_sizes = np.array([120,80,100])


# In[275]:


points1 = np.random.multivariate_normal(class_means[0,:], class_covariances[0,:,:], class_sizes[0])
points2 = np.random.multivariate_normal(class_means[1,:], class_covariances[1,:,:], class_sizes[1])
points3 = np.random.multivariate_normal(class_means[2,:], class_covariances[2,:,:], class_sizes[2])


X = np.vstack((points1, points2,points3))




y = np.concatenate((np.repeat(1, class_sizes[0]), np.repeat(2, class_sizes[1]),np.repeat(3, class_sizes[2])))

plt.figure(figsize = (10, 10))
plt.xlim(-10,10)
plt.plot(points1[:,0], points1[:,1], "r.", markersize = 10)
plt.plot(points2[:,0], points2[:,1], "g.", markersize = 10)
plt.plot(points3[:,0], points3[:,1], "b.", markersize = 10)

plt.show()


# In[276]:


sample_means = np.array([np.mean(X[y == (c + 1)],axis=0) for c in range(0,3)])



a= np.array(X[y == 1]- sample_means[0])

cov = (np.dot(a.transpose(),a))/ class_sizes[0]

                     
b = np.array(X[y == 2]- sample_means[1])
cov2 = (np.dot(b.transpose(),b))/ class_sizes[1]



c= np.array(X[y == 3]- sample_means[2])
cov3= (np.dot(c.transpose(),c))/ class_sizes[2]

                     
sample_covariances= np.array([cov,cov2,cov3])

class_priors= [np.sum(1*(y==c+1)/len(y)) for c in range(0,3)]


print(sample_means)
print()
print(sample_covariances)
print()
print(class_priors)






# In[277]:


c1=0
c2=0
c3=0

s1=[]
s2=[]
s3=[]

prediction_list=[]
for i in X[y==1]:
    score1= (-1)*np.log(math.pi*2) - (0.5)* np.log(np.linalg.det(cov)) + (-0.5) * np.dot(np.dot(np.array([i- sample_means[0]]),np.linalg.inv(cov)), np.array([i- sample_means[0]]).transpose()) + np.log(class_priors[0])   
    score2= (-1)*np.log(math.pi*2) - (0.5)* np.log(np.linalg.det(cov2)) + (-0.5) * np.dot(np.dot(np.array([i- sample_means[1]]),np.linalg.inv(cov2)), np.array([i- sample_means[1]]).transpose()) + np.log(class_priors[1]) 
    score3= (-1)*np.log(math.pi*2) - (0.5)* np.log(np.linalg.det(cov3)) + (-0.5) * np.dot(np.dot(np.array([i- sample_means[2]]),np.linalg.inv(cov3)), np.array([i- sample_means[2]]).transpose()) + np.log(class_priors[2])  
    
    if np.maximum(np.maximum(score2,score3), score1) == score1:
        c1 +=1
        prediction_list.append(1)
    elif np.maximum(np.maximum(score2,score3), score1) == score2:
        c2 +=1
        prediction_list.append(2)
    else:
        c3 +=1
        prediction_list.append(3)
    
    
s1.append(c1)
s2.append(c2)
s3.append(c3)
c1=0
c2=0
c3=0

for i in X[y==2]:
    score1= (-1)*np.log(math.pi*2) - (0.5)* np.log(np.linalg.det(cov)) + (-0.5) * np.dot(np.dot(np.array([i- sample_means[0]]),np.linalg.inv(cov)), np.array([i- sample_means[0]]).transpose()) + np.log(class_priors[0])   
    score2= (-1)*np.log(math.pi*2) - (0.5)* np.log(np.linalg.det(cov2)) + (-0.5) * np.dot(np.dot(np.array([i- sample_means[1]]),np.linalg.inv(cov2)), np.array([i- sample_means[1]]).transpose()) + np.log(class_priors[1]) 
    score3= (-1)*np.log(math.pi*2) - (0.5)* np.log(np.linalg.det(cov3)) + (-0.5) * np.dot(np.dot(np.array([i- sample_means[2]]),np.linalg.inv(cov3)), np.array([i- sample_means[2]]).transpose()) + np.log(class_priors[2])  
    
    if np.maximum(np.maximum(score2,score3), score1) == score1:
        c1 +=1
        prediction_list.append(1)
    elif np.maximum(np.maximum(score2,score3), score1) == score2:
        c2 +=1
        prediction_list.append(2)
    else:
        c3 +=1
        prediction_list.append(3)

s1.append(c1)
s2.append(c2)
s3.append(c3)
 
    
c1=0
c2=0
c3=0

for i in X[y==3]:
    score1= (-1)*np.log(math.pi*2) - (0.5)* np.log(np.linalg.det(cov)) + (-0.5) * np.dot(np.dot(np.array([i- sample_means[0]]),np.linalg.inv(cov)), np.array([i- sample_means[0]]).transpose()) + np.log(class_priors[0])   
    score2= (-1)*np.log(math.pi*2) - (0.5)* np.log(np.linalg.det(cov2)) + (-0.5) * np.dot(np.dot(np.array([i- sample_means[1]]),np.linalg.inv(cov2)), np.array([i- sample_means[1]]).transpose()) + np.log(class_priors[1]) 
    score3= (-1)*np.log(math.pi*2) - (0.5)* np.log(np.linalg.det(cov3)) + (-0.5) * np.dot(np.dot(np.array([i- sample_means[2]]),np.linalg.inv(cov3)), np.array([i- sample_means[2]]).transpose()) + np.log(class_priors[2])  
    
    if np.maximum(np.maximum(score2,score3), score1) == score1:
        c1 +=1
        prediction_list.append(1)
    elif np.maximum(np.maximum(score2,score3), score1) == score2:
        
        c2 +=1
        prediction_list.append(2)
    else:
        c3 +=1    
        prediction_list.append(3)
s1.append(c1)
s2.append(c2)
s3.append(c3)

confusion_matrix= np.array([s1,s2,s3])


print(confusion_matrix)


# In[234]:


plt.figure(figsize = (10, 10))
plt.plot(X[y == 1, 0], X[y == 1, 1], "r.", markersize = 10)
plt.plot(X[y == 2, 0], X[y == 2, 1], "g.", markersize = 10)
plt.plot(X[y == 3, 0], X[y == 3, 1], "b.", markersize = 10)
plt.plot(X[prediction_list != y, 0], X[prediction_list != y, 1], "ko", markersize = 12, fillstyle = "none")


plt.show()


# In[ ]:




