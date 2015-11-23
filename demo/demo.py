"""
Created on Mar 17, 2012

@author: Amy X Zhang
axz@mit.edu 
http://people.csail.mit.edu


Demo of OPTICS Automatic Clustering Algorithm
https://github.com/amyxzhang/OPTICS-Automatic-Clustering

"""

import numpy as np
import matplotlib.pyplot as plt
import OpticsClusterArea as OP
from itertools import *
import AutomaticClustering as AutoC

# generate some spatial data with varying densities
np.random.seed(0)

n_points_per_cluster = 250

X = np.empty((0, 2))
X = np.r_[X, [-5,-2] + .8 * np.random.randn(n_points_per_cluster, 2)]

X = np.r_[X, [4,-1] + .1 * np.random.randn(n_points_per_cluster, 2)]

X = np.r_[X, [1,-2] + .2 * np.random.randn(n_points_per_cluster, 2)]

X = np.r_[X, [-2,3] + .3 * np.random.randn(n_points_per_cluster, 2)]

X = np.r_[X, [3,-2] + 1.6 * np.random.randn(n_points_per_cluster, 2)]

X = np.r_[X, [5,6] + 2 * np.random.randn(n_points_per_cluster, 2)]


#plot scatterplot of points

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(X[:,0], X[:,1], 'b.', ms=2)

plt.savefig('Graph.png', dpi=None, facecolor='w', edgecolor='w',
    orientation='portrait', papertype=None, format=None,
    transparent=False, bbox_inches=None, pad_inches=0.1)
plt.show()



#run the OPTICS algorithm on the points, using a smoothing value (0 = no smoothing)
RD, CD, order = OP.optics(X,9)

RPlot = []
RPoints = []
        
for item in order:
    RPlot.append(RD[item]) #Reachability Plot
    RPoints.append([X[item][0],X[item][1]]) #points in their order determined by OPTICS

#hierarchically cluster the data
rootNode = AutoC.automaticCluster(RPlot, RPoints)

#print Tree (DFS)
AutoC.printTree(rootNode, 0)

#graph reachability plot and tree
AutoC.graphTree(rootNode, RPlot)

#array of the TreeNode objects, position in the array is the TreeNode's level in the tree
array = AutoC.getArray(rootNode, 0, [0])

#get only the leaves of the tree
leaves = AutoC.getLeaves(rootNode, [])

#graph the points and the leaf clusters that have been found by OPTICS
fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(X[:,0], X[:,1], 'y.')
colors = cycle('gmkrcbgrcmk')
for item, c in zip(leaves, colors):
    node = []
    for v in range(item.start,item.end):
        node.append(RPoints[v])
    node = np.array(node)
    ax.plot(node[:,0],node[:,1], c+'o', ms=5)

plt.savefig('Graph2.png', dpi=None, facecolor='w', edgecolor='w',
    orientation='portrait', papertype=None, format=None,
    transparent=False, bbox_inches=None, pad_inches=0.1)
plt.show()
