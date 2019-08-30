import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt


anchors = np.array([
    [23,-25,-25, 35],
    [25, 10,-25,-15]
])



#fig = plt.figure()
plt.plot(np.hstack((anchors[0,:],anchors[0,0])), np.hstack((anchors[1,:],anchors[1,0])))
plt.scatter(anchors[0,:], anchors[1,:])

plt.show()
