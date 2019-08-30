import numpy as np
import scipy.optimize
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from itertools import product, combinations
from time import time

def T(R, t):
    return (np.pad(np.array([[1]]), ((3,0),(3,0)), 'constant') +
           np.pad(R, ((0,1),(0,1)), 'constant') +
           np.pad(t, ((0,1),(3,0)), 'constant'))

def Rx(th):
    s = np.sin(th)
    c = np.cos(th)
    return np.array([
        [1, 0,  0],
        [0, c, -s],
        [0, s,  c]
    ])

def Ry(th):
    s = np.sin(th)
    c = np.cos(th)
    return np.array([
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c]
    ])

def Rz(th):
    s = np.sin(th)
    c = np.cos(th)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])

class Joint:
    def __init__(self, R, limits, t):
        self.R = R
        self.limits = np.deg2rad(limits)
        self.t = np.array(t)[:,np.newaxis]

    def T(self, th):
        return T(self.R(th), self.t)

chain = [
    Joint(Ry, (-119.5, 119.5), (0.00, 0.00, 0.00)), # LShoulderPitch
    Joint(Rz, (- 18.0,  76.0), (0.00, 0.03, 0.00)), # LShoulderRoll
    Joint(Rx, (-119.5, 119.5), (0.06, 0.00, 0.00)), # LElbowYaw
    Joint(Rz, (- 88.5,   2.0), (0.02, 0.00, 0.00))  # LElbowRoll
]

THand = T(np.eye(3), np.array([0.06,0.0,0.0])[:,np.newaxis])

angles = np.zeros(len(chain))

def FK(chain, angles):
    assert len(chain) == len(angles)
    frames = [chain[0].T(angles[0])]
    for i in range(1, len(chain)):
        frames.append(frames[i-1] @ chain[i].T(angles[i]))
    frames.append(frames[-1] @ THand)
    return frames

def IK(chain, angles0, target):
    def residuals(angles):
        frames = FK(chain, angles)
        r = []
        r += list(frames[-3][:3,-1] - target[:,0]) # LElbowYaw
        r += list(frames[-1][:3,-1] - target[:,1]) # Hand
        return r

    mask = np.array([
        [1, 1, 0, 0], # x LElbowYaw
        [0, 1, 0, 0], # y LElbowYaw
        [1, 1, 0, 0], # z LElbowYaw
        [1, 1, 1, 1], # x Hand
        [0, 1, 1, 1], # y Hand
        [1, 1, 1, 1]  # z Hand
    ])

    start = time()
    res = scipy.optimize.least_squares(
        fun=residuals,
        x0=angles0,
        args=(),
        method='trf',
        bounds=([joint.limits[0] for joint in chain], [joint.limits[1] for joint in chain]),
        #jac_sparsity=mask,
        loss='soft_l1',
        ftol=1e-4,
        xtol=(180./np.pi) / 100.,
    )
    print("IK took: %.1fms" % ((time()-start)*1000))
    return res.x if res.success else None



#print(FK(chain, angles)[:3,-1])

initial_pose = np.array([
    [ 0.07,  0.09],
    [ 0.02, -0.03],
    [-0.04, -0.07]
])


def setup_frame(fig, ii, N):
    fig.clf()
    ax = fig.gca(projection='3d')
    ax.set_aspect('equal')
    view = ii/N*20
    ax.view_init(view*2, view)

    # draw cube to create equaivalently scaled axis
    r = [-0.2, 0.2] # region the cube covers
    for s, e in combinations(np.array(list(product(r, r, r))), 2):
        if np.sum(np.abs(s-e)) == r[1]-r[0]:
            ax.plot3D(*zip(s, e), color="w",alpha=0)
    return ax

fig = plt.figure()

N = 150

setup_frame(fig, 0, N)
plt.pause(10.0)

for ii in range(N):

    vive = initial_pose.copy()
    if ii < N/2:
        vive[:,0] += np.array([0.0,0.0,0.05]) * np.sin(ii/(N/2)*2*np.pi)
        vive[:,1] += np.array([0.0,0.0,-0.03]) * np.sin(ii/(N/2)*2*np.pi)
    else:
        vive[:,0] += np.array([0.0,0.0,0.0]) * np.sin((ii-N/2)/(N/2)*2*np.pi)
        vive[:,1] += np.array([0.0,0.07,0.0]) * np.sin((ii-N/2)/(N/2)*2*np.pi)

    angles = IK(chain, angles, vive)

    ax = setup_frame(fig, ii, N)

    frames = FK(chain, angles)
    #print(frames[-2])
    #print(frames[-1])
    #exit()

    coords = np.zeros((3,len(frames)))
    for i in range(len(frames)):
        coords[:,i] = frames[i][:3,-1]

    ax.scatter(vive[0,:], vive[1,:], vive[2,:], c=[0,1], cmap='jet')

    ax.plot3D(coords[0,:], coords[1,:], coords[2,:], 'blue')

    #plt.show()
    plt.pause(0.01)

plt.show()
