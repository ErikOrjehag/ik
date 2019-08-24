import numpy as np
import scipy.optimize
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from itertools import product, combinations

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

LShoulderPitch = Joint(Ry, (-119.5, 119.5), (0.00, 0.00, 0.00))
LShoulderRoll =  Joint(Rz, (- 18.0,  76.0), (0.00, 0.03, 0.00))
LElbowRoll =     Joint(Rz, (- 88.5,   2.0), (0.12, 0.00, 0.00))
LElbowYaw =      Joint(Rx, (-119.5, 119.5), (0.00, 0.00, 0.00))
LWristYaw =      Joint(Rx, (-104.5, 104.5), (0.12, 0.00, 0.00))

chain = [
    LShoulderPitch,
    LShoulderRoll,
    LElbowRoll,
    LElbowYaw,
    LWristYaw
]

#angles = np.deg2rad([10.0,-5.0,-20.0,0.0,0.0])
angles = np.zeros(5)

def FK(chain, angles):
    assert len(chain) == len(angles)
    Tee = np.eye(4)
    for i in range(len(chain)):
        Tee = Tee @ chain[i].T(angles[i])
    return Tee

def IK(chain, angles0, target):
    def objective(angles):
        #print("target", target)
        #print("fk", FK(chain, angles)[:3,-1])
        return np.sum((target - FK(chain, angles)[:3,-1])**2)
    res = scipy.optimize.minimize(
        fun=objective,
        x0=angles0,
        args=(),
        method='L-BFGS-B',
        tol=1.48e-08,
        options={ 'maxiter': 50 },
        bounds=[joint.limits for joint in chain]
    )
    return res.x if res.success else None

#print(FK(chain, angles)[:3,-1])

hand = np.array([0.15, -0.03, -0.04])

fig = plt.figure()


ii = 10
while ii:
    ii -= 1
    th = IK(chain, angles, hand)
    hand += np.array([0.0,0.0,0.01])

    fig.clf()
    ax = fig.gca(projection='3d')
    ax.set_aspect('equal')

    # draw cube to create equaivalently scaled axis
    r = [-0.3, 0.3] # region the cube covers
    for s, e in combinations(np.array(list(product(r, r, r))), 2):
        if np.sum(np.abs(s-e)) == r[1]-r[0]:
            ax.plot3D(*zip(s, e), color="w",alpha=0)

    coords = np.zeros((3,len(angles)))
    for i in range(len(angles)):
        coords[:,i] = FK(chain[:i+1], th[:i+1])[:3,-1]

    ax.plot3D(coords[0,:], coords[1,:], coords[2,:], 'blue')

    plt.pause(0.01)
