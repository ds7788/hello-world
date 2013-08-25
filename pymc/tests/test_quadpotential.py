import itertools as its
from checks import *
from pymc import *
from pymc.lbfgs import *
from numpy import array, inf,dot, cov, eye
import numpy as np
from numpy.linalg import inv


def normal_model(H):
    with Model() as model:
        x = MvNormal('x', 0, H)

    return model

def test_lbfgs():

    d = array([.005, 1, 400])
    H = np.diag(d)
    #model = normal_model(H)

    g = HessApproxGen(0, 1/d)

    check_quad(LBFGSQuadpotential(g), H)

def check_quad(q, H):
    n = 600
    x = np.array([q.random() for _ in range(400)])
    close_to(np.diag(cov(x.T))/ np.diag(H), 1, 3.1 * n**-.5)

