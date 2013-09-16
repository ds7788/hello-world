from checks import *
from pymc import *
from pymc.lbfgs import *
from pymc.step_methods.quadpotential import quad_potential
from numpy import array, inf,dot, cov, eye
import numpy as np
from numpy.linalg import inv, eigh


def normal_model(H):
    with Model() as model:
        x = MvNormal('x', 0, H)

    return model

def test_lbfgs():
    n = 5

    H = array([[.25, .25, .5],
               [.25,    1,    0],
               [.5,    0,  4]])
    d = np.diag(H)

    model = normal_model(H)

    g = HessApproxGen(n, 1/d)


    logp = model.logpc
    dlogp = model.dlogpc()

    for _ in range(n):
        x = random.normal()
        g.update(x,
                 logp({'x' : x}),
                 dlogp({'x' : x}))

    check_quad(LBFGSQuadpotential(g), H)

def check_quad(potential, H):
    ref = quad_potential(H, False, False)

    n = 100

    for _ in range(n):
        x = ref.random()
        close_to(ref.velocity(x), potential.velocity(x), 1e-3)
        close_to(ref.energy(x), potential.energy(x), 1e-3)

    P, V = eigh(H)

    x = np.array([q.random() for _ in range(n)])

    P_inv = V.dot(C).dot(V.T)
    close_to(P, np.diag(P_inv), 1e-3)

    #close_to(np.diag(cov(x.T))/ np.diag(H), 1, 3.1 * n**-.5)





