from checks import *
from pymc import *
from pymc.lbfgs import *
from pymc.step_methods.quadpotential import quad_potential
from numpy import array, inf,dot, cov, eye
import numpy as np
from numpy.linalg import inv, eigh


def normal_model(H):
    with Model() as model:
        x = MvNormal('x', 0, H, shape = 3)

    return model


H_full = array([[.25, .1, .5],
            [.1,    1,    0],
            [.5,    0,  4]])

H_diag = np.diag(H_full)
H_diagonal= np.diag(H_diag)


def test_lbfgs():
    check_lbfgs(H_diagonal, 0)
    check_lbfgs(H_diagonal, 1)
    check_lbfgs(H_diagonal, 5)
    check_lbfgs(H_diagonal, 20)

    check_lbfgs(H_full, 15)


def check_lbfgs(H, n):

    model = normal_model(H)
    d = np.diag(H)

    g = HessApproxGen(n, 1/d)


    logp = model.logpc
    dlogp = model.dlogpc()

    ref = quad_potential(H, False, False)
    for _ in range(n):
        x = ref.random()
        g.update(x,
                 logp({'x' : x}),
                 dlogp({'x' : x}))

    pot = LBFGSQuadpotential(g)
    #print pot.lbfgs.lbfgs.S.A0

    check_quad(pot, H)

def test_elemwise():
    P = quad_potential(H_diag, False, False)

    check_quad(P, np.diag(H_diag))


def check_quad(potential, H):
    ref = quad_potential(H, False, False)

    n = 3000

    for i in range(20):
        x = ref.random()
        name = "trial: " + str(i) + " x: " + str(x) 
        rel_close_to(ref.velocity(x) , potential.velocity(x), 1e-2, name)
        rel_close_to(ref.energy(x), potential.energy(x), 1e-2, name)

    P, V = eigh(H)

    x = np.array([potential.random() for _ in range(n)])
    C = cov(x.T)
    P_inv = V.dot(C).dot(V.T)

    close_to(P, np.diag(P_inv), 1e-1*P)

    #close_to(np.diag(cov(x.T))/ np.diag(H), 1, 3.1 * n**-.5)

