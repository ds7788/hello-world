__all__ = ['guess_starting_scaling']
from core import *
from nuts2  import NUTS2
from sample import sample
from tuning.scaling import guess_scaling, flatten_samples
from lbfgs import *

def guess_starting_scaling(s=None, model=None, n = 30, na = 15):
    model = modelcontext(model)
    if s is None:
        s = model.test_point

    s0 = guess_scaling(s)
    qp = LBFGSQuadpotential(HessApproxGen(na, 1./s0))

    step = NUTS2(model=model, potential=qp)
    trace  = sample(n, step,s)
    s = trace[-1]

    return s, step.potential
