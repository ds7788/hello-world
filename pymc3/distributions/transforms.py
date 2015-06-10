from .dist_math import *
from ..model import FreeRV
import theano

__all__ = ['transform', 'logtransform', 'simplextransform']

class Transform(object):
    """A transformation of a random variable from one space into another."""
    def __init__(self, name, forward, backward, jacobian_det):
        """
        Parameters
        ----------
        name : str
        forward : function 
            forward transformation
        backwards : function 
            backwards transformation
        jacobian_det : function 
            jacobian determinant of the transformation"""
        self.__dict__.update(locals())

    def apply(self, dist):
        return TransformedDistribution.dist(dist, self)

    def __str__(self):
        return name + " transform"

class TransformedDistribution(Distribution):
    """A distribution that has been transformed from one space into another."""
    def __init__(self, dist, transform, *args, **kwargs):
        """
        Parameters
        ----------
        dist : Distribution 
        transform : Transform
        args, kwargs
            arguments to Distribution"""
        forward = transform.forward 
        testval = forward(dist.default())
        
        self.dist = dist
        self.transform_used = transform
        v = forward(FreeRV(name='v', distribution=dist))
        self.type = v.type

        super(TransformedDistribution, self).__init__(v.shape.tag.test_value,
                v.dtype, 
                testval, dist.defaults, 
                *args, **kwargs)


    def logp(self, x):
        return self.dist.logp(self.transform_used.backward(x)) + self.transform_used.jacobian_det(x)

transform = Transform


logtransform = transform("log", log, exp, idfn)


logistic = t.nnet.sigmoid
def logistic_jacobian(x):
    ex = exp(-x)
    return log(ex/(ex +1)**2)

def logit(x): 
    return log(x/(1-x))
logoddstransform = transform("logodds", logit, logistic, logistic_jacobian)


def interval_transform(a, b):
    def interval_real(x):
        r= log((x-a)/(b-x))
        return r

    def real_interval(x):
        r =  (b-a)*exp(x)/(1+exp(x)) + a
        return r

    def real_interval_jacobian(x):
        ex = exp(-x)
        jac = log(ex*(b-a)/(ex + 1)**2)
        return jac

    return transform("interval", interval_real, real_interval, real_interval_jacobian)

class SimplexTransform(Transform): 
    def __init__(self):
        pass

    name = "simplex"
    def forward(self, x):
        return theano.tensor.zeros_like(t.as_tensor_variable(x[:-1]))

    def backward(self, y):
        z = logistic(y)
        yl = concatenate([z, [1]])
        yu = concatenate([[1], 1-z])
        #S,_ = theano.scan(fn=lambda prior_result, s_i: prior_result * s_i, sequences=[yu], outputs_info=t.ones((), dtype='float64'))
        S = t.extra_ops.cumprod(yu)
        x = S * yl
        print (x.tag.test_value)
        return x

    def jacobian_det(self, y): 
        yl = logistic(y)
        yu = concatenate([[1], 1-yl])
        #S,_ = theano.scan(fn=lambda prior_result, s_i: prior_result * s_i, sequences=[yu], outputs_info=t.ones((), dtype='float64'))
        S = t.extra_ops.cumprod(yu)
        return sum(log(S[:-1]) - log(1+exp(yl)) - log(1+exp(-yl)))
simplextransform = SimplexTransform()


"""
def unsimplex(logodds):
    p = logistic(logodds)
    return concatenate([p, 1 - sum(p, keepdims=True)])

simplextransform = transform("simplex",
                             lambda p: logit(p[:-1]),
                             unsimplex,
                             lambda p: sum(logistic_jacobian(p)))
"""
