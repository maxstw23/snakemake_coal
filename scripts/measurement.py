import numpy as np
from uncertainties import unumpy, ufloat, core

class Measurement:
    def __init__(self, uf, *args):
        """
        if one argument is given, it should be a numpy array of uncertainties variables
        if three arguments are given, they should be numpy arrays of weights, num and num2
        """
        if type(uf) is not np.ndarray:
            raise ValueError('Input should be a numpy array!')
        if len(args) == 0:
            if type(uf[0]) is not core.Variable and type(uf[0]) is not core.AffineScalarFunc:
                raise ValueError('Input should be an array of uncertainties variables!')
            self.uf = uf
            values = unumpy.nominal_values(uf)
            errors = unumpy.std_devs(uf)
            if not np.any(errors):
                self.weights = np.ones(len(values))
            else:
                self.weights = 1. / errors**2
        else:
            raise ValueError('Wrong number of arguments!')
        
    
    def __add__(self, other):
        if not isinstance(other, Measurement):
            raise ValueError('Input should be a Measurement object!')
        if len(self.weights) != len(other.weights):
            raise ValueError('Input arrays should have the same length!')
        
        uf_sum = np.average([self.uf, other.uf], axis=0, weights=[self.weights, other.weights])
        return Measurement(uf_sum)


    def get_measurement(self):
        return self.uf
    