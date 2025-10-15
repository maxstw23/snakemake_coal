import numpy as np
import uncertainties


def add_helper(v, c, s):
    """
    Expect reshaped input.
    :param v: Values. Should be of shape (k, n) where k runs over
              profiles to add and n is the number of bin, same below.
    :param c: Counts
    :param s: Standard deviation of the y values.
    :return: Added v, c, and s
    """
    cc = np.sum(c, axis=0)
    vv = np.divide(np.sum(v * c, axis=0), cc, out=np.zeros(np.shape(cc)), where=cc != 0)
    E = np.sum(c * (s ** 2 + v ** 2), axis=0)
    ss = (np.divide(E, cc, out=np.zeros(np.shape(cc)), where=cc != 0) - vv ** 2) ** 0.5
    return vv, cc, ss


class SimpleProfile:
    def __init__(self, values, counts, std_devs, edges_or_bin_centers, use_edges=True):
        if len(values) != len(counts):
            raise ValueError('Values and counts not the same length!')
        if len(values) != len(std_devs):
            raise ValueError('Values and std_devs not the same length!')
        if use_edges:
            if len(values) != len(edges_or_bin_centers) - 1:
                raise ValueError('Edges should have n+1 dimension! len(values) = {}, len(edges) = {}'.format(len(values), len(edges_or_bin_centers)))
        else:
            if len(values) != len(edges_or_bin_centers):
                raise ValueError('Bin centers should have n dimension! len(values) = {}, len(bin_centers) = {}'.format(len(values), len(edges_or_bin_centers)))

        self.v = np.where(np.isnan(values), 0, values)
        self.c = counts
        self.s = np.where(np.isnan(std_devs), 0, std_devs)
        self.isrebinned = False
        if use_edges:
            self.e = edges_or_bin_centers
            self.bc = 0.5 * (self.e[:-1] + self.e[1:])
        else:
            self.bc = edges_or_bin_centers
            width = edges_or_bin_centers[1] - edges_or_bin_centers[0]
            self.e = np.linspace(edges_or_bin_centers[0] - width / 2,
                                 edges_or_bin_centers[-1] + width / 2,
                                 len(edges_or_bin_centers) + 1)

    
    def values(self):
        return self.v

    def counts(self):
        return self.c
    
    def std_devs(self):
        return self.s

    def errors(self):
        sumwy2 = np.sum(self.c*self.v**2)
        sumwy  = np.sum(self.c*self.v)
        sumw   = np.sum(self.c)
        if sumw == 0:
            sumw = 1
        global_mean = sumwy / sumw

        cond = (sumwy2 != 0) & (self.c < 5)
        test = np.where(cond, self.s * self.s * self.c / sumwy2, 1)
        zero_cond = (test < 1e-4) | (self.s <= 0) | (self.c <= 0)
        err = np.divide(self.s, self.c ** 0.5, out=np.zeros(np.shape(self.s)), where=np.logical_not(zero_cond))
        zero_ind = np.where(zero_cond)
        err[zero_ind] = 2*(abs(sumwy2/sumw - global_mean**2))**0.5
        return err
    
    def global_mean(self):
        sumwy  = np.sum(self.c*self.v)
        print(f'sumwy: {sumwy}')
        sumw   = np.sum(self.c)
        print(f'sumw: {sumw}')
        if sumw == 0:
            sumw = 1
        return sumwy / sumw
    
    def global_mean_error(self):
        sumwy2 = np.sum(self.c*self.v**2)
        sumwy  = np.sum(self.c*self.v)
        sumw   = np.sum(self.c)
        if sumw == 0:
            sumw = 1
        global_mean = sumwy / sumw
        return 2*(abs(sumwy2/sumw - global_mean**2))**0.5

    def edges(self):
        return self.e

    def bin_centers(self):
        return self.bc

    def Rebin(self, nrebin):
        if nrebin == 1:
            return

        nbin = len(self.v)
        if nbin % nrebin != 0:
            raise ValueError('nrebin must divide the number of bins!')
        old_c = self.c.copy()
        self.v, self.c, self.s = add_helper(self.v.reshape(nbin // nrebin, nrebin).T,
                                            self.c.reshape(nbin // nrebin, nrebin).T,
                                            self.s.reshape(nbin // nrebin, nrebin).T)
        self.e = np.linspace(self.e[0], self.e[-1], nbin // nrebin + 1)

        # new bin centers should be weighted average of the old ones based on counts
        self.e = np.arange(self.e[0], self.e[-1], nrebin * (self.e[1] - self.e[0]))
        # replace 0 weight with 1 to avoid ZeroDivisionError
        old_c = np.where(old_c == 0, 1, old_c)
        self.bc = np.average(self.bc.reshape(-1, nrebin), weights=old_c.reshape(-1, nrebin), axis=1)
        self.isrebinned = True

    def Add(self, other, c1=1, c2=1):
        # Similar to TProfile::Add, but not really what we want
        if not isinstance(other, SimpleProfile):
            raise ValueError('Input should be a SimpleProfile object!')
        if len(self.v) != len(other.v):
            raise ValueError('Input profiles should have the same number of bins!')
        if self.isrebinned or other.isrebinned:
            raise ValueError('Cannot add rebinned profiles! Add first then rebin!')
        h_1 = self.c * self.v
        h_2 = other.c * other.v
        e_1 = self.c * (self.s ** 2 + self.v ** 2)
        e_2 = other.c * (other.s ** 2 + other.v ** 2)
        c_1 = self.c
        c_2 = other.c

        # new parameters
        new_h = c1 * h_1 + c2 * h_2
        new_e = abs(c1) * e_1 + abs(c2) * e_2
        new_c = abs(c1) * c_1 + abs(c2) * c_2
        vo = np.divide(new_h, new_c, out=np.zeros(np.shape(new_h)), where=new_c != 0)
        so = np.sqrt(np.divide(new_e, new_c, out=np.zeros(np.shape(new_e)), where=new_c != 0) - vo ** 2)
        co = new_c
        return SimpleProfile(vo, co, so, self.e)
    
    def Add2(self, other, c1=1, c2=1):
        # An operation such that v_N = c1*v_1 + c2*v_2, error propagation accordingly
        if not isinstance(other, SimpleProfile):
            raise ValueError('Input should be a SimpleProfile object!')
        if len(self.v) != len(other.v):
            raise ValueError('Input profiles should have the same number of bins!')
        if self.isrebinned or other.isrebinned:
            raise ValueError('Cannot add rebinned profiles! Add first then rebin!')
        
        new_v = c1 * self.v + c2 * other.v
        new_c = abs(c1) * self.c + abs(c2) * other.c
        new_error = np.sqrt((c1 * self.errors()) ** 2 + (c2 * other.errors()) ** 2)
        new_s = new_error * np.sqrt(new_c)
        return SimpleProfile(new_v, new_c, new_s, self.e)

    def __add__(self, other):
        # technically this is merging
        vi = np.vstack((self.v, other.v))
        ci = np.vstack((self.c, other.c))
        si = np.vstack((self.s, other.s))
        vo, co, so = add_helper(vi, ci, si)
        return SimpleProfile(vo, co, so, self.e)

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)
        
    def __truediv__(self, scalar):
        if not isinstance(scalar, (int, float, uncertainties.core.Variable, uncertainties.core.AffineScalarFunc)):
            raise ValueError('Can only divide by a scalar or a uncertainty ufloat!')
        if isinstance(scalar, (uncertainties.core.Variable, uncertainties.core.AffineScalarFunc)):
            new_v = self.v / scalar.n
            old_e = self.errors()
            new_e = np.sqrt((old_e / scalar.n) ** 2 + (self.v * scalar.s / scalar.n ** 2) ** 2)
            new_s = new_e * np.sqrt(self.c)
            return SimpleProfile(new_v, self.c, new_s, self.e)
        return SimpleProfile(self.v / scalar, self.c, self.s / scalar, self.e)


if __name__ == '__main__':
    print('Testing SimpleProfile class...')