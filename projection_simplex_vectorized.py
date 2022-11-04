# Copyright (c) 2022, Mathieu Blondel
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np


def projection_simplex(V, z=1, axis=None):
    """
    Projection of x onto the simplex, scaled by z:
        P(x; z) = argmin_{y >= 0, sum(y) = z} ||y - x||^2
    z: float or array
        If array, len(z) must be compatible with V
    axis: None or int
        axis=None: project V by P(V.ravel(); z)
        axis=1: project each V[i] by P(V[i]; z[i])
        axis=0: project each V[:, j] by P(V[:, j]; z[j])
    """
    if axis == 1:
        n_features = V.shape[1]
        U = np.sort(V, axis=1)[:, ::-1]
        z = np.ones(len(V)) * z
        cssv = np.cumsum(U, axis=1) - z[:, np.newaxis]
        ind = np.arange(n_features) + 1
        cond = U - cssv / ind > 0
        rho = np.count_nonzero(cond, axis=1)
        theta = cssv[np.arange(len(V)), rho - 1] / rho
        return np.maximum(V - theta[:, np.newaxis], 0)

    elif axis == 0:
        return projection_simplex(V.T, z, axis=1).T

    else:
        V = V.ravel().reshape(1, -1)
        return projection_simplex(V, z, axis=1).ravel()


def _projection_simplex(v, z=1):
    """
    Old implementation for test and benchmark purposes.
    The arguments v and z should be a vector and a scalar, respectively.
    """
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w


def test():
    from sklearn.utils.testing import assert_array_almost_equal


    rng = np.random.RandomState(0)
    V = rng.rand(100, 10)

    # Axis = None case.
    w = projection_simplex(V[0], z=1, axis=None)
    w2 = _projection_simplex(V[0], z=1)
    assert_array_almost_equal(w, w2)

    w = projection_simplex(V, z=1, axis=None)
    w2 = _projection_simplex(V.ravel(), z=1)
    assert_array_almost_equal(w, w2)

    # Axis = 1 case.
    W = projection_simplex(V, axis=1)

    # Check same as with for loop.
    W2 = np.array([_projection_simplex(V[i]) for i in range(V.shape[0])])
    assert_array_almost_equal(W, W2)

    # Check works with vector z.
    W3 = projection_simplex(V, np.ones(V.shape[0]), axis=1)
    assert_array_almost_equal(W, W3)

    # Axis = 0 case.
    W = projection_simplex(V, axis=0)

    # Check same as with for loop.
    W2 = np.array([_projection_simplex(V[:, i]) for i in range(V.shape[1])]).T
    assert_array_almost_equal(W, W2)

    # Check works with vector z.
    W3 = projection_simplex(V, np.ones(V.shape[1]), axis=0)
    assert_array_almost_equal(W, W3)


def benchmark():
    import time

    n_features = 100
    n_repeats = 5
    sizes = (10, 100, 1000, 10000)

    rng = np.random.RandomState(0)

    vectorized = np.zeros(len(sizes))
    loop = np.zeros(len(sizes))

    for i, n_samples in enumerate(sizes):
        for _ in range(n_repeats):
            V = rng.rand(n_samples, 10)

            start = time.clock()
            projection_simplex(V, axis=0)
            vectorized[i] += time.clock() - start

            start = time.clock()
            [_projection_simplex(V[i]) for i in range(V.shape[0])]
            loop[i] += time.clock() - start

        vectorized[i] /= n_repeats
        loop[i] /= n_repeats

    import matplotlib.pylab as plt

    plt.figure()
    plt.plot(sizes, loop / vectorized, linewidth=3)
    plt.title("Vectorized projection onto the simplex")
    plt.xscale("log")
    plt.xlabel("Number of vectors to project")
    plt.ylabel("Speedup compared to using a for loop")
    plt.show()

if __name__ == '__main__':
    test()
    benchmark()
