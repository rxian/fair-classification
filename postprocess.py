from itertools import chain

import numpy as np
from qpsolvers import solve_ls
from scipy.optimize import linprog
from scipy.sparse import csc_matrix
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class PostProcessorDP(BaseEstimator):
  """Post-processing mapping for DP fairness.

  Attributes:
    n_classes_: int
      Number of classes.
    n_groups_: int
      Number of demographic groups.
    score_: float
      Group-weighted classification error of training examples post-processed
      with Kantorovich transports.
    barycenter_: array-like, shape (n_classes,)
      Wasserstein-barycenter of class probabilities.
    q_by_group_: list of array-like, shape (n_classes,)
      Output class distributions of each demographic group.
      May not be equal to barycenter_ when eps > 0.
    psi_by_group_: array-like, shape (n_groups, n_classes)
      Parameters of post-processing maps of each group.
    gamma_by_group_: array-like, shape (n_groups, n_examples, n_classes)
      Kantorovich transports (optimal coupling) of each group (unnormalized).
  """

  def fit(self, probas, groups, eps=0.0, w=None, p=None, q_by_group=None):
    """Estimate a post-processing map.

    Args:
      probas: array-like, shape (n_examples, n_classes)
        Class probabilities (predicted) of each example.
      groups: array-like, shape (n_examples,)
        Group label (zero-indexed) of each example.
      eps: float, optional
        Amount of relaxation of DP constraint.  Specifies desired DP gap from
        post-processing.  Default is 0.
      w: array-like, shape (n_groups,), optional
        Weights assigned to each group for weighting classification error.
        Default is uniform (group-balanced).
      p: array-like, shape (n_examples,), optional
        Probability masses of each example.  Default is uniform.
      q_by_group: list of array-like, shape (n_classes,), optional
        Specify target output class distributions of each demographic group.
    """
    probas, groups = check_X_y(probas, groups)
    if p is not None:
      _, p = check_X_y(probas, p)

    self.n_classes_ = probas.shape[-1]
    self.n_groups_ = int(1 + np.max(groups))

    probas_by_group = [probas[groups == i] for i in range(self.n_groups_)]
    p_by_group = None if p is None else [
        p[groups == i] / np.sum(p[groups == i]) for i in range(self.n_groups_)
    ]

    gammas_unnormalized, cost, barycenter = self.linprog_dp_(
        probas_by_group,
        eps=eps,
        w=w,
        p_by_group=p_by_group,
        q_by_group=q_by_group)
    self.score_ = cost
    self.gamma_by_group_ = [
        gamma / gamma.sum() for gamma in gammas_unnormalized
    ]
    self.barycenter_ = barycenter
    self.q_by_group_ = [gamma.sum(axis=0) for gamma in self.gamma_by_group_]
    self.psi_by_group_ = np.stack([
        self.find_point_(probas_by_group[i], gammas_unnormalized[i])
        for i in range(self.n_groups_)
    ])
    return self

  def linprog_dp_(self,
                  probas_by_group,
                  eps=0.0,
                  w=None,
                  p_by_group=None,
                  q_by_group=None):
    """Find barycenter and Kantorovich simplex-vertex transports of each group.
       Implements the OPT linear program in the paper."""

    # Decision variables are the probability mass of the couplings, followed by
    # the barycenter, and the slack variables.
    #
    # They are flattened into a single vector, which when unraveled is a 3d
    # tensor of shape (n_groups, n_examples, n_classes), followed by a vector of
    # length (n_classes,), and two 2d tensors of shape (n_groups, n_classes).

    n_examples = [len(probas) for probas in probas_by_group]
    a_max = np.argmax(n_examples)

    offset_barycenter = sum(n_examples) * self.n_classes_
    offset_qs = offset_barycenter + self.n_classes_
    offset_slacks = offset_qs + self.n_groups_ * self.n_classes_
    n_decisions = offset_slacks + self.n_groups_ * self.n_classes_

    def ravel_index(multi_index):
      """Get indexes to flattened couplings given multi-index of groups, example
         index, and classes."""
      indices = np.empty(len(multi_index[0]), dtype=np.uint)
      offset = 0
      for a in range(self.n_groups_):
        indices[multi_index[0] == a] = offset + np.ravel_multi_index(
            [idx[multi_index[0] == a] for idx in multi_index[1:]],
            (n_examples[a], self.n_classes_))
        offset += n_examples[a] * self.n_classes_
      return [int(x) for x in indices]

    # l_1 transportation costs
    q = []
    for a in range(self.n_groups_):
      s = np.repeat(np.arange(n_examples[a]), self.n_classes_, axis=0)
      y = np.tile(np.arange(self.n_classes_), n_examples[a])
      costs = 1 - probas_by_group[a][s, y]
      if w is not None:
        costs *= w[a]
      # Normalize, due to the upscaling (see implementation below)
      costs *= n_examples[a_max] / n_examples[a]
      q.extend(costs)
    q = np.array(q)

    # Get constraints
    G_i = []
    G_j = []
    G_v = []
    G_rows = 0
    h = []

    A_i = []
    A_j = []
    A_v = []
    A_rows = 0
    b = []

    # \sum_y \gamma_{a, s, y} = p_{a, s}
    for a in range(self.n_groups_):
      for s in range(n_examples[a]):
        A_i.extend([A_rows] * self.n_classes_)
        A_j.extend(
            ravel_index([
                np.full(self.n_classes_, a),
                np.full(self.n_classes_, s),
                np.arange(self.n_classes_),
            ]))
        # Upscale by n_examples[a] to prevent underflow
        A_v.extend([1.0] * self.n_classes_)
        b.append(p_by_group[a][s] * n_examples[a] if p_by_group is not None else
                 1.0)  # (1 / n_examples[a]) * n_examples[a] = 1
        A_rows += 1

    # \sum_s \gamma_{a, s, y} = q_{a, y}
    for a in range(self.n_groups_):
      for y in range(self.n_classes_):
        A_i.extend([A_rows] * n_examples[a])
        A_j.extend(
            ravel_index([
                np.full(n_examples[a], a),
                np.arange(n_examples[a]),
                np.full(n_examples[a], y),
            ]))
        # Upscaled by n_examples[a] to prevent underflow
        A_v.extend([1.0] * n_examples[a])
        A_i.append(A_rows)
        A_j.append(offset_qs + a * self.n_classes_ + y)
        A_v.append(-1.0 * n_examples[a])
        b.append(0.0)
        A_rows += 1

    if q_by_group is None:
      # -\xi_{a, y} <= q_{a, y} - barycenter_{y} <= \xi_{a, y}
      for a in range(self.n_groups_):
        for y in range(self.n_classes_):
          for i in [-1.0, 1.0]:
            G_i.append(G_rows)
            G_j.append(offset_qs + a * self.n_classes_ + y)
            G_v.append(i)
            G_i.append(G_rows)
            G_j.append(offset_barycenter + y)
            G_v.append(-i)
            G_i.append(G_rows)
            G_j.append(offset_slacks + a * self.n_classes_ + y)
            G_v.append(-1.0)
            h.append(0.0)
            G_rows += 1
    else:
      for a in range(self.n_groups_):
        for y in range(self.n_classes_):
          A_i.append(A_rows)
          A_j.append(offset_qs + a * self.n_classes_ + y)
          A_v.append(1.0)
          b.append(q_by_group[a][y])
          A_rows += 1

    # \sum_y \xi_{a, y} <= \epsilon
    for a in range(self.n_groups_):
      G_i.extend([G_rows] * self.n_classes_)
      G_j.extend(
          range(offset_slacks + a * self.n_classes_,
                offset_slacks + (a + 1) * self.n_classes_))
      G_v.extend([1.0] * self.n_classes_)
      h.append(eps)
      G_rows += 1

    ## `scipy.optimize.linprog` interface
    G = csc_matrix((G_v, (G_i, G_j)), shape=(G_rows, n_decisions))
    h = np.array(h)
    A = csc_matrix((A_v, (A_i, A_j)), shape=(A_rows, n_decisions))
    b = np.array(b)
    sol = linprog(np.concatenate([q, [0.0] * (n_decisions - len(q))]),
                  A_ub=G,
                  b_ub=h,
                  A_eq=A,
                  b_eq=b,
                  bounds=(0, None),
                  method="highs")
    x = sol.x

    gammas_unnormalized = []
    cost = 0.0
    offset = 0
    for a in range(self.n_groups_):
      this_gamma = x[offset:offset + n_examples[a] * self.n_classes_]
      gammas_unnormalized.append(
          this_gamma.reshape((n_examples[a], self.n_classes_)))
      this_cost = q[offset:offset + n_examples[a] *
                    self.n_classes_] * n_examples[a] / n_examples[a_max]
      cost += np.sum(this_gamma / np.sum(this_gamma) * this_cost)
      offset += n_examples[a] * self.n_classes_
    if w is None:
      cost /= self.n_groups_
    return gammas_unnormalized, cost, x[
        offset_barycenter:offset_barycenter +
        self.n_classes_] if q_by_group is None else None

  def find_point_(self, probas, gamma):
    """Extract an approximate Monge transport from a Kantorovich simplex-vertex
       transport."""

    # Compute boundaries and get constraints for finding feasible point in
    # convex polytope
    G_i = []
    G_j = []
    G_v = []
    G_rows = 0
    boundaries = np.zeros((self.n_classes_, self.n_classes_))

    for i in range(self.n_classes_):
      for j in chain(range(i), range(i + 1, self.n_classes_)):
        idx = gamma[:, i] > 0  # or ~np.isclose(gamma[:, i], 0)?
        boundaries[i, j] = np.max(probas[idx, j] - probas[idx, i] + 1,
                                  initial=0)
        G_i.extend([G_rows] * 2)
        G_j.extend([i, j])
        G_v.extend([1, -1])
        G_rows += 1
    boundaries -= np.clip(boundaries + boundaries.T - 2, 0,
                          None) / 2  # "Fix" numerical imprecisions
    gaps = (2 - boundaries.T -
            boundaries)[np.where(~np.eye(self.n_classes_, dtype=bool))]
    gaps = np.clip(gaps, 1e-2, None)

    G = csc_matrix((G_v, (G_i, G_j)), shape=(G_rows, self.n_classes_))
    h = 1 - boundaries[np.where(~np.eye(self.n_classes_, dtype=bool))]

    ## `scipy.optimize.linprog` interface
    # sol = linprog(np.zeros(self.n_classes_),
    #               A_ub=G,
    #               b_ub=h,
    #               bounds=None,
    #               method="highs")
    # z = res.x

    ## `qpsolvers` interface
    W = csc_matrix(np.diag(1 / gaps**2))
    z = solve_ls(G, h, G, h, W=W, solver="osqp")

    return np.array([0] +
                    [2 * (z[0] - z[j]) for j in range(1, self.n_classes_)])

  def predict(self, probas, groups):
    """Output class assignments given probas and group labels.

    Args:
      probas: array-like, shape (n_examples, n_classes)
        Class probabilities (predicted) of each example.
      groups: array-like, shape (n_examples,)
        Group label (zero-indexed) of each example.
    """
    check_is_fitted(self, "psi_by_group_")
    probas = check_array(probas)
    groups = check_array(groups, ensure_2d=False)
    probas, groups = check_X_y(probas, groups)
    return np.argmin(2 * (1 - probas) - self.psi_by_group_[groups], axis=1)
