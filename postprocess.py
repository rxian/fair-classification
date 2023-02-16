from contextlib import redirect_stdout
import io
from itertools import chain

from cvxopt import matrix, spmatrix, solvers
import numpy as np
import scipy
from scipy.optimize import linprog
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

    gammas, cost, barycenter = self.find_barycenter_and_kantorovich_transports_(
        probas_by_group,
        eps=eps,
        w=w,
        p_by_group=p_by_group,
        q_by_group=q_by_group)
    self.score_ = cost
    self.gamma_by_group_ = gammas
    self.barycenter_ = barycenter
    self.q_by_group_ = [gamma.sum(axis=0) / gamma.sum() for gamma in gammas]
    self.psi_by_group_ = np.stack([
        self.extract_monge_transport_(probas_by_group[i], gammas[i])
        for i in range(self.n_groups_)
    ])
    return self

  def find_barycenter_and_kantorovich_transports_(self,
                                                  probas_by_group,
                                                  eps=0.0,
                                                  w=None,
                                                  p_by_group=None,
                                                  q_by_group=None):
    """Find barycenter and Kantorovich simplex-vertex transports of each group.
       Implements the OPT linear program in the paper."""

    # Design variables are the probability mass of the couplings, followed by
    # the barycenter, and the slack variables.
    #
    # They are flattened into a single vector, which when unraveled is a 3d
    # tensor of shape (n_groups, n_examples, n_classes), followed by a vector of
    # shape (n_classes,), and two 2d tensors of shape (n_groups, n_classes).

    n_examples = [len(probas) for probas in probas_by_group]
    a_max = np.argmax(n_examples)

    offset_barycenter = sum(n_examples) * self.n_classes_
    offset_q = offset_barycenter + self.n_classes_
    offset_slacks = offset_q + self.n_groups_ * self.n_classes_
    n_designs = offset_slacks + self.n_groups_ * self.n_classes_

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
    c = []
    for a in range(self.n_groups_):
      s = np.repeat(np.arange(n_examples[a]), self.n_classes_, axis=0)
      y = np.tile(np.arange(self.n_classes_), n_examples[a])
      costs = 1 - probas_by_group[a][s, y]
      if w is not None:
        costs *= w[a]
      # Normalize, due to the upscaling (see implementation below)
      costs *= n_examples[a_max] / n_examples[a]
      c.extend(costs)
    c = np.array(c)

    # Get constraints
    A_ubs_sparse_i = []
    A_ubs_sparse_j = []
    A_ubs_sparse_v = []
    b_ubs = []
    row_ubs = 0

    A_eqs_sparse_i = []
    A_eqs_sparse_j = []
    A_eqs_sparse_v = []
    b_eqs = []
    row_eqs = 0

    # \sum_y \gamma_{a, s, y} = p_{a, s}
    for a in range(self.n_groups_):
      for s in range(n_examples[a]):
        A_eqs_sparse_i.extend([row_eqs] * self.n_classes_)
        A_eqs_sparse_j.extend(
            ravel_index([
                np.full(self.n_classes_, a),
                np.full(self.n_classes_, s),
                np.arange(self.n_classes_),
            ]))
        # Upscale by n_examples[a] to prevent underflow
        A_eqs_sparse_v.extend([1.0] * self.n_classes_)
        b_eqs.append(p_by_group[a][s] * n_examples[a] if p_by_group is not None
                     else 1.0)  # (1 / n_examples[a]) * n_examples[a] = 1
        row_eqs += 1

    # \sum_s \gamma_{a, s, y} = q_{a, y}
    for a in range(self.n_groups_):
      for y in range(self.n_classes_):
        A_eqs_sparse_i.extend([row_eqs] * n_examples[a])
        A_eqs_sparse_j.extend(
            ravel_index([
                np.full(n_examples[a], a),
                np.arange(n_examples[a]),
                np.full(n_examples[a], y),
            ]))
        # Upscaled by n_examples[a] to prevent underflow
        A_eqs_sparse_v.extend([1.0] * n_examples[a])
        A_eqs_sparse_i.append(row_eqs)
        A_eqs_sparse_j.append(offset_q + a * self.n_classes_ + y)
        A_eqs_sparse_v.append(-1.0 * n_examples[a])
        b_eqs.append(0.0)
        row_eqs += 1

    # \sum_y q_{a, y} = 1
    for a in range(self.n_groups_):
      A_eqs_sparse_i.extend([row_eqs] * self.n_classes_)
      A_eqs_sparse_j.extend(
          range(offset_q + a * self.n_classes_,
                offset_q + (a + 1) * self.n_classes_))
      A_eqs_sparse_v.extend([1.0] * self.n_classes_)
      b_eqs.append(1.0)
      row_eqs += 1

    if q_by_group is None:
      # -\xi_{a, y} <= q_{a, y} - barycenter_{y} <= \xi_{a, y}
      for a in range(self.n_groups_):
        for y in range(self.n_classes_):
          for i in [-1.0, 1.0]:
            A_ubs_sparse_i.append(row_ubs)
            A_ubs_sparse_j.append(offset_q + a * self.n_classes_ + y)
            A_ubs_sparse_v.append(i)
            A_ubs_sparse_i.append(row_ubs)
            A_ubs_sparse_j.append(offset_barycenter + y)
            A_ubs_sparse_v.append(-i)
            A_ubs_sparse_i.append(row_ubs)
            A_ubs_sparse_j.append(offset_slacks + a * self.n_classes_ + y)
            A_ubs_sparse_v.append(-1.0)
            b_ubs.append(0.0)
            row_ubs += 1
    else:
      for a in range(self.n_groups_):
        for y in range(self.n_classes_):
          A_eqs_sparse_i.append(row_eqs)
          A_eqs_sparse_j.append(offset_q + a * self.n_classes_ + y)
          A_eqs_sparse_v.append(1.0)
          b_eqs.append(q_by_group[a][y])
          row_eqs += 1

    # \sum_y \xi_{a, y} <= \epsilon
    for a in range(self.n_groups_):
      A_ubs_sparse_i.extend([row_ubs] * self.n_classes_)
      A_ubs_sparse_j.extend(
          range(offset_slacks + a * self.n_classes_,
                offset_slacks + (a + 1) * self.n_classes_))
      A_ubs_sparse_v.extend([1.0] * self.n_classes_)
      b_ubs.append(eps)
      row_ubs += 1

    ## `scipy.optimize.linprog` implementation
    A_ubs = scipy.sparse.coo_matrix(
        (A_ubs_sparse_v, (A_ubs_sparse_i, A_ubs_sparse_j)),
        shape=(row_ubs, n_designs))
    A_eqs = scipy.sparse.coo_matrix(
        (A_eqs_sparse_v, (A_eqs_sparse_i, A_eqs_sparse_j)),
        shape=(row_eqs, n_designs))
    sol = linprog(np.concatenate([c, [0.0] * (n_designs - len(c))]),
                  A_ub=A_ubs,
                  b_ub=b_ubs,
                  A_eq=A_eqs,
                  b_eq=b_eqs,
                  bounds=(0, None),
                  method="highs")
    x = sol.x

    gammas = []
    cost = 0.0
    offset = 0
    for a in range(self.n_groups_):
      this_gamma = x[offset:offset + n_examples[a] * self.n_classes_]
      gammas.append(this_gamma.reshape((n_examples[a], self.n_classes_)))
      this_cost = c[offset:offset + n_examples[a] *
                    self.n_classes_] * n_examples[a] / n_examples[a_max]
      cost += np.sum(this_gamma / np.sum(this_gamma) * this_cost)
      offset += n_examples[a] * self.n_classes_
    if w is None:
      cost /= self.n_groups_
    return gammas, cost, x[offset_barycenter:offset_barycenter +
                           self.n_classes_] if q_by_group is None else None

  def extract_monge_transport_(self, probas, gamma):
    """Extract an approximate Monge transport from a Kantorovich simplex-vertex
       transport."""

    # Compute boundaries and get constraints for finding feasible point in
    # convex polytope
    A_ubs_sparse_i = []
    A_ubs_sparse_j = []
    A_ubs_sparse_v = []
    B = np.zeros((self.n_classes_, self.n_classes_))
    rows = 0

    for i in range(self.n_classes_):
      for j in chain(range(i), range(i + 1, self.n_classes_)):
        idx = gamma[:, i] > 0  # or ~np.isclose(gamma[:, i], 0)?
        B[i, j] = np.max(probas[idx, j] - probas[idx, i] + 1, initial=0)
        A_ubs_sparse_i.extend([rows] * 2)
        A_ubs_sparse_j.extend([i, j])
        A_ubs_sparse_v.extend([1, -1])
        rows += 1

    # Fix numerical imprecisions
    B -= np.clip(B + B.T - 2, 0, None) / 2
    b_ubs = -B[np.where(~np.eye(B.shape[0], dtype=bool))] + 1

    # ## `scipy.optimize.linprog` implementation
    # A_ubs = scipy.sparse.coo_matrix(
    #     (A_ubs_sparse_v, (A_ubs_sparse_i, A_ubs_sparse_j)),
    #     shape=(rows, self.n_classes_))
    # sol = linprog(
    #     np.zeros(self.n_classes_),
    #     A_ub=A_ubs,
    #     b_ub=b_ubs,
    #     bounds=None,
    #     method="highs")
    # z = res.x

    ## `cvxopt.solvers.qp` implementation
    h = matrix(b_ubs.reshape(-1, 1))
    G = spmatrix(A_ubs_sparse_v,
                 A_ubs_sparse_i,
                 A_ubs_sparse_j,
                 size=(rows, self.n_classes_))
    P = -1.0 / 2.0 * (G.T * G)
    q = 2.0 * G.T * h

    with redirect_stdout(io.StringIO()):
      sol = solvers.qp(P,
                       q,
                       G,
                       h,
                       kktsolver="ldl",
                       options={"kktreg": 1e-9})
    z = np.array(sol["x"]).squeeze()

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
