from itertools import chain

import numpy as np
from qpsolvers import solve_ls
from scipy.optimize import linprog
from scipy.sparse import csc_matrix
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class PostProcessorDP(BaseEstimator):
  """Post-processing mapping for DP fairness.

  Based on the paper https://arxiv.org/abs/2211.01528.

  Attributes:
    n_classes_: int
      Number of classes.
    n_groups_: int
      Number of demographic groups.
    score_: float
      Weighted classification error on post-processed training examples.
    psi_by_group_: array-like, shape (n_groups, n_classes)
      Parameters of post-processing maps.
    q_by_group_: list of array-like, shape (n_classes,)
      Distributions of class assignments on post-processed training examples.
    gamma_by_group_: array-like, shape (n_groups, n_examples, n_classes)
      Class assignments of each post-processed training example (unnormalized).
  """

  def fit(self, scores, groups, alpha=0.0, w=None, r=None, q_by_group=None):
    """Estimate a post-processing map.

    Args:
      scores: array-like, shape (n_examples, n_classes)
        Predictor scores/class probabilities of each example.
      groups: array-like, shape (n_examples,)
        Group label (zero-indexed) of each example.
      alpha: float, optional
        Relaxation of DP constraint.  Specifies desired DP gap from 
        post-processing.  Default is 0 (exact DP).
      w: array-like, shape (n_groups,), optional
        Weight of each group for weighting classification error (need not be
        normalized).  Default is uniform (group-balanced).
      r: array-like, shape (n_examples,), optional
        Probability mass of each example (need not be normalized, e.g., to avoid
        underflow).  Default is uniform.
      q_by_group: list of array-like, shape (n_classes,), optional
        Specify desired distributions of class assignments of each group.
    """
    scores, groups = check_X_y(scores, groups)
    if r is not None:
      _, r = check_X_y(scores, r)

    self.n_classes_ = scores.shape[-1]
    self.n_groups_ = int(1 + np.max(groups))
    self.alpha_ = alpha
    if w is None:
      w = [1.0 for _ in range(self.n_groups_)]
    self.w_ = w

    scores_by_group = [scores[groups == a] for a in range(self.n_groups_)]
    r_by_group = [(r[groups == a] if r is not None else np.ones(
        (groups == a).sum())) for a in range(self.n_groups_)]

    self.score_, self.q_by_group_, self.gamma_by_group_ = self.linprog_dp_(
        scores_by_group,
        alpha=alpha,
        w=w,
        r_by_group=r_by_group,
        q_by_group=q_by_group)
    self.psi_by_group_ = np.stack([
        self.find_point_(scores_by_group[a], self.gamma_by_group_[a])
        for a in range(self.n_groups_)
    ])
    return self

  def linprog_dp_(self,
                  scores_by_group,
                  alpha,
                  w,
                  r_by_group,
                  q_by_group=None,
                  tol=1e-6):
    """This implements the LP in the paper (Line 3 of Algorithm 2)."""

    # Decision variables are the probability mass of the couplings, followed by
    # the barycenter, the output distributions, and the slack variables.
    #
    # They are flattened and concatenated into a single vector, which when
    # unpacked and unraveled are tensors of shapes:
    # - (n_examples[a], n_classes), for a = 1, ..., n_groups
    # - (n_classes,)
    # - (n_groups, n_classes)
    # - (n_groups, n_classes)

    n_examples = [len(scores) for scores in scores_by_group]
    total_r = [r.sum() for r in r_by_group]
    a_max = np.argmax(total_r)

    offset_barycenter = sum(n_examples) * self.n_classes_
    offset_qs = offset_barycenter + self.n_classes_
    offset_slacks = offset_qs + self.n_groups_ * self.n_classes_
    n_variables = offset_slacks + self.n_groups_ * self.n_classes_

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
    cost = []
    for a in range(self.n_groups_):
      s = np.repeat(np.arange(n_examples[a]), self.n_classes_, axis=0)
      y = np.tile(np.arange(self.n_classes_), n_examples[a])
      c = 1 - scores_by_group[a][s, y]
      # Normalize according to total p_by_group
      c *= total_r[a_max] / total_r[a]
      c *= w[a]
      cost.extend(c)
    cost = np.array(cost)

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

    # \sum_y \gamma_{a, s, y} = r_{a, s}
    for a in range(self.n_groups_):
      for s in range(n_examples[a]):
        A_i.extend([A_rows] * self.n_classes_)
        A_j.extend(
            ravel_index([
                np.full(self.n_classes_, a),
                np.full(self.n_classes_, s),
                np.arange(self.n_classes_),
            ]))
        A_v.extend([1.0] * self.n_classes_)
        A_rows += 1
      b.extend(r_by_group[a])

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
        A_v.extend([1.0] * n_examples[a])
        A_i.append(A_rows)
        A_j.append(offset_qs + a * self.n_classes_ + y)
        A_v.append(-1.0 * total_r[a])
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

      # \xi_{a, y} <= \alpha / 2
      G_i.extend(range(G_rows, G_rows + self.n_groups_ * self.n_classes_))
      G_j.extend(
          range(offset_slacks,
                offset_slacks + self.n_groups_ * self.n_classes_))
      G_v.extend([1.0] * self.n_groups_ * self.n_classes_)
      h.extend([alpha / 2] * self.n_groups_ * self.n_classes_)
      G_rows += self.n_groups_ * self.n_classes_

    else:
      for a in range(self.n_groups_):
        for y in range(self.n_classes_):
          A_i.append(A_rows)
          A_j.append(offset_qs + a * self.n_classes_ + y)
          A_v.append(1.0)
          b.append(q_by_group[a][y])
          A_rows += 1

    # `scipy.optimize.linprog` interface
    G = csc_matrix((G_v, (G_i, G_j)), shape=(G_rows, n_variables))
    h = np.array(h)
    A = csc_matrix((A_v, (A_i, A_j)), shape=(A_rows, n_variables))
    b = np.array(b)
    sol = linprog(np.concatenate([cost, [0.0] * (n_variables - len(cost))]),
                  A_ub=G,
                  b_ub=h,
                  A_eq=A,
                  b_eq=b,
                  bounds=(0, None),
                  method="highs",
                  options={
                      "dual_feasibility_tolerance": tol,
                      "primal_feasibility_tolerance": tol,
                  })
    assert sol.status == 0, sol.message
    x = sol.x

    gammas_by_group_unnormalized = []
    total_cost = 0.0
    offset = 0
    for a in range(self.n_groups_):
      this_gamma = x[offset:offset + n_examples[a] * self.n_classes_]
      gammas_by_group_unnormalized.append(
          this_gamma.reshape((n_examples[a], self.n_classes_)))
      c = cost[offset:offset + n_examples[a] * self.n_classes_]
      total_cost += np.sum(this_gamma * c) / total_r[a_max]
      offset += n_examples[a] * self.n_classes_
    total_cost /= sum(w)

    return total_cost, [
        x[offset_qs + a * self.n_classes_:offset_qs + (a + 1) * self.n_classes_]
        for a in range(self.n_groups_)
    ], gammas_by_group_unnormalized

  def find_point_(self, scores, gamma):
    """Extract post-processing map from LP solution (Lines 6-9 of Alg. 2)."""

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
        boundaries[i, j] = np.max(scores[idx, j] - scores[idx, i] + 1,
                                  initial=0)
        G_i.extend([G_rows] * 2)
        G_j.extend([i, j])
        G_v.extend([1, -1])
        G_rows += 1
    boundaries -= np.clip(
        boundaries + boundaries.T - 2, 0,
        None) / 2  # in case of non-optimal LP solution, incl. numerical issues
    gaps = (2 - boundaries.T -
            boundaries)[np.where(~np.eye(self.n_classes_, dtype=bool))]

    G = csc_matrix((G_v, (G_i, G_j)), shape=(G_rows, self.n_classes_))
    h = 1 - boundaries[np.where(~np.eye(self.n_classes_, dtype=bool))]

    # # `scipy.optimize.linprog` interface
    # sol = linprog(np.zeros(self.n_classes_),
    #               A_ub=G,
    #               b_ub=h,
    #               bounds=None,
    #               method="highs")
    # z = res.x

    # `qpsolvers` interface
    W = csc_matrix(np.diag(1 / np.clip(gaps, 1e-2, None)**
                           2))  # ignore small gaps, prevent division by zero
    z = solve_ls(G, h, G, h, W=W, solver="osqp")
    assert z is not None, f"No feasible point found; should not happen...\nboundaries =\n{boundaries}"

    return np.array([0] +
                    [2 * (z[0] - z[j]) for j in range(1, self.n_classes_)])

  def predict(self, scores, groups):
    """Output fair class assignments given predictor scores.

    Args:
      scores: array-like, shape (n_examples, n_classes)
        Predictor scores/class probabilities of each example.
      groups: array-like, shape (n_examples,)
        Group label (zero-indexed) of each example.
    """
    check_is_fitted(self, "psi_by_group_")
    scores = check_array(scores)
    groups = check_array(groups, ensure_2d=False)
    scores, groups = check_X_y(scores, groups)
    return np.argmin(2 * (1 - scores) - self.psi_by_group_[groups], axis=1)
