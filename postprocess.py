from collections import defaultdict
from itertools import chain

import numpy as np
import cvxpy as cp
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class PostProcessorDP(BaseEstimator):
  """Post-processor for DP fairness.

  Based on Algorithm 2 of https://arxiv.org/abs/2211.01528.

  Attributes:
    n_classes_: int
      Number of classes.
    n_groups_: int
      Number of demographic groups.
    score_: float
      Weighted classification error on post-processed training examples (naming
      follows sklearn convention; to be distinguished from the class
      probabilities scores). 
    psi_by_group_: array-like, shape (n_groups, n_classes)
      Parameters of post-processing maps.
    q_by_group_: list of array-like, shape (n_classes,)
      Distributions of class assignments on post-processed training examples.
    gamma_by_group_: array-like, shape (n_groups, n_examples, n_classes)
      Probabilistic class assignments of each post-processed training example
      (unnormalized).
  """

  def fit(self,
          scores,
          groups,
          alpha=0.0,
          group_weight=None,
          sample_weight=None,
          q_by_group=None,
          tol=1e-8):
    """Perform post-processing using finite samples.

    Args:
      scores: array-like, shape (n_examples, n_classes)
        Scores/class probabilities of each example.
      groups: array-like, shape (n_examples,)
        Group label (zero-indexed) of each example; the sensitive attribute.
      alpha: float, optional
        Relaxation of DP constraint.  Specifies desired DP gap from 
        post-processing.  Default is 0 (exact DP).
      group_weight: array-like, shape (n_groups,), optional
        Weight of each group for weighting classification error (need not be
        normalized).  Default is empirical distribution of groups.
      sample_weight: array-like, shape (n_examples,), optional
        Weight of each example (need not be normalized).  Default is uniform.
      q_by_group: list of array-like, shape (n_classes,), optional
        Specify desired distributions of class assignments of each group.
    """
    scores, groups = check_X_y(scores, groups)
    if sample_weight is not None:
      _, r = check_X_y(scores, sample_weight)
    else:
      r = None

    self.n_classes_ = scores.shape[-1]
    self.n_groups_ = int(1 + np.max(groups))
    self.alpha_ = alpha
    if group_weight is None:
      w = np.bincount(groups, minlength=self.n_groups_) / len(groups)
    else:
      w = group_weight
    self.w_ = w

    scores_by_group = [scores[groups == a] for a in range(self.n_groups_)]
    # self.scores_by_group_ = scores_by_group  # for debugging
    r_by_group = []
    for a in range(self.n_groups_):
      if r is not None:
        this_r = r[groups == a]
        # Upscale to prevent underflow
        this_r *= len(this_r) / this_r.sum()
        r_by_group.append(this_r)
      else:
        r_by_group.append(np.ones((groups == a).sum()))
    total_r_max = max(len(r) for r in r_by_group)

    problem = self.linprog_dp_(scores_by_group,
                               alpha=alpha,
                               w=w,
                               r_by_group=r_by_group,
                               q_by_group=q_by_group)
    problem.solve(solver=cp.CBC, integerTolerance=tol)

    # Downscale, due to the upscaling to `r_by_group` above
    self.score_ = problem.value / total_r_max
    self.q_by_group_ = problem.var_dict["q"].value
    self.gamma_by_group_ = [
        problem.var_dict[f'gamma_{a}'].value for a in range(self.n_groups_)
    ]

    psi_by_group = []
    for a in range(self.n_groups_):
      try:
        problem = self.quadprog_find_point_(scores_by_group[a],
                                            self.gamma_by_group_[a],
                                            tol=tol)
        problem.solve(solver=cp.OSQP)
        z = problem.var_dict["z"].value
        if z is None:
          raise cp.error.SolverError
        psi_by_group.append(
            [0] + [2 * (z[0] - z[j]) for j in range(1, self.n_classes_)])
      except cp.error.SolverError:
        # Occurs if OSQP fails to converge, or `gamma_by_group_` is not optimal.
        # Fall back to an alternative linear program formulation.
        problem = self.linprog_score_transform_(scores_by_group[a],
                                                self.gamma_by_group_[a],
                                                tol=tol)
        problem.solve(solver=cp.CBC, integerTolerance=tol)
        psi_by_group.append(2 * problem.var_dict["bias"].value)
    self.psi_by_group_ = np.stack(psi_by_group)

    return self

  def linprog_dp_(self,
                  scores_by_group,
                  alpha,
                  w=None,
                  r_by_group=None,
                  q_by_group=None):
    """This implements the LP in the paper (Line 3 of Algorithm 2)."""

    alpha = cp.Parameter(value=alpha, name="alpha")

    # Variables are the probability mass of the couplings, the barycenter,
    # the output distributions, and slacks
    gamma_by_group = [
        cp.Variable(scores_by_group[a].shape, name=f"gamma_{a}")
        for a in range(self.n_groups_)
    ]
    barycenter = cp.Variable(self.n_classes_, name="barycenter")
    q = cp.Variable((self.n_groups_, self.n_classes_), name="q")
    slack = cp.Variable((self.n_groups_, self.n_classes_), name="slack")

    total_r = np.array([r.sum() for r in r_by_group])

    # Get l1 transportation costs
    # Upscale to prevent underflow, avoid numerical issues, and improve run time
    cost_by_group = [
        (1 - scores_by_group[a]) * w[a] * total_r.max() / total_r[a] / w.sum()
        for a in range(self.n_groups_)
    ]
    cost = sum([
        cp.sum(cp.multiply(gamma_by_group[a], cost_by_group[a]))
        for a in range(self.n_groups_)
    ])

    # Build constraints
    constraints = []

    # \sum_y \gamma_{a, s, y} = r_{a, s}
    for a in range(self.n_groups_):
      constraints.append(cp.sum(gamma_by_group[a], axis=1) == r_by_group[a])

    # \sum_s \gamma_{a, s, y} = q_{a, y}
    for a in range(self.n_groups_):
      constraints.append(cp.sum(gamma_by_group[a], axis=0) == q[a] * total_r[a])

    # -\xi_{a, y} <= q_{a, y} - barycenter_{y} <= \xi_{a, y}
    if q_by_group is None:
      for a in range(self.n_groups_):
        constraints.append(-slack[a] <= q[a] - barycenter)
        constraints.append(q[a] - barycenter <= slack[a])
    else:
      q_by_group = cp.Parameter((self.n_groups_, self.n_classes_),
                                value=q_by_group,
                                name="q_by_group")
      for a in range(self.n_groups_):
        constraints.append(-slack[a] <= q[a] - q_by_group[a])
        constraints.append(q[a] - q_by_group[a] <= slack[a])

    # \xi_{a, y} <= \alpha / 2
    constraints.append(slack <= alpha / 2)

    # All variables are nonnegative
    constraints.extend([gamma >= 0 for gamma in gamma_by_group])
    constraints.append(q >= 0)
    constraints.append(barycenter >= 0)
    constraints.append(slack >= 0)

    return cp.Problem(cp.Minimize(cost), constraints)

  def quadprog_find_point_(self, scores, gamma, tol=1e-8):
    """Extract post-processing map from LP solution (Lines 6-9 of Alg. 2)."""

    z = cp.Variable(self.n_classes_, name="z")

    # Compute boundaries that defines the convex polytope
    boundaries = np.zeros((self.n_classes_, self.n_classes_))
    for i in range(self.n_classes_):
      for j in chain(range(i), range(i + 1, self.n_classes_)):
        idx = gamma[:, i] > tol  # or ~np.isclose(gamma[:, i], 0)?
        boundaries[i, j] = np.max(scores[idx, j] - scores[idx, i] + 1,
                                  initial=0)
    boundaries -= np.clip(boundaries + boundaries.T - 2, 0,
                          None) / 2  # in case of numerical issues
    gaps = np.clip((2 - boundaries.T - boundaries), 1e-2, None)

    # Get cost and build constraints
    cost = 0
    constraints = []
    for i in range(self.n_classes_):
      for j in chain(range(i), range(i + 1, self.n_classes_)):
        # z_j - z_i >= B_ij - 1
        constraints.append(z[j] - z[i] >= boundaries[i, j] - 1)
        cost += cp.square(z[j] - z[i] - (boundaries[i, j] - 1)) / gaps[i, j]**2
    return cp.Problem(cp.Minimize(cost), constraints)

  def linprog_score_transform_(self, scores, gamma, tol=1e-8):
    """Extract post-processing map from LP solution (Lines 6-9 of Alg. 2)."""

    diffs = defaultdict(dict)
    for s, g in zip(scores, gamma):
      candidates = set(np.where(g > tol)[0])  # or use np.isclose?
      for i in candidates:
        for j in chain(range(i), range(i + 1, self.n_classes_)):
          # b_i + s_i >= b_j + s_j <==> b_i - b_j >= s_j - s_i
          d = s[j] - s[i]
          new_d = d if j not in diffs[i] else max(d, diffs[i][j])
          diffs[i][j] = new_d

    # Slack variables for handling numerical issues
    bias = cp.Variable(self.n_classes_, name="bias")
    slack = cp.Variable((self.n_classes_, self.n_classes_), name="slack")
    constraints = [slack >= 0]
    for i in diffs:
      for j in diffs[i]:
        constraints.append(bias[i] + slack[i][j] >= bias[j] + diffs[i][j])
    return cp.Problem(cp.Minimize(cp.sum(slack)), constraints)

  def predict(self, scores, groups):
    """Output DP fair class assignments given scores.

    Args:
      scores: array-like, shape (n_examples, n_classes)
        Scores/class probabilities of each example.
      groups: array-like, shape (n_examples,)
        Group label (zero-indexed) of each example; the sensitive attribute.

    Returns:
      array-like, shape (n_examples,)
      Fair class assignments (zero-indexed) of each example.
    """
    scores = check_array(scores)
    groups = check_array(groups, ensure_2d=False)
    scores, groups = check_X_y(scores, groups)
    check_is_fitted(self, "psi_by_group_")
    # argmin(2 * (1 - s) - \psi) = argmin(-2s - \psi) = argmax(2s + \psi)
    return np.argmin(-2 * scores - self.psi_by_group_[groups], axis=1)
