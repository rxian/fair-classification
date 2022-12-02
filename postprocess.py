from contextlib import redirect_stdout
import io
from itertools import chain

from cvxopt import matrix, spmatrix, solvers
import numpy as np
import scipy
from scipy.optimize import linprog
from sklearn import metrics
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from projection_simplex_vectorized import projection_simplex


class PostProcessorDP(BaseEstimator):
  """Post-processing mapping for DP fairness.

  Attributes:
    n_classes_: int
      Number of classes.
    n_groups_: int
      Number of demographic groups.
    score_: float
      Classification error of training examples after post-processing using
      Monge transports.
    barycenter_: array-like, shape (n_classes,)
      Wasserstein-barycenter of class probabilities.
    qs_: list of array-like, shape (n_classes,)
      Class probabilities of each demographic group.
    psis_: array-like, shape (n_groups, n_classes)
      Parameter of post-processing mapping for each group.
    gammas_: array-like, shape (n_groups, n_examples, n_classes)
      Monge transports to barycenter for each group.
  """

  def fit(self, S, a, eps=0.0, w=None, p=None):
    """Estimate a post-processing mapping.

    Args:
      S: array-like, shape (n_examples, n_classes)
        Predicted class probabilities of each example.
      a: array-like, shape (n_examples,)
        Group membership of each example.
      eps: float, optional
        Relaxation of the Wasserstein barycenter.  Default is 0.
      w: array-like, shape (n_groups,), optional
        Weight of each group, used for weighted Wasserstein-barycenter.
        Default is uniform (group-balanced Wasserstein-barycenter).
      p: array-like, shape (n_groups, n_classes), optional
        Probability mass of each example.  Default is uniform.
    """
    S, a = check_X_y(S, a)
    if p is not None:
      _, p = check_X_y(S, p)

    self.n_classes_ = S.shape[-1]
    self.n_groups_ = int(1 + np.max(a))

    # Group predicted class probas by sensitive attribute
    Ss = [S[a == i] for i in range(self.n_groups_)]
    ps = None if p is None else [
        p[a == i] / np.sum(p[a == i]) for i in range(self.n_groups_)
    ]
    w = None if w is None else np.array(w) / np.sum(w)

    gammas, cost, barycenter = self.find_barycenter_and_kantorovich_transports_(
        Ss, eps, w, ps)
    self.score_ = cost
    self.gammas_ = gammas
    self.barycenter_ = barycenter
    self.qs_ = [gamma.sum(axis=0) / gamma.sum() for gamma in gammas]
    self.psis_ = np.stack([
        self.extract_monge_transport_(Ss[i], gammas[i])
        for i in range(self.n_groups_)
    ])
    return self

  def find_barycenter_and_kantorovich_transports_(self,
                                                  Ss,
                                                  eps=0.0,
                                                  w=None,
                                                  ps=None):
    """Find barycenter and optimal discrete transport for each group.
       This implements the OPT linear program in the paper."""

    # Design variables are the probability mass of the couplings, followed by
    # the barycenter, and the slack variables.
    #
    # They are flattened into a single vector, which when unraveled is a 3d
    # tensor of shape (n_groups, n_examples, n_classes), followed by a vector of
    # shape (n_classes,), followed by a 2d tensor shaped (n_groups, n_classes).

    def ravel_index(multi_index):
      """Get index to flattened pmfs of the couplings from multi-index of group,
         example index, and class index."""
      indices = np.empty(len(multi_index[0]), dtype=np.uint)
      group_offset = 0
      for a in range(self.n_groups_):
        indices[multi_index[0] == a] = group_offset + np.ravel_multi_index(
            [idx[multi_index[0] == a] for idx in multi_index[1:]],
            (len(Ss[a]), self.n_classes_))
        group_offset += len(Ss[a]) * self.n_classes_
      return [int(x) for x in indices]

    ns = [len(S) for S in Ss]
    a_max = np.argmax(ns)

    # \ell_1 transportation costs
    c = []
    for a in range(self.n_groups_):
      s = np.repeat(np.arange(ns[a]), self.n_classes_, axis=0)
      y = np.tile(np.arange(self.n_classes_), ns[a])
      costs = 1 - Ss[a][s, y]
      if w is not None:
        costs *= w[a]
      # Normalize, due to the upscaling described below
      costs *= ns[a_max] / ns[a]
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

    # \sum_y \gamma_{a, s, y} = ps_{a, s}
    for a in range(self.n_groups_):
      for s in range(ns[a]):
        A_eqs_sparse_i.extend([row_eqs] * self.n_classes_)
        A_eqs_sparse_j.extend(
            ravel_index([
                np.full(self.n_classes_, a),
                np.full(self.n_classes_, s),
                np.arange(self.n_classes_),
            ]))
        # Upscale by ns[a] to prevent underflow
        A_eqs_sparse_v.extend([1.0] * self.n_classes_)
        b_eqs.append(
            ps[a][s] *
            ns[a] if ps is not None else 1.0)  # (1 / ns[a]) * ns[a] = 1
        row_eqs += 1

    # -\xi_{a, y} <= \sum_s \gamma_{a, s, y} - q_{y} <= \xi_{a, y}
    for a in range(self.n_groups_):
      for y in range(self.n_classes_):
        for i in [-1.0, 1.0]:
          A_ubs_sparse_i.extend([row_ubs] * ns[a])
          A_ubs_sparse_j.extend(
              ravel_index([
                  np.full(ns[a], a),
                  np.arange(ns[a]),
                  np.full(ns[a], y),
              ]))
          # Upscaled by ns[a] to prevent underflow
          A_ubs_sparse_v.extend([i] * ns[a])
          A_ubs_sparse_i.append(row_ubs)
          A_ubs_sparse_j.append(len(c) + y)
          A_ubs_sparse_v.append(-i * ns[a])
          A_ubs_sparse_i.append(row_ubs)
          A_ubs_sparse_j.append(len(c) + (a + 1) * self.n_classes_ + y)
          A_ubs_sparse_v.append(-1.0 * ns[a])
          b_ubs.append(0.0)
          row_ubs += 1

    # \sum_y \xi_{a, y} <= \epsilon
    for a in range(self.n_groups_):
      A_ubs_sparse_i.extend([row_ubs] * self.n_classes_)
      A_ubs_sparse_j.extend(
          range(
              len(c) + (a + 1) * self.n_classes_,
              len(c) + (a + 2) * self.n_classes_))
      A_ubs_sparse_v.extend([1.0] * self.n_classes_)
      b_ubs.append(eps)
      row_ubs += 1

    # \sum_y q_{y} = 1
    A_eqs_sparse_i.extend([row_eqs] * self.n_classes_)
    A_eqs_sparse_j.extend(range(len(c), len(c) + self.n_classes_))
    A_eqs_sparse_v.extend([1.0] * self.n_classes_)
    b_eqs.append(1.0)
    row_eqs += 1

    ## scipy.optimize.linprog implementation
    A_ubs = scipy.sparse.coo_matrix(
        (A_ubs_sparse_v, (A_ubs_sparse_i, A_ubs_sparse_j)),
        shape=(row_ubs, len(c) + (self.n_groups_ + 1) * self.n_classes_))
    A_eqs = scipy.sparse.coo_matrix(
        (A_eqs_sparse_v, (A_eqs_sparse_i, A_eqs_sparse_j)),
        shape=(row_eqs, len(c) + (self.n_groups_ + 1) * self.n_classes_))
    sol = linprog(np.concatenate(
        [c, [0.0] * (self.n_groups_ + 1) * self.n_classes_]),
                  A_ub=A_ubs,
                  b_ub=b_ubs,
                  A_eq=A_eqs,
                  b_eq=b_eqs,
                  bounds=(0, None),
                  method="highs")
    x = sol.x

    # Get optimal couplings
    gammas = []
    cost = 0
    group_offset = 0
    for a in range(self.n_groups_):
      this_gamma = x[group_offset:group_offset + ns[a] * self.n_classes_]
      gammas.append(this_gamma.reshape((ns[a], self.n_classes_)))
      this_c = c[group_offset:group_offset + ns[a] * self.n_classes_]
      cost += np.sum(this_gamma / np.sum(this_gamma) * this_c)
      group_offset += ns[a] * self.n_classes_
    return gammas, cost, x[len(c):len(c) + self.n_classes_]

  def extract_monge_transport_(self, S, gamma):
    """Extract estimated Monge transport an optimal coupling."""

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
        B[i, j] = np.max(S[idx, j] - S[idx, i] + 1, initial=0)
        A_ubs_sparse_i.extend([rows] * 2)
        A_ubs_sparse_j.extend([i, j])
        A_ubs_sparse_v.extend([1, -1])
        rows += 1

    # ``Fix'' numerical imprecisions
    B -= np.clip(B + B.T - 2, 0, None) / 2
    b_ubs = -B[np.where(~np.eye(B.shape[0], dtype=bool))] + 1

    # ## scipy.optimize.linprog implementation
    # A_ubs = scipy.sparse.coo_matrix(
    #     (A_ubs_sparse_v, (A_ubs_sparse_i, A_ubs_sparse_j)),
    #     shape=(rows, self.n_classes_))
    # sol = linprog(
    #     np.zeros(self.n_classes_),
    #     A_ub=A_ubs,
    #     b_ub=b_ubs,
    #     # A_eq=np.ones((1, self.n_classes_)),
    #     # b_eq=[1],
    #     bounds=None,
    #     method="highs")
    # Do projection instead of enforcing equality constraints, due to numerical
    # imprecisions
    # z = res.x - (res.x.sum() - 1) / self.n_classes_

    ## cvxopt.solvers.qp implementation
    h = matrix(b_ubs.reshape(-1, 1))
    G = spmatrix(A_ubs_sparse_v,
                 A_ubs_sparse_i,
                 A_ubs_sparse_j,
                 size=(rows, self.n_classes_))
    P = -1.0 / 2.0 * (G.T * G)
    q = 2.0 * G.T * h
    A = matrix(np.ones((1, self.n_classes_)))
    b = matrix(1.0)

    with redirect_stdout(io.StringIO()):
      sol = solvers.qp(P,
                       q,
                       G,
                       h,
                       A,
                       b,
                       kktsolver="ldl",
                       options={"kktreg": 1e-9})
    z = np.array(sol["x"]).squeeze()

    return np.array([0] +
                    [2 * (z[0] - z[j]) for j in range(1, self.n_classes_)])

  def predict(self, S, a):
    """Predict class labels given probas and group memberships.

    Args:
      S: array-like, shape (n_examples, n_classes)
        Predicted class probabilities of each example.
      a: array-like, shape (n_examples,)
        Group membership of each example.
    """
    check_is_fitted(self, "psis_")
    S = check_array(S)
    a = check_array(a, ensure_2d=False)
    S, a = check_X_y(S, a)
    return np.argmin(2 * (1 - S) - self.psis_[a], axis=1)


def postprocess(predict_fn,
                train_data,
                train_groups,
                eps=0,
                noise_fn=None,
                n_perturbations=0):
  """Post-process a predictor for DP with `PostProcessorDP'.

  Args:
    predict_fn: Forward function of predictor.
    train_data: array-like, shape (n_examples, n_features)
      Training data, input to `predict_fn'.
    train_groups: array-like, shape (n_examples,)
      Group membership of each example.
    eps: float, optional
      Trade-off parameter between accuracy and fairness, a smaller `eps' means
      more fairness.
    noise_fn: Function that takes as input a shape and outputs random noise of
      that shape.  Required if `n_perturbations' > 0.
    n_perturbations: int, optional
      Number of perturbed examples to be generated from from each example by
      adding `noise_fn'.  Default is 0.

  Returns: `PostProcessorDP' object
  """
  probas = predict_fn(train_data)
  if noise_fn is not None and n_perturbations > 0:
    probas = np.repeat(probas, n_perturbations, axis=0)
    train_groups = np.repeat(train_groups, n_perturbations, axis=0)
    probas += noise_fn(probas.shape)
    probas = projection_simplex(probas, axis=1)
  postprocessor = PostProcessorDP()
  postprocessor.fit(probas, train_groups, eps=eps)
  return postprocessor


def evaluate(predict_fn,
             postprocessor,
             test_data,
             test_labels,
             test_groups,
             n_labels,
             n_groups,
             noise_fn=None,
             n_perturbations=0):
  """Compare pre-trained predictor to its post-processed version.

  Metrics include accuracy, balanced accuracy, and the maximum pairwise DP
  fairness gap in l_1 and l_inf norms.

  Returns:
    Dictionary with metrics.
  """
  probas = predict_fn(test_data)
  if noise_fn is not None and n_perturbations > 0:
    probas = np.repeat(probas, n_perturbations, axis=0)
    test_labels = np.repeat(test_labels, n_perturbations, axis=0)
    test_groups = np.repeat(test_groups, n_perturbations, axis=0)
    probas += noise_fn(probas.shape)
    probas = projection_simplex(probas, axis=1)

  y_predicted = np.argmax(probas, axis=1)
  y_postprocessed = postprocessor.predict(probas, test_groups)

  res = {}
  res["accuracy"] = {
      "predictor": metrics.accuracy_score(test_labels, y_predicted),
      "postprocessor": metrics.accuracy_score(test_labels, y_postprocessed),
  }

  _, counts = np.unique(test_groups, return_counts=True)
  weights = np.sum(counts) / counts[test_groups]
  res["balanced_accuracy"] = {
      "predictor":
          metrics.accuracy_score(test_labels,
                                 y_predicted,
                                 sample_weight=weights),
      "postprocessor":
          metrics.accuracy_score(test_labels,
                                 y_postprocessed,
                                 sample_weight=weights),
  }

  class_dist_predicted = np.zeros((n_groups, n_labels))
  class_dist_postprocessed = np.zeros((n_groups, n_labels))
  for a in range(n_groups):
    y, counts = np.unique(y_predicted[test_groups == a], return_counts=True)
    class_dist_predicted[a, y] = counts / np.sum(counts)
    y, counts = np.unique(y_postprocessed[test_groups == a], return_counts=True)
    class_dist_postprocessed[a, y] = counts / np.sum(counts)

  diff_predicted = np.abs(class_dist_predicted[:, None, :] -
                          class_dist_predicted[None, :, :])
  diff_postprocessed = np.abs(class_dist_postprocessed[:, None, :] -
                              class_dist_postprocessed[None, :, :])

  res["dp_gap_linf_max"] = {
      "predictor": np.max(diff_predicted),
      "postprocessor": np.max(diff_postprocessed),
  }
  res["dp_gap_l1_max"] = {
      "predictor": np.max(1 / 2 * np.sum(diff_predicted, axis=2)),
      "postprocessor": np.max(1 / 2 * np.sum(diff_postprocessed, axis=2)),
  }
  res["dp_gap_l1_avg"] = {
      "predictor":
          np.mean(1 / 2 *
                  np.sum(diff_predicted, axis=2)[np.triu_indices(n_groups, 1)]),
      "postprocessor":
          np.mean(
              1 / 2 *
              np.sum(diff_postprocessed, axis=2)[np.triu_indices(n_groups, 1)]),
  }

  return res
