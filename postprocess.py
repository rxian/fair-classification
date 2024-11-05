"""Post-processing for fair classification."""

from typing import Any, Callable, Dict, Optional, Tuple
from typing_extensions import Self

import numpy as np
import cvxpy as cp


class PostProcessor:
  """
  A post-processor on top of pre-trained predictors for achieving fair
  classification (maximizing for classification accuracy).
  """

  def __init__(self,
               n_classes: int,
               n_groups: int,
               pred_a_fn: Optional[Callable] = None,
               pred_y_fn: Optional[Callable] = None,
               pred_ay_fn: Optional[Callable] = None,
               criterion: str = 'sp',
               alpha: float = 0.001,
               noise: float = 1e-4,
               seed: Optional[int] = None) -> None:
    """
    Initialize the post-processor.

    For `eo` and `eopp` criteria, a predictor for A and Y given X is required.
    Output shape of `pred_ay_fn` should be (batch_size, n_groups, n_classes),
    or (batch_size, n_groups * n_classes) if flattened (unraveled).

    Args:
      n_classes (int): Number of classes.
      n_groups (int): Number of categories for the sensitive attribute A.
      pred_a_fn (function, optional): Function to predict A given X.
      pred_y_fn (function, optional): Function to predict Y given X.
      pred_ay_fn (function, optional): Function to predict A and Y given X.
      criterion (str, optional): Fairness criterion.
          `sp` for statistical parity, `eopp` for (binary or multi-class) equal
          opportunity (depending on `n_classes`), and `eo` for equalized odds.
      alpha (float, optional): Fairness tolerance.
      noise (float, optional): Factor for the width of uniform random noise used
          to perturb the risk.
      seed (int, optional): Seed for random number generator.
    """
    self.n_classes = n_classes
    self.n_groups = n_groups
    self.pred_a_fn = pred_a_fn
    self.pred_y_fn = pred_y_fn
    self.pred_ay_fn = pred_ay_fn
    self.criterion = criterion
    self.alpha = alpha
    self.noise = noise
    self.rng = np.random.default_rng(seed)
    self.cls_loss_fn = 1 - np.eye(n_classes)

    if criterion not in ['sp', 'eopp', 'eo']:
      raise ValueError("criterion must be one of `sp`, `eopp`, `eo`")
    if criterion == 'sp' and (pred_ay_fn is None and
                              (pred_a_fn is None or pred_y_fn is None)):
      raise ValueError(
          '(pred_a_fn and pred_y_fn) or pred_ay_fn must be provided for `sp` criterion'
      )
    if criterion in ['eopp', 'eo'] and pred_ay_fn is None:
      raise ValueError(
          'pred_ay_fn must be provided for `eopp` or `eo` criterion')

  # TODO: sample weight
  def fit(self,
          x: np.ndarray,
          solver: str = cp.GUROBI,
          solve_kwargs: Optional[Dict[str, Any]] = None,
          solve_primal: bool = True) -> Self:
    """
    Fit the post-processor.

    Args:
      x (array-like): Input data.
      solver (str, optional): LP solver from `cvxpy` to use.
      solve_kwargs (dict, optional): Keyword arguments for the solver.
      solve_primal (bool, optional): Whether to solve the primal problem.

    If Gurobi is not available, a (slower) alternative is
    `solver=cp.CBC, solve_kwargs={'integerTolerance': 1e-8}, solve_primal=False`

    There are two ways to solve for the parameters of the post-processor, (1)
    solve the primal problem and extract the dual values (solve_primal=True), or
    (2) solve the dual problem directly (solve_primal=False).  The former is
    usually faster, but not all solvers support it (e.g., CBC).

    Returns:
      self: Returns an instance of the PostProcessor object.
    """
    solve_kwargs = solve_kwargs or {}

    (risk, constraint_gamma, constraint_y, p_a,
     p_ay) = self.compute_risk_and_constraint_(x)

    # Perturb risk to circumvent colinearity
    self.risk_mean_ = np.mean(risk)
    risk += self.risk_mean_ * self.rng.uniform(
        -self.noise, self.noise, size=risk.shape)

    if solve_primal:
      problem = self.linprog_primal_(risk, constraint_gamma, constraint_y,
                                     self.alpha)
      problem.solve(solver=solver, **solve_kwargs)
      n_constraints = constraint_gamma.shape[1]
      self.psi_ = (np.array([
          c.dual_value for c in problem.constraints[-2 * n_constraints::2]
      ]) - np.array([
          c.dual_value for c in problem.constraints[-2 * n_constraints + 1::2]
      ]))
      self.phi_ = -problem.constraints[0].dual_value
      self.pi_ = problem.var_dict['pi'].value
    else:
      problem = self.linprog_dual_(risk, constraint_gamma, constraint_y,
                                   self.alpha)
      problem.solve(solver=solver, **solve_kwargs)
      self.psi_ = problem.var_dict['psi_pos'].value - problem.var_dict[
          'psi_neg'].value
      self.phi_ = problem.var_dict['phi'].value

    # TODO: catch situations where the solver fails (i.e., numerical issues)

    self.score_ = problem.value
    self.risk_ = risk  # for debugging
    self.constraint_gamma_ = constraint_gamma
    self.p_a_ = p_a
    self.p_ay_ = p_ay
    return self

  def predict_score(self, x: np.ndarray) -> np.ndarray:
    """
    Post-process the riskes of the input data by adding the cost of fairness.

    Args:
      x (array-like): Input data.

    Returns:
      array-like: Post-processed risk values.
    """
    risk, constraint_gamma, constraint_y, _, _ = self.compute_risk_and_constraint_(
        x, self.p_a_, self.p_ay_)

    # Perturb risk to circumvent colinearity
    risk += self.risk_mean_ * self.rng.uniform(
        -self.noise, self.noise, size=risk.shape)

    mask_y = np.where(
        constraint_y[None, :] == np.arange(self.n_classes)[:, None], 1, 0)
    fair_cost = np.sum(
        np.sum(self.psi_ * constraint_gamma, axis=-1)[:, None, :] *
        mask_y[None, :, :],
        axis=-1)  # shape = (n_examples, n_classes)

    fair_risk = risk - fair_cost
    return fair_risk

  def predict(self, x: np.ndarray) -> np.ndarray:
    """
    Make fair predictions for the input data.

    Args:
      x (array-like): Input data.

    Returns:
      array-like: Predicted class labels.
    """
    fair_risk = self.predict_score(x)
    return np.argmin(fair_risk, axis=1)

  def compute_risk_and_constraint_(
      self,
      x: np.ndarray,
      p_a: Optional[np.ndarray] = None,
      p_ay: Optional[np.ndarray] = None
  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the risk and constraints from the input data. Required for fitting
    and prediction.

    Args:
      x (array-like): Input data.
      p_a (array-like, optional): Probabilities of A given X.
      p_ay (array-like, optional): Probabilities of A and Y given X.

    Returns:
      tuple: Tuple containing risk, constraint_gamma, constraint_y, p_a, and p_ay.
    """
    # mask.shape = (n_classes, n_constraints)
    # risk.shape = (n_examples, n_classes)
    # gamma.shape = (n_examples, n_constraints, n_events)

    if self.criterion == 'sp':

      # Get predicted p(A | X) and p(Y | X)
      if self.pred_ay_fn is not None:
        p_ay_x = self.pred_ay_fn(x).reshape(-1, self.n_groups, self.n_classes)
        p_a_x = p_ay_x.sum(axis=2)
        p_y_x = p_ay_x.sum(axis=1)
      else:
        p_a_x = self.pred_a_fn(x)
        p_y_x = self.pred_y_fn(x)

      if p_a is None:
        p_a = p_a_x.mean(axis=0)  # shape = (n_groups,)

      constraint_y = np.arange(self.n_classes)
      constraint_gamma = np.repeat((p_a_x / p_a)[:, None, :],
                                   self.n_classes,
                                   axis=1)

    if self.criterion in ['eopp', 'eo']:

      p_ay_x = self.pred_ay_fn(x).reshape(
          -1, self.n_groups,
          self.n_classes)  # shape = (n_examples, n_groups, n_classes)
      p_y_x = p_ay_x.sum(axis=1)  # shape = (n_examples, n_classes)

      if p_ay is None:
        p_ay = p_ay_x.mean(axis=0)  # shape = (n_groups, n_classes)

      constraint_y = []
      constraint_gamma = []
      for y_ in range(self.n_classes):
        for y in range(self.n_classes):
          if self.criterion == 'eopp' and (y != y_ or
                                           (self.n_classes == 2 and y == 0)):
            continue
          constraint_y.append(y_)
          constraint_gamma.append(p_ay_x[:, :, y] / p_ay[:, y])
      constraint_y = np.array(constraint_y)
      constraint_gamma = np.array(constraint_gamma).transpose(1, 0, 2)

    risk = np.sum(p_y_x[:, :, None] * self.cls_loss_fn[None, :],
                  axis=1)  # shape = (n_examples, n_classes)

    return risk, constraint_gamma, constraint_y, p_a, p_ay

  def linprog_primal_(self, risk: np.ndarray, constraint_gamma: np.ndarray,
                      constraint_y: np.ndarray, alpha: float) -> cp.Problem:
    """
    Solve the fair classification problem in primal LP formulation.

    Args:
      risk (array-like): Risk values.
      constraint_gamma (array-like): Constraint function values.
      constraint_y (array-like): Classes to be constrained.
      alpha (float): Fairness tolerance.

    Returns:
      cp.Problem: Linear programming problem.
    """
    n_examples = risk.shape[0]
    n_constraints = constraint_gamma.shape[1]

    alpha = cp.Parameter(value=alpha, name="alpha")
    pi = cp.Variable((n_examples, self.n_classes), name="pi", nonneg=True)
    q = cp.Variable(n_constraints, name="q", nonneg=True)

    # Get constraints
    constraints = []

    # \sum_y \pi(y | x) = 1, for all x
    constraints.append(cp.sum(pi, axis=1) == 1)

    # | \sum_x \gamma_{i, j}(x) * \pi(y_i | x) * p(x) - q_i | <= \alpha / 2, for all i, j
    for i in range(n_constraints):
      t = cp.sum(cp.multiply(constraint_gamma[:, i],
                             pi[:, constraint_y[i]][:, None]),
                 axis=0)
      constraints.append(-alpha * n_examples / 2 <= t - q[i] * n_examples)
      constraints.append(t - q[i] * n_examples <= alpha * n_examples / 2)

    return cp.Problem(cp.Minimize(cp.sum(cp.multiply(pi, risk))), constraints)

  def linprog_dual_(self, risk: np.ndarray, constraint_gamma: np.ndarray,
                    constraint_y: np.ndarray, alpha: float) -> cp.Problem:
    """
    Solve the fair classification problem in dual LP formulation.

    Args:
      risk (array-like): Risk values.
      constraint_gamma (array-like): Constraint function values.
      constraint_y (array-like): Classes to be constrained.
      alpha (float): Fairness tolerance.

    Returns:
      cp.Problem: Linear programming problem.
    """
    n_examples = risk.shape[0]
    n_classes = risk.shape[1]
    n_constraints = constraint_gamma.shape[1]

    alpha = cp.Parameter(value=alpha, name="alpha")
    phi = cp.Variable(risk.shape[0], name="phi")
    psi_pos = cp.Variable(
        (constraint_gamma.shape[1], constraint_gamma.shape[2]),
        name="psi_pos",
        nonneg=True)
    psi_neg = cp.Variable(
        (constraint_gamma.shape[1], constraint_gamma.shape[2]),
        name="psi_neg",
        nonneg=True)

    # Get constraints
    constraints = []

    # \sum_j \psi_pos_{i, j} - \psi_neg_{i, j} = 0, for all i (*)
    constraints.append(cp.sum(psi_pos - psi_neg, axis=1) == 0)

    # \phi(x) + \sum_ij 1[y_i = y] * (\psi_pos_{i, j} - \psi_neg_{i, j}) * \gamma_{i, j}(x)
    #     <= \risk(x, y), for all x, y
    t = [0 for _ in range(n_classes)]
    for i in range(n_constraints):
      t[constraint_y[i]] += cp.sum(cp.multiply(
          constraint_gamma[:, i, :], (psi_pos[i, :] - psi_neg[i, :])[None, :]),
                                   axis=1)
    # constraints.append(phi[:, None] + t <= risk)
    for y, s in enumerate(t):
      constraints.append(phi + s <= risk[:, y])

    # Note that \sum_j \psi_pos_{i, j} = \sum_j \psi_neg_{i, j} because of constraint (*), so `/ 2` is removed
    return cp.Problem(
        cp.Maximize(cp.sum(phi) - alpha * cp.sum(psi_pos) * n_examples),
        constraints)
