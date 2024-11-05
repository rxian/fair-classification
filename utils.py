from functools import partial
import traceback

import numpy as np
import pandas as pd
import sklearn.metrics
import tqdm

from joblib import Parallel, delayed
# from tqdm.contrib.concurrent import process_map

from models import BinningCalibrator
import postprocess


def error_rate(y_true, y_preds, groups=None, w=None, n_groups=None):
  """Compute group-weighted error rate."""
  if groups is None or w is None:
    return np.mean(y_true != y_preds)
  else:
    if n_groups is None:
      group_names, groups = np.unique(groups, return_inverse=True)
      n_groups = len(group_names)
    return sum([
        w[a] * np.mean(y_true[groups == a] != y_preds[groups == a])
        for a in range(n_groups)
    ])


def delta_sp(y_preds, groups, n_classes, n_groups, ord=np.inf):
  """Compute violation of statistical parity."""
  pred_counts = np.array([
      np.bincount(y_preds[groups == a], minlength=n_classes)
      for a in range(n_groups)
  ])
  output_dists = pred_counts / np.sum(pred_counts, axis=1, keepdims=True)
  diffs = np.linalg.norm(output_dists[:, None, :] - output_dists[None, :, :],
                         ord=ord,
                         axis=2)
  return np.max(diffs)


def confusion_matrix(y_true, y_preds, groups, n_classes, n_groups):
  """Compute group-wise confusion matrices (conditioned on y_true)."""
  return np.array([
      sklearn.metrics.confusion_matrix(y_true[groups == a],
                                       y_preds[groups == a],
                                       labels=np.arange(n_classes),
                                       normalize='true')
      for a in range(n_groups)
  ])


def delta_eo(y_true, y_preds, groups, n_classes, n_groups, ord=np.inf):
  """Compute violation of equalized odds."""
  conf_mtxs = confusion_matrix(
      y_true,
      y_preds,
      groups,
      n_classes,
      n_groups,
  ).reshape(n_groups, -1)  # shape = (n_groups, n_classes**2)
  with np.errstate(invalid='ignore'):  # Ignore groups with no positive examples
    # Pairwise differences
    diffs = np.linalg.norm(conf_mtxs[:, None, :] - conf_mtxs[None, :, :],
                           ord=ord,
                           axis=2)
    diffs = np.nan_to_num(diffs, nan=0.0)
  return np.max(diffs)


def delta_eopp(y_true, y_preds, groups, n_classes, n_groups, ord=np.inf):
  """
  Compute violation of (binary or multi-class) equalized opportunity (depending
  on `n_classes`).
  """
  conf_mtxs = confusion_matrix(
      y_true, y_preds, groups, n_classes,
      n_groups)  # shape = (n_groups, n_classes, n_classes)
  tprs = np.array([np.diag(conf_mtx) for conf_mtx in conf_mtxs
                  ])  # shape = (n_groups, n_classes)
  if n_classes == 2:
    tprs = tprs[:, 1].reshape(-1, 1)  # shape = (n_groups, 1)
  with np.errstate(invalid='ignore'):  # Ignore groups with no positive examples
    # Pairwise differences
    diffs = np.linalg.norm(tprs[:, None, :] - tprs[None, :, :], ord=ord, axis=2)
    diffs = np.nan_to_num(diffs, nan=0.0)
  return np.max(diffs)


def calibration_error(probas, labels, n_bins, seed=0):
  """Computes binned expected calibration error, with bins selected by k-means.
  """
  calib = BinningCalibrator(n_bins=n_bins,
                            random_state=seed).fit(probas, labels)
  # bins = calib.binning_fn_(probas)
  # bin_to_proba = {b: probas[bins == b].mean(axis=0) for b in np.unique(bins)}
  # probas_binned = np.array([bin_to_proba[b] for b in bins])
  p = np.mean(probas, axis=0)
  probas_cal = calib.predict_proba(probas)
  p_cal = np.mean(probas_cal, axis=0)
  return np.max(np.mean(np.abs(probas / p - probas_cal / p_cal), axis=0))


# Define some utility functions


def postprocess_and_evaluate(alphas,
                             seeds,
                             criterion,
                             metrics,
                             n_test,
                             n_classes,
                             n_groups,
                             labels,
                             groups,
                             probas_y=None,
                             probas_a=None,
                             probas_ay=None,
                             calibrator_factory=None,
                             max_workers=1,
                             postproc_kwargs=None,
                             return_vals=False,
                             print_code=False):

  ## This wrapper is for our algorithm defined in postprocess.PostProcessor

  if postproc_kwargs is None:
    postproc_kwargs = {}

  if print_code:
    if probas_ay is not None:
      print(
          f'''Code for post-processing a single model (with precomputed probas):

    postprocessor = postprocess.PostProcessor(
        n_classes,
        n_groups,
        pred_ay_fn=lambda x: x,  # dummy pred_fn
        criterion='{criterion}',
        alpha=alpha,
        seed=seed,
    )
    postprocessor.fit(probas_ay_postproc)
    preds = postprocessor.predict(probas_ay_test)''')
    else:
      print(
          f'''Code for post-processing a single model (with precomputed probas):

    postprocessor = postprocess.PostProcessor(
        n_classes,
        n_groups,
        pred_a_fn=lambda x: x[0],  # dummy pred_fns
        pred_y_fn=lambda x: x[1],
        criterion='{criterion}',
        alpha=alpha,
        seed=seed,
    )
    postprocessor.fit([probas_a_postproc, probas_y_postproc])
    preds = postprocessor.predict((probas_a_test, probas_y_test))''')

  fn = partial(
      postprocess_and_evaluate_,
      postprocessor_factory=partial(postprocess.PostProcessor,
                                    criterion=criterion),
      metrics=metrics,
      n_test=n_test,
      n_classes=n_classes,
      n_groups=n_groups,
      labels=labels,
      groups=groups,
      probas_y=probas_y,
      probas_a=probas_a,
      probas_ay=probas_ay,
      calibrator_factory=calibrator_factory,
  )

  if max_workers == 1:
    res = []
    for alpha in alphas:
      for seed in seeds:
        res.append(fn((alpha, seed, postproc_kwargs)))
        # print(res[-1])  # to monitor progress
  else:

    alpha_seed_and_kwargs = [
        (alpha, seed, postproc_kwargs) for alpha in alphas for seed in seeds
    ]

    res = Parallel(n_jobs=max_workers)(
        delayed(fn)(alpha_seed_and_kwargs[i])
        for i in tqdm.tqdm(range(len(alpha_seed_and_kwargs))
                          ))  # each val = (alpha, seed, metrics, postprocessor)

    ## process_map does not work with sklearn
    # res = process_map(
    #     fn,
    #     alpha_seed_and_kwargs,
    #     max_workers=max_workers,
    # )  # each val = (alpha, seed, metrics, postprocessor)

  ret = pd.DataFrame([{
      'alpha': alpha,
      **result
  } for alpha, _, result, _ in res if result is not None
                     ]).groupby('alpha').agg(['mean', np.std
                                             ]).sort_index(ascending=False)
  if return_vals:
    return ret, res
  return ret


def dict_get_key(d, k):
  return d[k]


def postprocess_and_evaluate_(
    alpha_seed_and_kwargs,
    postprocessor_factory,
    metrics,
    n_test,
    n_classes,
    n_groups,
    labels,
    groups,
    probas_y=None,
    probas_a=None,
    probas_ay=None,
    calibrator_factory=None,
):

  if len(alpha_seed_and_kwargs) == 2:
    alpha, seed = alpha_seed_and_kwargs
    kwargs = {}
  else:
    alpha, seed, kwargs = alpha_seed_and_kwargs
  # kwargs can contain, e.g., solver settings

  # Split the remaining data into post-processing and test data
  n_total = len(labels)
  idx_test = np.random.default_rng(seed).choice(np.arange(n_total),
                                                size=n_test,
                                                replace=False)
  idx_postproc = np.setdiff1d(np.arange(n_total), idx_test)

  labels_postproc = labels[idx_postproc]
  groups_postproc = groups[idx_postproc]
  labels_test = labels[idx_test]
  groups_test = groups[idx_test]

  # Create dummy prediction functions.  The probas are precomputed, and are
  # stored in `probas_postproc` and `probas_test`, dictionaries with keys
  # 'pred_y_fn', 'pred_a_fn' and/or 'pred_ay_fn', whose values are `probas_y`,
  # `probas_a` and `probas_ay`, respectively.  To "predict" the probas, we
  # simply get the value from the dictionary with the corresponding key, which
  # is what `pred_fns` do.
  pred_fns = {}
  probas_postproc = {}
  probas_test = {}
  for n, p in zip(['y', 'a', 'ay'], [probas_y, probas_a, probas_ay]):
    if n == 'y':
      target = labels_postproc
    elif n == 'a':
      target = groups_postproc
    else:
      target = groups_postproc * n_classes + labels_postproc
    n = f'pred_{n}_fn'
    if p is not None:
      if calibrator_factory is not None:
        calib = calibrator_factory(random_state=seed)
        # print(p[idx_postproc].reshape(len(idx_postproc), -1).shape)
        # asfas
        calib.fit(p[idx_postproc].reshape(len(idx_postproc), -1), target)
        # dfasf
        p = calib.predict_proba(p.reshape(len(p), -1)).reshape(p.shape)
      pred_fns[n] = partial(dict_get_key, k=n)
      probas_postproc[n] = p[idx_postproc]
      probas_test[n] = p[idx_test]

  if alpha == np.inf:
    # Evaluate the unprocessed model
    postprocessor = None
    if probas_y is None:
      probas_test_ = probas_test['pred_ay_fn'].sum(axis=1)
    else:
      probas_test_ = probas_test['pred_y_fn']
    preds_test = probas_test_.argmax(axis=1)
  else:
    try:
      # Post-process the predicted probabilities
      postprocessor = postprocessor_factory(
          n_classes=n_classes,
          n_groups=n_groups,
          alpha=alpha,
          seed=seed,
          **pred_fns,
      ).fit(probas_postproc, **kwargs)
      # Evaluate the post-processed model
      preds_test = postprocessor.predict(probas_test)
    except Exception:
      print(
          f"Post-processing failed with alpha={alpha} and seed={seed}:\n{traceback.format_exc()}",
          flush=True)
      return alpha, seed, None, None

  return alpha, seed, evaluate(labels_test,
                               preds_test,
                               groups_test,
                               n_groups=n_groups,
                               n_classes=n_classes,
                               metrics=metrics), postprocessor


def evaluate(test_labels,
             test_preds,
             test_groups,
             n_groups=2,
             n_classes=2,
             metrics=[]):
  result = {}
  for metric in metrics:
    if metric == 'accuracy':
      result[metric] = 1 - error_rate(
          test_labels,
          test_preds,
          test_groups,
          n_groups=n_groups,
      )
    elif metric.startswith('delta_sp'):
      result[metric] = delta_sp(
          test_preds,
          test_groups,
          n_classes=n_classes,
          n_groups=n_groups,
          ord=2 if metric.endswith('rms') else np.inf,
      ) / (np.sqrt(n_classes) if metric.endswith('rms') else 1)
    elif metric.startswith('delta_eopp'):
      result[metric] = delta_eopp(
          test_labels,
          test_preds,
          test_groups,
          n_classes=n_classes,
          n_groups=n_groups,
          ord=2 if metric.endswith('rms') else np.inf,
      ) / (np.sqrt(n_classes) if
           (metric.endswith('rms') and n_classes > 2) else 1)
    elif metric.startswith('delta_eo'):
      result[metric] = delta_eo(
          test_labels,
          test_preds,
          test_groups,
          n_classes=n_classes,
          n_groups=n_groups,
          ord=2 if metric.endswith('rms') else np.inf,
      ) / (n_classes if metric.endswith('rms') else 1)
    elif metric.startswith('dist'):
      label = int(metric.split('_')[-1])
      result[metric] = (test_preds == label).mean()
  return result


def plot_results(ax, df, x_col, y_col, label=None, **kwargs):
  if 'fmt' not in kwargs:
    kwargs['fmt'] = '-'
  markers, caps, bars = ax.errorbar(
      df[x_col]['mean'].values,
      df[y_col]['mean'].values,
      xerr=df[x_col]['std'].values,
      yerr=df[y_col]['std'].values,
      lw=2,
      label=label,
      **kwargs,
  )
  for b in bars:
    b.set_alpha(0.4)
  ax.set_xlabel(x_col)
  ax.set_ylabel(y_col)
  ax.grid(True, which="both", zorder=0)
