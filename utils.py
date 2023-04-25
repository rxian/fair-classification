import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def delta_dp(y_preds, groups, n_classes=None, n_groups=None, ord=np.inf):
  """Compute difference in output distributions."""
  if n_classes is None:
    class_names, y_preds = np.unique(y_preds, return_inverse=True)
    n_classes = len(class_names)
  if n_groups is None:
    group_names, groups = np.unique(groups, return_inverse=True)
    n_groups = len(group_names)
  pred_counts = np.array([
      np.bincount(y_preds[groups == a], minlength=n_classes)
      for a in range(n_groups)
  ])
  output_dists = pred_counts / np.sum(pred_counts, axis=1, keepdims=True)
  diffs = np.linalg.norm(output_dists[:, None, :] - output_dists[None, :, :],
                         ord=ord,
                         axis=2)
  return np.max(diffs)


def perturb(s, repeat, bw=1.0, eps=10, seed=33):
  """Perturb points on the simplex via the dirichlet distribution."""
  s = np.array(s)
  scale = np.min([
      np.min(s + eps * bw, axis=1, keepdims=True),
      np.min(1 - s + eps * bw, axis=1, keepdims=True)
  ],
                 axis=0)
  scale = (scale * (1 - scale)) / bw
  s = (s + eps * bw) * scale
  return np.concatenate(
      [np.random.default_rng(seed).dirichlet(r, size=repeat) for r in s],
      axis=0)


# Define some utility functions


def postprocess(alpha_seed_and_kwargs,
                postprocessor_factory,
                evaluate_fn,
                probas,
                labels,
                groups,
                n_test,
                n_post,
                n_perturbs=0,
                perturb_fn=None):

  if len(alpha_seed_and_kwargs) == 2:
    alpha, seed = alpha_seed_and_kwargs
    kwargs = {}
  else:
    alpha, seed, kwargs = alpha_seed_and_kwargs

  # Split the remaining data into post-processing and test data
  idx_post = np.random.default_rng(seed).choice(np.arange(n_test + n_post),
                                                size=n_post,
                                                replace=False)
  idx_test = np.setdiff1d(np.arange(n_test + n_post), idx_post)

  train_probas_post = probas[idx_post]
  train_labels_post = labels[idx_post]
  train_groups_post = groups[idx_post]
  test_probas = probas[idx_test]
  test_labels = labels[idx_test]
  test_groups = groups[idx_test]

  if n_perturbs > 0:
    train_probas_post = perturb_fn(train_probas_post, n_perturbs)
    train_labels_post = np.repeat(train_labels_post, n_perturbs)
    train_groups_post = np.repeat(train_groups_post, n_perturbs)
    test_probas = perturb_fn(test_probas, 1)

  if alpha == np.inf:
    # Evaluate the unprocessed model
    postprocessor = None
    test_preds = test_probas.argmax(axis=1)
  else:
    try:
      # Post-process the predicted probabilities
      postprocessor = postprocessor_factory().fit(train_probas_post,
                                                  train_groups_post,
                                                  alpha=alpha,
                                                  **kwargs)
      # Evaluate the post-processed model
      test_preds = postprocessor.predict(test_probas, test_groups)
    except Exception:
      print(f"Post-processing failed with alpha={alpha} and seed={seed}:")
      print(traceback.format_exc())
      return alpha, seed, None, None

  return alpha, seed, evaluate_fn(test_labels, test_preds,
                                  test_groups), postprocessor


def evaluate(test_labels,
             test_preds,
             test_groups,
             n_groups=2,
             n_classes=2,
             metrics=None):
  result = {}
  result['error_rate'] = error_rate(
      test_labels,
      test_preds,
      test_groups,
      n_groups=n_groups,
  )
  if metrics is not None and 'delta_dp' in metrics:
    result['delta_dp'] = delta_dp(
        test_preds,
        test_groups,
        n_classes=n_classes,
        n_groups=n_groups,
    )
  if metrics is not None and 'delta_dp_rms' in metrics:
    result['delta_dp_rms'] = delta_dp(
        test_preds,
        test_groups,
        n_classes=n_classes,
        n_groups=n_groups,
        ord=2,
    ) / np.sqrt(n_classes)
  return result


def plot_results(results, metric):
  df = pd.DataFrame(results).groupby('alpha').agg(['mean', np.std
                                                  ]).sort_index(ascending=False)
  fig, ax = plt.subplots(1, 1, figsize=(3.5, 3))
  df1 = df[df.index != np.inf]
  markers, caps, bars = ax.errorbar(
      df1[metric]['mean'].values,
      df1['error_rate']['mean'].values,
      xerr=df1[metric]['std'].values,
      yerr=df1['error_rate']['std'].values,
      fmt='o',
  )
  [bar.set_alpha(0.4) for bar in bars]
  if np.inf in df.index:
    # Plot the unprocessed model
    df2 = df[df.index == np.inf]
    markers, caps, bars = ax.errorbar(
        df2[metric]['mean'].values,
        df2['error_rate']['mean'].values,
        xerr=df2[metric]['std'].values,
        yerr=df2['error_rate']['std'].values,
        fmt='o',
        color='tab:blue',
        markerfacecolor='w',
    )
    [bar.set_alpha(0.4) for bar in bars]
  ax.set_xlabel(metric)
  ax.set_ylabel("Error rate")
  ax.grid(True, which="both", zorder=0)
  return (fig, ax), df
