import numpy as np


def error_rate(y, y_preds, groups=None, w=None, n_groups=None):
  """Compute group-weighted error rate."""
  if groups is None or w is None:
    return np.mean(y != y_preds)
  else:
    if n_groups is None:
      group_names, groups = np.unique(groups, return_inverse=True)
      n_groups = len(group_names)
    return sum([
        w[a] * np.mean(y[groups == a] != y_preds[groups == a])
        for a in range(n_groups)
    ])


def dp_gap(y_preds, groups, n_classes=None, n_groups=None):
  """Compute DP gap."""
  if n_classes is None:
    class_names, y_preds = np.unique(y_preds, return_inverse=True)
    n_classes = len(class_names)
  if n_groups is None:
    group_names, groups = np.unique(groups, return_inverse=True)
    n_groups = len(group_names)

  output_dists = np.array([
      np.bincount(y_preds[groups == a], minlength=n_classes)
      for a in range(n_groups)
  ])
  output_dists = output_dists / np.sum(output_dists, axis=1, keepdims=True)
  diffs = np.abs(output_dists[:, None, :] - output_dists[None, :, :])
  return np.max(diffs)


def perturb(rng, s, bw=1.0, repeat=1, eps=10):
  """Perturb points on the simplex via the dirichlet distribution."""
  s = np.array(s)
  scale = np.min([
      np.min(s + eps * bw, axis=1, keepdims=True),
      np.min(1 - s + eps * bw, axis=1, keepdims=True)
  ],
                 axis=0)
  scale = (scale * (1 - scale)) / bw
  s = (s + eps * bw) * scale
  return np.concatenate([rng.dirichlet(r, size=repeat) for r in s], axis=0)
