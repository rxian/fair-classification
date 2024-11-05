import numpy as np
import torch
import sklearn, sklearn.cluster


class BinningCalibrator:

  def __init__(self,
               n_bins=None,
               binning_fn=None,
               prior_strength=0,
               random_state=None):
    self.n_bins = n_bins
    self.binning_fn_ = binning_fn
    self.prior_strength = prior_strength
    self.random_state = random_state

  def fit(self, P, y):
    if self.binning_fn_ is None:
      binning = sklearn.cluster.KMeans(n_clusters=self.n_bins,
                                       n_init='auto',
                                       random_state=self.random_state)
      bins = binning.fit_predict(P)
      self.binning_fn_ = binning.predict
    else:
      bins = self.binning_fn_(P)
    self.bin_counts_ = []
    self.bin_vals_true_ = []
    self.bin_vals_model_ = []
    self.bin_vals_ = []
    self.score_ = 0
    for b in range(np.max(bins) + 1):
      mask = bins == b
      p = np.bincount(y[mask], minlength=P.shape[1])
      true_p = p / mask.sum(axis=0)
      self.bin_vals_true_.append(true_p)
      model_p = P[mask].mean(axis=0)
      self.bin_vals_model_.append(model_p)
      counts = mask.sum()
      self.bin_counts_.append(counts)
      self.bin_vals_.append((true_p * counts + model_p * self.prior_strength) /
                            (counts + self.prior_strength))
      self.score_ += np.sum(np.abs(true_p - model_p)) * mask.sum() / len(y)
    return self

  def predict_proba(self, P):
    bins = self.binning_fn_(P)
    return np.stack([self.bin_vals_[b] for b in bins], axis=0)

  def predict(self, P):
    return self.predict_proba(P).argmax(axis=1)


class MLPClassifier:

  def __init__(self,
               hidden_layer_sizes=(100, 100),
               activation=torch.nn.ReLU(),
               n_classes=None,
               n_epochs=20,
               batch_size=128,
               lr=1e-3,
               gamma=0.8,
               device='cpu',
               random_state=33):
    self.hidden_layer_sizes = hidden_layer_sizes
    self.activation = activation
    self.n_classes = n_classes
    self.n_epochs = n_epochs
    self.batch_size = batch_size
    self.lr = lr
    self.gamma = gamma
    self.device = device
    self.random_state = random_state
    self.model = None

  def fit(self, X, y, sample_weight=None):

    if sample_weight is None:
      sample_weight = np.ones(len(y))

    if self.n_classes is None:
      self.n_classes = len(np.unique(y))

    if self.model is None:
      torch.manual_seed(self.random_state)
      layers = []
      hidden_layer_sizes = [X.shape[1]] + list(self.hidden_layer_sizes)
      for i in range(1, len(hidden_layer_sizes)):
        layers.append(
            torch.nn.Linear(hidden_layer_sizes[i - 1], hidden_layer_sizes[i]))
        layers.append(self.activation)
      layers.append(torch.nn.Linear(hidden_layer_sizes[-1], self.n_classes))
      self.model = torch.nn.Sequential(*layers).to(self.device)
    else:
      raise ValueError("Refitting is not supported")

    dataloader_train = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(X, dtype=torch.float32).to(self.device),
            torch.tensor(y, dtype=torch.long).to(self.device),
            torch.tensor(sample_weight, dtype=torch.float32).to(self.device),
        ),
        batch_size=self.batch_size,
        shuffle=True,
        drop_last=True,
    )

    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=1,
                                                gamma=self.gamma)

    self.model.train()
    for epoch in range(self.n_epochs):
      for x, y, w in dataloader_train:
        optimizer.zero_grad()
        outputs = self.model(x)
        losses = loss_fn(outputs, y)
        loss = (losses * w).mean()
        loss.backward()
        optimizer.step()
      scheduler.step()

    return self

  def predict_proba(self, X):
    self.model.eval()
    probas = []
    with torch.no_grad():
      for x in torch.utils.data.DataLoader(
          torch.tensor(X, dtype=torch.float32).to(self.device),
          batch_size=self.batch_size,
          shuffle=False,
      ):
        probas.append(torch.softmax(self.model(x), dim=1).cpu().numpy())
    return np.concatenate(probas, axis=0)

  def predict(self, X):
    return self.predict_proba(X).argmax(axis=1)
