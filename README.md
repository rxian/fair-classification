# Fair Classification via Post-Processing

A post-processing algorithm for fair classification applied to  predictors of the form `Pr(Y|X)` and `Pr(A|X)`, or `Pr(A,Y|X)`, depending on the fairness criterion.  Supports (multi-class) *statistical parity*, *equal opportunity*, and *equalized odds*, under the *attribute-aware* or *attribute-blind* setting.

See `example.ipynb` for a quick tutorial.  To reproduce our results:

- ([arXiv 2024 preprint](https://arxiv.org/abs/2405.04025)).  See the notebooks `adult.ipynb`, `compas.ipynb`, `acsincome2.ipynb`, `acsincome5.ipynb`, and `biasbios.ipynb`.
- ([ICML 2023](https://proceedings.mlr.press/v202/xian23b.html)).  Archived under the `icml.23` tag, since the new version generalizes the algorithm for attribute-aware statistical parity.

**LP solvers.**  Our algorithm involves solving linear programs, and they are set up in our code using the `cvxpy` package.  For large-scale problems, we recommend the Gurobi optimizer for speed.

## Citation

```bibtex
@misc{xian2024OptimalGroupFair,
  title         = {{Optimal Group Fair Classifiers from Linear Post-Processing}},
  author        = {Xian, Ruicheng and Zhao, Han},
  year          = {2024},
  archiveprefix = {arXiv},
  eprint        = {2405.04025},
  primaryclass  = {cs.LG}
}
```

```bibtex
@inproceedings{xian2023FairOptimalClassification,
  title     = {{Fair and Optimal Classification via Post-Processing}},
  booktitle = {{Proceedings of the 40th International Conference on Machine Learning}},
  author    = {Xian, Ruicheng and Yin, Lang and Zhao, Han},
  year      = {2023}
}
```
