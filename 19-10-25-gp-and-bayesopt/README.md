# Gaussian Processes and Bayesian Optimization

## Retroactive agenda for the day

* The problem of hyperparameter tuning
* Grid search
* Surprising efficiency of random search
* High-dimensionality effects
* Gaussian distributions
* Multivariate Gaussian distributions
* Covariance / correlation matrix
* Covariance matrix defined by a kernel
* Parametric models
* Gaussian processes are non-parametric models
* Lightly touched on Dirichlet Process
* Intuitive picture of how to condition gaussian processes
* Bayesian optimization
* Acquisition functions
  * Probability of Improvement
  * Expected Improvement
  * Entropy Search
* scikit-optimize
* Connections to Multi-armed bandits and Reinforcement Learning
* Freeze-Thaw bayesian optimization
* BO as a service (SigOpt)

## Some links to check out

* [Video](https://www.youtube.com/watch?v=92-98SYOdlY) with a really good intro to gaussian processes. **HIGHLY** recommend.
* [Scikit-Optimize](https://scikit-optimize.github.io/) and 
  [this video](https://www.youtube.com/watch?v=DGJTEBt0d-s) explaining the motivation behind the library.
* [HyperOpt](https://github.com/hyperopt/hyperopt) and a [video](https://www.youtube.com/watch?v=tdwgR1AqQ8Y) explaining how it works.
* [MOE](https://github.com/Yelp/MOE)
* [Google Vizier](https://ai.google/research/pubs/pub46180)
  * And its [open-source implementation](https://advisor.readthedocs.io/en/latest/) (?).
* [SigOpt](https://sigopt.com/) and a [video](https://www.youtube.com/watch?v=J6UcAdH54RE) explaining what they do.
* [BoTorch](https://www.botorch.org/)

```bash
pip3 install scikit-optimize
pip3 install gpy
pip3 install gpyopt
pip3 install botorch
```
