# GP Sampling

Update: This functionality is now available in BoTorch, [see here.](https://github.com/pytorch/botorch/blob/main/botorch/utils/gp_sampling.py) It is recommended to use the BoTorch version since it will be better maintained going forward.

A minimal implementation of decoupled samplers from "Efficiently Sampling Functions from
 Gaussian Process  Posteriors" (https://arxiv.org/abs/2002.09309) on BoTorch / GPyTorch.

Original implementation by the author (based on GPflow) is found at 
https://github.com/j-wilson/GPflowSampling .
 
Supports decoupled sampling from BoTorch `SingleTaskGP` models. It is set to work with
 the default Matern kernel (`ScaleKernel(MaternKernel(...))`), and also supports the
  Squared Exponential kernel (`ScaleKernel(RBFKernel(...))`).
  
 See `notebooks/` for examples.
 
