# GP Sampling

A minimal implementation of decoupled samplers from "Efficiently Sampling Functions from
 Gaussian Process  Posteriors" (https://arxiv.org/abs/2002.09309) on BoTorch / GPyTorch.

Original implementation by the author (based on GPflow) is found at 
https://github.com/j-wilson/GPflowSampling .
 
Supports decoupled sampling from BoTorch `SingleTaskGP` models. It is set to work with
 the default Matern kernel (`ScaleKernel(MaterKernel(...))`), and also supports the
  Squared Exponential kernel (`ScaledKernel(RBFKernel(...))`).
  
 See `notebooks/` for examples.
 