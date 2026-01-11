# Modular Manifolds

**Date:** September 26, 2025  
**Author:** Jeremy Bernstein  
**Source:** [Thinking Machines Lab](https://thinkingmachines.ai/blog/modular-manifolds/)

## Summary

An exploration of constraining neural network weight matrices to submanifolds to improve training stability and efficiency in large networks.

## Core Concepts

### Healthy Tensor Sizes
Maintaining healthy tensor sizes in large neural networks is critical. Normalization of weight matrices prevents issues like:
- Numerical underflow
- Numerical overflow

### Manifold Constraints
The approach constrains weight matrices to specific submanifolds at each layer, facilitating the co-design of optimization algorithms with these constraints.

## Stiefel Manifold Example

The article presents a manifold version of the Muon optimizer where:
- Weights are constrained to the Stiefel manifold
- Stiefel manifold = set of matrices with unit condition numbers
- Results in more stable training dynamics

## Modular Manifolds

Introduces "modular manifolds" - composable manifolds aimed at simplifying the scaling and training of large networks.

## Future Research Directions

The post encourages further research in:
- Modularity
- Numerical considerations
- Convex optimization
- Convergence analysis
- Regularization
- Architecture-optimizer co-design
- Non-Riemannian geometry
- Practical implementation
