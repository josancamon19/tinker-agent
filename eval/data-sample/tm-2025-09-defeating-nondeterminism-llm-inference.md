# Defeating Nondeterminism in LLM Inference

**Date:** September 10, 2025  
**Authors:** Horace He et al.  
**Source:** [Thinking Machines Lab](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)

## Summary

Addresses the challenges of achieving reproducible results in large language model inference, proposing solutions for consistent outputs even when temperature is set to 0.

## The Problem

Even with temperature set to 0 (intended to make sampling deterministic), LLM APIs often produce varying outputs. This creates challenges for:
- Research reproducibility
- Testing and debugging
- Production reliability

## Common Misconception

### "Concurrency + Floating Point" Hypothesis
A common explanation attributes nondeterminism to:
- Non-associative nature of floating-point arithmetic
- Combined with concurrent execution

### Reality
- Some GPU kernels are nondeterministic
- The forward pass of an LLM inference server CAN be deterministic
- From user perspective, results appear nondeterministic due to factors like varying server load affecting batch sizes

## Proposed Solution: Batch Invariance

The authors propose achieving batch invariance in kernels through:

### Strategies
1. Using consistent reduction strategies across different batch sizes
2. Implementing fixed split-size strategies in attention mechanisms

### Results
By adopting these approaches, it's possible to obtain truly reproducible results in LLM inference.

## Experiments

The post includes:
- Experiments showcasing extent of nondeterminism in LLM completions
- Performance evaluation of batch-invariant kernels

## Conclusion

Understanding and addressing the root causes of nondeterminism is essential for achieving reliable and reproducible outcomes in machine learning systems.
