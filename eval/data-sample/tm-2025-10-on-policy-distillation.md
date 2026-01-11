# On-Policy Distillation

**Date:** October 27, 2025  
**Author:** Kevin Lu et al.  
**Source:** [Thinking Machines Lab](https://thinkingmachines.ai/blog/on-policy-distillation/)

## Summary

A novel training supervision method for language models that combines on-policy training with dense supervision to enhance model training efficiency.

## Background: Traditional Training Stages

1. **Pre-training**: Develops general language understanding and world knowledge
2. **Mid-training**: Imparts domain-specific knowledge
3. **Post-training**: Elicits targeted behaviors like instruction following or reasoning

## Limitations of Existing Methods

### Supervised Fine-Tuning (SFT)
- Off-policy training with dense reward signals
- May not generalize well

### Reinforcement Learning (RL)
- On-policy training with sparse rewards
- Leads to inefficiencies

## On-Policy Distillation Approach

Combines on-policy sampling with dense supervision:

1. **Student model** generates outputs
2. **High-performing teacher model** evaluates each token
3. Teacher provides detailed feedback on every token

### Benefits
- Reliability of on-policy training
- Efficiency of dense rewards
- Cost-effective post-training technique
- High performance with significantly less computational resources

## Applications

- Training models for mathematical reasoning
- Developing assistant models that integrate domain knowledge with instruction-following capabilities

## Results

Experiments demonstrate that on-policy distillation achieves high performance with significantly less computational resources compared to traditional RL methods.
