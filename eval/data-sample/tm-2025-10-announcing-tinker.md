# Announcing Tinker

**Date:** October 1, 2025  
**Source:** [Thinking Machines Lab](https://thinkingmachines.ai/blog/announcing-tinker/)

## Summary

Thinking Machines Lab introduces Tinker, a flexible API designed to simplify the fine-tuning of language models. Tinker allows researchers and developers to focus on their data and algorithms while managing the complexities of distributed training.

## Key Features

### Model Support
- Supports fine-tuning various open-weight models
- Includes large mixture-of-experts models like Qwen-235B-A22B
- Switching between models requires only a single string change in code

### Managed Service
- Operates on internal clusters and training infrastructure
- Handles scheduling, resource allocation, and failure recovery
- Users can initiate training runs without infrastructure management burden

### LoRA Implementation
- Uses Low-Rank Adaptation (LoRA) for fine-tuning
- Trains small adapter instead of modifying all original model weights
- Multiple training runs can share same compute resources
- Reduces overall costs

### Tinker Cookbook
- Open-source library accompanying Tinker
- Provides modern implementations of post-training methods
- Facilitates experimentation and customization

## Early Adopters

Several academic and research groups have utilized Tinker:

- **Princeton Goedel Team**: Trained mathematical theorem provers
- **Stanford Rotskoff Chemistry Group**: Fine-tuned models for chemistry reasoning tasks
- **Berkeley SkyRL Group**: Experiments on custom asynchronous off-policy RL training loops with multi-agents and multi-turn tool use
- **Redwood Research**: Applied reinforcement learning to Qwen3-32B on complex AI control tasks

## Pricing

Initially launched in private beta with a waitlist. Free to start, with usage-based pricing introduced subsequently.
