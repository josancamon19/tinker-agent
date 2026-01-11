# MolmoAct: An Action Reasoning Model that reasons in 3D space

**Date:** August 12, 2025  
**Source:** [Allen Institute for AI](https://allenai.org/blog/molmoact)

## Summary

MolmoAct is an Action Reasoning Model (ARM) that enables machines to reason about actions in three-dimensional space. It builds upon Ai2's Molmo family of vision-language models, bridging the gap between language and physical action.

## Three-Stage Autoregressive Process

### 1. Understanding the Physical World
The model generates spatially grounded perception tokens that encode geometric structures, allowing it to estimate distances and relationships between objects.

### 2. Planning in Image Space
MolmoAct predicts a sequence of image-space waypoints, visually outlining the task's progression while remaining adaptable to various robotic embodiments.

### 3. Action Decoding
The model translates the waypoints into detailed, low-level action commands tailored to specific robotic hardware configurations.

## Key Advantages

- Adapts to different robotic forms (humanoids, gripper arms) with minimal fine-tuning
- Superior performance compared to baselines like Physical Intelligence's π0 and OpenVLA
- Intuitive control through natural language or visual traces on devices

## Training Details

### MolmoAct-7B
- Pre-trained on 26.3 million samples
- Training cluster: 256 NVIDIA H100 GPUs
- Pre-training time: ~1 day
- Fine-tuning time: ~2 hours on 64 H100 GPUs

### Data
- Pre-trained on curated subset of Open-X Embodiment data and multimodal reasoning dataset
- Post-trained on MolmoAct dataset (~10,000 distinct robot episodes)
- Videos of robots performing various household tasks

## Benchmark Results

- State-of-the-art out-of-distribution task success rate of 72.1% on SimplerEnv benchmark
- Outperformed several leading models on key robotics benchmarks

## Open Release

All components released openly:
- Model weights
- Training data
- Evaluation tools
