# SciArena: A new platform for evaluating foundation models in scientific literature tasks

**Date:** July 2025  
**Source:** [Allen Institute for AI](https://allenai.org/blog/sciarena)

## Summary

SciArena is an open and collaborative platform designed to evaluate foundation models on scientific literature tasks. It addresses challenges posed by the rapid expansion of scientific literature and the limitations of traditional benchmarks.

## Platform Components

### 1. SciArena Platform
- Researchers submit questions
- View side-by-side responses from different foundation models
- Vote for the preferred output

### 2. Leaderboard
- Uses Elo rating system
- Ranks models based on community votes
- Provides dynamic, up-to-date assessment

### 3. SciArena-Eval
- Meta-evaluation benchmark
- Built on collected human preference data
- Assesses accuracy of model-based evaluation systems

## Model Performance (as of June 30, 2025)

The platform hosted 23 frontier foundation models:

- **o3**: Consistently top performance across all scientific domains; detailed elaborations of cited papers; more technical outputs in engineering
- **Claude-4-Opus**: Excelled in healthcare domains
- **DeepSeek-R1-0528**: Excelled in natural science domains

## Key Finding

Even the top-performing model (o3) achieved only 65.1% accuracy in predicting human preferences, highlighting the need for more robust automated evaluation methods in scientific reasoning tasks.

## Open Source

Ai2 has open-sourced:
- SciArena-Eval
- Code and data used to develop SciArena
- Encouraging continuous evaluation of foundation model advancements
