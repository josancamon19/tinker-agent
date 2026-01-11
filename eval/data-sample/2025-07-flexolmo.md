# Introducing FlexOlmo: A new paradigm for language model training and data collaboration

**Date:** July 9, 2025  
**Source:** [Allen Institute for AI](https://allenai.org/blog/flexolmo)

## Summary

FlexOlmo is a new paradigm for language model training that enables collaborative AI development through data collaboration. It allows data owners to contribute to language model development without relinquishing control over their data.

## Problems Addressed

Traditional AI development faces several challenges:

- **No flexibility**: Standard training pipelines require one-time, irreversible data inclusion
- **Loss of control**: Data owners cannot control access after publishing
- **Loss of value**: Data owners lose ability to protect their valuable asset
- **Lack of attribution**: No credit given to data contributors

## How FlexOlmo Works

### Core Concept
Each data owner can:
1. Locally branch from a shared public model
2. Add an expert trained on their data locally
3. Contribute this expert module back to the shared model

### Architecture
Uses a mixture-of-experts (MoE) approach where:
- Each expert is trained independently on private datasets
- Experts are later integrated into the MoE
- Data owners can contribute asynchronously without sharing private data
- Supports continual updates with new data
- Provides strong guarantees for data opt-out

## Data Owner Benefits

Contributors can:
- Decide when their data is active in the model
- Deactivate data at any time
- Receive attributions whenever their data is used for inference

## Performance Results

- Augmenting the public model with expert modules leads to significantly better performance
- The shared model retains—or enhances—each expert's specialized capabilities
- Benefits from the diversity of private datasets

## Community Participation

Ai2 is seeking participants to advance this research and build the future of secure, transparent, and truly open AI in the public interest.
