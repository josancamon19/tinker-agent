# Going beyond open data – increasing transparency and trust in language models with OLMoTrace

**Date:** April 2025  
**Source:** [Allen Institute for AI](https://allenai.org/blog/olmotrace)

## Summary

OLMoTrace is a unique feature in the Ai2 Playground that allows users to trace language model outputs back to their extensive, multi-trillion-token training data in real time. It exemplifies Ai2's dedication to fostering an open AI ecosystem.

## Key Features

### Real-Time Tracing
- Click "Show OLMoTrace" button to highlight spans in the model's response
- Highlighted spans appear verbatim in the training data
- Side panel displays documents from training corpus containing these spans
- See the origins of specific outputs

### Interpretation of Results
- Assists in fact-checking
- Helps understand the model's generation process
- Shows exact documents where information was learned
- Aids in verifying accuracy of model responses

### Technical Innovations
- Efficiently identifies and ranks spans
- Scans model output and matches against training data
- Optimized to handle vast scale of training corpus
- Ensures timely and relevant results

## Supported Models

OLMoTrace is available with Ai2's flagship models:
- OLMo 2 32B Instruct
- OLMo 2 13B Instruct
- OLMoE 1B 7B Instruct

## Purpose

By providing this level of transparency, Ai2 aims to empower researchers, developers, and the general public to better understand and trust the outputs of language models.
