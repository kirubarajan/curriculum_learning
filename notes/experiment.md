# Experiment
> Disclaimer: these notes are heavily a work in progress - they might be incohesive, incorrect, or otherwise useless.

## Optimization Heuristics
- Learning rate scheduling

## Curriculum Definition
These notes are limited to curriculums consisting of natural language tasks:

- Word Frequency
- Sentence length
- Sentence complexity
- Attention scores

Competence can be used as a function of current loss, and current curriculum state (e.g. number of iterations etc). Considerations to be made are regarding exploration/exploitation tradeoff between re-learning important samples and experimenting to discover other potentially impactful data.

## Implementation
All models are implemented using PyTorch:

- Sentiment analysis (with Recurrent Neural Networks)
- Language modelling (with Transformer)

## Transfer Learning
- Learning a curriculum from another language  model's latent information about input text "difficulty"
