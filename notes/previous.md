# Previous Work
The core of Curriculum Learning (as it relates to this research project) can be distilled into these four papers, which are arranged in chronological order and increasingly become more focused on natural language processing (my interest was sparked at NAACL in 2019). 

## Curriculum Learning 
> [Bengio et al.] ICML 2009

## Automated Curriculum Learning for Neural Networks 
> [Graves et al.] ICML 2017

## Curriculum Learning by Transfer Learning: Theory and Experiments with Deep Networks 
> [Weinshall et al.] ICML 2018

## Competence-based Curriculum Learning for Neural Machine Translation
> [Platanios et al.] NAACL 2019

### Overview
- Develops curriculum as a function of current model "competence" and sample difficulty
- Does NOT depend on multiple hyperparameter optimizations
- 70% decrease in training time, using both Recurrent Neural Networks and Transformer models
- Main impacts are seen using Transformer models

### Details
- RNNs trained without learning rate scheduling
- Difficulty metrics include sentence length, word rarity
- Competence functions simple derivations, NOT a function of loss output/gradient

### Future Work
- Exploring different heuristics of measuring difficulty and competence
- Expanding framework outside of Neural Machine Translation