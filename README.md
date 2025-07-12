# Temporal Distance-aware Transition Augmentation for Offline Model-based Reinforcement Learning

## Overview
This is the official implementation of [TempDATA](https://dongsuleetech.github.io/projects/tempdata/).

This repository provides the implementatino of the TempDATA agent and how to build temporal distance-aware representation for RL.

## Installation
```
conda create --name tempdata python=3.8
conda activate tempdata
pip install -r requirements.txt --no-deps
pip install "jax[cuda11_cudnn82]==0.4.3" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Misc.
### Other links
[ICML 2025 OpenReview](https://openreview.net/forum?id=drBVowFvqf&referrer=%5Bthe%20profile%20of%20Dongsu%20Lee%5D(%2Fprofile%3Fid%3D~Dongsu_Lee1))

### Acknowledgements
Our codebase is inspired by the [HILP](https://github.com/seohongpark/HILP) repository.
