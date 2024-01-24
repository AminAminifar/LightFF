# Lightweight Inference for Forward-Forward Training Algorithm
Code to run the simulations of the paper: 
**Lightweight Inference for Forward-Forward Training Algorithm**

![lightinferenceFF](./img/main.png)

We apply our proposed lightweight inference in the context of three state-of-the-art techniques, namely, the Forward-Forward Algorithm<sup>[2]</sup> (Multi-Pass [MP] and One-Pass [OP]) and PEPITA<sup>[2]</sup> [PT].

Lightweight-FF is based on [loewex's FF implementation](https://github.com/loeweX/Forward-Forward). Lightweight-PT is based on [GiorgiaD's PT implementation](https://github.com/GiorgiaD/PEPITA). 

> In the ``./test`` folder, we also provide the lightweight inference code based on [mpezeshki's FF implementation](https://github.com/mpezeshki/pytorch_forward_forward), as a test version.

[1] Hinton, Geoffrey. "The forward-forward algorithm: Some preliminary investigations." arXiv preprint arXiv:2212.13345 (2022).

[2] Dellaferrera, Giorgia, and Gabriel Kreiman. "Error-driven input modulation: solving the credit assignment problem without a backward pass." International Conference on Machine Learning. PMLR, 2022.

