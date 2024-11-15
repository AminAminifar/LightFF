# LightFF: Lightweight Inference for Forward-Forward Algorithm

![lightinferenceFF](./img/main.png)

We apply our proposed lightweight inference in the context of three state-of-the-art techniques, namely, the Forward-Forward Algorithm<sup>[1]</sup> (Multi-Pass [MP] and One-Pass [OP]) and PEPITA<sup>[2]</sup> [PT].

Code to run the simulations of the paper: 
**LightFF: Lightweight Inference for Forward-Forward Algorithm**. Taking MNIST as an example, the codes are shown as follows:

- Lightweight-MP-MNIST: ``python Lightweight-FF/Lightweight-MP/main.py``
- Lightweight-OP-MNIST: ``python Lightweight-FF/Lightweight-OP/main.py``
- Lightweight-PEPITA-MNIST: ``python Lightweight-PT/pepita_MNIST.py``


Lightweight-FF is based on [loewex's FF implementation](https://github.com/loeweX/Forward-Forward). Lightweight-PT is based on [GiorgiaD's PT implementation](https://github.com/GiorgiaD/PEPITA). 

> In the ``./test`` folder, we also provide the lightweight inference code based on [mpezeshki's FF implementation](https://github.com/mpezeshki/pytorch_forward_forward), as a test version.

[1] Hinton, Geoffrey. "The forward-forward algorithm: Some preliminary investigations." arXiv preprint arXiv:2212.13345 (2022).

[2] Dellaferrera, Giorgia, and Gabriel Kreiman. "Error-driven input modulation: solving the credit assignment problem without a backward pass." International Conference on Machine Learning. PMLR, 2022.

## Citation

```
@inproceedings{aminifar2024lightff,
  title={LightFF: Lightweight Inference for Forward-Forward Algorithm},
  author={Aminifar, Amin and Huang, Baichuan and Fahliani, Azra Abtahi and Aminifar, Amir},
  booktitle={27th European Conference on Artificial Intelligence, ECAI-2024},
  pages={1728--1735},
  year={2024},
  organization={IOS Press}
}
```

