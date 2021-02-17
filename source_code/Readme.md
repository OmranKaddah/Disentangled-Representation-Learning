### Methods of Learning Disentangled Representation with VAEs

* **Standard VAE Loss** from [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
* **β-VAE<sub>H</sub>** from [β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/pdf?id=Sy2fzU9gl)
* **β-VAE<sub>B</sub>** from [Understanding disentangling in β-VAE](https://arxiv.org/abs/1804.03599)
* **FactorVAE** from [Disentangling by Factorising](https://arxiv.org/abs/1802.05983)
* **β-TCVAE** from [Isolating Sources of Disentanglement in Variational Autoencoders](https://arxiv.org/abs/1802.04942)
* **Joint-VAE** from [Learning Disentangled Joint Continuous and Discrete Representations](https://arxiv.org/abs/1804.00104) (NIPS 2018).

# Instructions
    To load model, should be named according to the pattern LOSsFUNCTION_model_DATASET
    where DATASET is either of 3Dshapes,mnist, or dsprites
    LOSsFUNCTION is either betavae (for all loss variations except joint-vae and vade), vade, jointvae.

    type --help for argument passing