# ML Paper Reimplementations: A Journey Through Foundational Concepts

This repository documents my self-study project focused on building an understanding of fundamental Machine Learning (ML), Deep Learning (DL), and Reinforcement Learning (RL) concepts by reading and reimplementing seminal research papers.

## Current Progress & Syllabus

Status Legend: ✅ Done, ⏳ In Progress

---

This project is a work in progress. Feedback, suggestions, and discussions are welcome!

### Phase 1 - Core Learning Mechanics

| #   | Paper (Year)                          | Implementation Goal                                             | Status |
| --- | ------------------------------------- | --------------------------------------------------------------- | ------ |
|  1  | **Back‑propagation** – Rumelhart 1986 | Build a 2‑layer NumPy MLP (sigmoid/tanh) trained with plain SGD | ✅     |

### Phase 2 - LeNet + Training Fixes and Optimizations

| #   | Paper (Year)                  | Implementation Goal                                         | Status |
| --- | ----------------------------- | ----------------------------------------------------------- | ------ |
|  2  | **LeNet‑5** – LeCun 1998      | First CNN on MNIST using sigmoid/tanh; feel slow training   |        |
|  3  | **Weight Decay** – Krogh 1991 | Add L2 regularisation switch and observe over‑fit reduction |        |
|  4  | **ReLU** – Glorot 2011        | Swap activation to ReLU and compare learning speed          |        |
|  5  | **Adam** – Kingma 2014        | Plug in Adam optimiser; contrast vs SGD curves              |        |
|  6  | **Dropout** – Srivastava 2014 | Add dropout layer; track test‑set accuracy boost            |        |

### Phase 3 - Deep CNNs & Stability Tricks

| #   | Paper (Year)                  | Implementation Goal                          | Status |
| --- | ----------------------------- | -------------------------------------------- | ------ |
|  7  | **AlexNet** – Krizhevsky 2012 | Port to PyTorch; train on CIFAR‑10           |        |
|  8  | **BatchNorm** – Ioffe 2015    | Insert BN into AlexNet; compare convergence  |        |
| 9   | **ResNet‑18** – He 2015       | Implement skip connections; test on CIFAR‑10 |        |

### Phase 4 - Sequences, Attention, Transformers

| #   | Paper (Year)                           | Implementation Goal                                 | Status |
| --- | -------------------------------------- | --------------------------------------------------- | ------ |
| 10  | **Vanilla RNN** – Williams 1989        | Implement tiny RNN; observe gradient issues         |        |
|  11 | **LSTM** – Hochreiter 1997             | Code LSTM cell; solve long‑range toy task           |        |
|  12 | **Word2Vec** – Mikolov 2013            | Train Skip‑gram / CBOW embeddings                   |        |
|  13 | **Seq2Seq** – Sutskever 2014           | Build encoder–decoder LSTM for toy translation      |        |
|  14 | **Additive Attention** – Bahdanau 2014 | Add attention layer; visualise alignment heatmap    |        |
|  15 | **Transformer** – Vaswani 2017         | Implement mini‑Transformer; self‑attention demo     |        |
|  16 | **BERT (mini)** – Devlin 2018          | Pre‑train tiny masked‑LM, fine‑tune sentiment       |        |
|  17 | **ViT** – Dosovitskiy 2020             | Patchify CIFAR‑10; train mini Vision Transformer    |        |
|  18 | **Mixture‑of‑Experts** – Shazeer 2017  | Replace MLP block with sparse MoE; inspect capacity |        |
|  19 | **Mamba** – Gu 2024                    | Read & prototype simple SSM layer                   |        |
|  20 | **FlashAttention** – Dao 2022          | Integrate flash‑attn kernel or read for context     |        |
|  21 | **LoRA** – Hu 2021                     | Implement low‑rank adapters; quick BERT fine‑tune   |        |

### Phase 5 - Generative Models

| #   | Paper (Year)                      | Implementation Goal                                 | Status |
| --- | --------------------------------- | --------------------------------------------------- | ------ |
|  22 | **VAE** – Kingma 2013             | MNIST VAE; latent interpolation demo                |        |
|  23 | **GAN / DCGAN** – Goodfellow 2014 | Train DCGAN on MNIST/FashionMNIST; note instability |        |
|  24 | **Diffusion (DDPM)** – Ho 2020    | Implement toy 32×32 diffusion model                 |        |
|  25 | **WaveNet** – van den Oord 2016   | Skim / prototype 1‑D causal convs for audio         |        |

### Phase 6 - Deep RL & Robotics

| #   | Paper (Year)                           | Implementation Goal                         | Status |
| --- | -------------------------------------- | ------------------------------------------- | ------ |
|  26 | **DQN** – Mnih 2015                    | CartPole then Pong with replay & target net |        |
|  27 | **A3C / A2C** – Mnih 2016 ＊           | Actor‑critic baseline on CartPole           |        |
|  28 | **PPO** – Schulman 2017                | Implement PPO; compare stability vs A2C     |        |
|  29 | **DDPG** – Lillicrap 2016              | Continuous control (Pendulum)               |        |
|  30 | **Soft Actor‑Critic** – Haarnoja 2018  | SAC on LunarLanderContinuous                |        |
|  31 | **Hand‑Eye Grasping** – Levine 2016    | Reproduce small sim grasp task              |        |
|  32 | **Domain Randomization** – Tobin 2017  | Apply simple sim‑to‑real randomisations     |        |
