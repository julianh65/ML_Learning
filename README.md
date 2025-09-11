# ML Paper Reimplementations: A Journey Through Foundational Concepts

This repository documents my self-study project focused on building an understanding of fundamental Machine Learning (ML), Deep Learning (DL), and Reinforcement Learning (RL) concepts by reading and reimplementing seminal research papers.

## Current Progress & Syllabus

Status Legend: ✅ Done, ⏳ In Progress

---

This project is a work in progress. Feedback, suggestions, and discussions are welcome!

For the sake of time some papers will just be a read and not a reimplement. I've noted which ones in the implementation goal.

### Phase 1 - Core Learning Mechanics

| #   | Paper (Year)                          | Implementation Goal                                             | Status |
| --- | ------------------------------------- | --------------------------------------------------------------- | ------ |
|  1  | **Back‑propagation** – Rumelhart 1986 | Build a 2‑layer NumPy MLP (sigmoid/tanh) trained with plain SGD | ✅     |

### Phase 2 - LeNet + Training Fixes and Optimizations

| #   | Paper (Year)                  | Implementation Goal                                         | Status |
| --- | ----------------------------- | ----------------------------------------------------------- | ------ |
|  2  | **LeNet‑5** – LeCun 1998      | First CNN on MNIST using sigmoid/tanh; feel slow training   | ✅     |
|  3  | **Weight Decay** – Krogh 1991 | Add L2 regularisation switch and observe over‑fit reduction | ✅     |
|  4  | **ReLU** – Glorot 2011        | Swap activation to ReLU and compare learning speed          | ✅     |
|  5  | **Adam** – Kingma 2014        | Plug in Adam optimiser; contrast vs SGD curves              | ✅     |
|  6  | **Dropout** – Srivastava 2014 | Add dropout layer; track test‑set accuracy boost            | ✅     |

**RL-Focused Ramp**

### Phase 3 - RL Foundations (Policy & Value)

| #   | Paper (Year)                   | Implementation Goal                                                         | Status |
| --- | ------------------------------ | --------------------------------------------------------------------------- | ------ |
|  7  | **DQN** – Mnih 2015            | Implement on CartPole → optional Pong; replay buffer + target network       | ✅     |
|  8  | **A2C (from A3C)** – Mnih 2016 | Read and micro‑implementation by simplifying PPO (no clipping, sync update) | ✅     |
| 9   | **GAE** – Schulman 2015        | Read and make notes; integrate the estimator inside PPO                     | ⏳     |
| 10  | **TRPO** – Schulman 2015       | Read only; understand KL constraint and motivation for PPO                  |        |
| 11  | **PPO** – Schulman 2017        | Full PyTorch implementation with GAE; CartPole → LunarLander                |        |

### Phase 4 - Off‑Policy Continuous Control

| #   | Paper (Year)                          | Implementation Goal                                          | Status |
| --- | ------------------------------------- | ------------------------------------------------------------ | ------ |
| 12  | **DDPG** – Lillicrap 2016             | Read and make notes only                                     |        |
| 13  | **TD3** – Fujimoto 2018               | Read and make notes only; note twin critics and target noise |        |
| 14  | **Soft Actor‑Critic** – Haarnoja 2018 | Implement SAC on LunarLanderContinuous (or Pendulum)         |        |

### Phase 5 - Scaling RL Systems & Tooling

| #   | Paper / Item                                    | Implementation Goal                                                  | Status |
| --- | ----------------------------------------------- | -------------------------------------------------------------------- | ------ |
| 15  | **Prioritized Experience Replay** – Schaul 2015 | Read only; optional DQN add‑on later                                 |        |
| 16  | **IMPALA** – Espeholt 2018                      | Read only; diagram actor–learner split and V‑trace corrections       |        |
| 17  | **Ape‑X** – Horgan 2018                         | Read only; replay server pattern; contrast with IMPALA               |        |
| 18  | **R2D2** – Kapturowski 2019                     | Read only; recurrent replay, burn‑in sequences                       |        |
| 19  | **SEED RL** – Espeholt 2019                     | Read only; latency and TPU considerations                            |        |
| 20  | **RLlib & Flow (frameworks)**                   | Run RLlib PPO/SAC tutorials; inspect configs. Flow is a stretch goal |        |
| 21  | **Mini distributed PPO/SAC (no paper)**         | Later task: rollout workers + learner via Ray/torch.multiprocessing  |        |

### Phase 6 - Multi‑Agent RL

| #   | Paper (Year)           | Implementation Goal                                                   | Status |
| --- | ---------------------- | --------------------------------------------------------------------- | ------ |
| 23  | **MADDPG** – Lowe 2017 | Read only; centralized training, decentralized execution              |        |
| 24  | **QMIX** – Rashid 2018 | Read only; value factorisation for cooperative tasks                  |        |
| 25  | **MAPPO** – Yu 2021    | Implement later; PPO variant for multi‑agent (PettingZoo/highway-env) |        |

### Phase 7 - Deep CNNs & Stability Tricks

| #   | Paper (Year)                  | Implementation Goal                          | Status |
| --- | ----------------------------- | -------------------------------------------- | ------ |
| 26  | **AlexNet** – Krizhevsky 2012 | Port to PyTorch; train on CIFAR‑10           |        |
| 27  | **BatchNorm** – Ioffe 2015    | Insert BN into AlexNet; compare convergence  |        |
| 28  | **ResNet‑18** – He 2015       | Implement skip connections; test on CIFAR‑10 |        |

### Phase 8 - Sequences, Attention, Transformers

| #   | Paper (Year)                           | Implementation Goal                                 | Status |
| --- | -------------------------------------- | --------------------------------------------------- | ------ |
| 29  | **Vanilla RNN** – Williams 1989        | Implement tiny RNN; observe gradient issues         |        |
| 30  | **LSTM** – Hochreiter 1997             | Code LSTM cell; solve long‑range toy task           |        |
| 31  | **Word2Vec** – Mikolov 2013            | Train Skip‑gram / CBOW embeddings                   |        |
| 32  | **Seq2Seq** – Sutskever 2014           | Build encoder–decoder LSTM for toy translation      |        |
| 33  | **Additive Attention** – Bahdanau 2014 | Add attention layer; visualise alignment heatmap    |        |
| 34  | **Transformer** – Vaswani 2017         | Implement mini‑Transformer; self‑attention demo     |        |
| 35  | **BERT (mini)** – Devlin 2018          | Pre‑train tiny masked‑LM, fine‑tune sentiment       |        |
| 36  | **ViT** – Dosovitskiy 2020             | Patchify CIFAR‑10; train mini Vision Transformer    |        |
| 37  | **Mixture‑of‑Experts** – Shazeer 2017  | Replace MLP block with sparse MoE; inspect capacity |        |
| 38  | **Mamba** – Gu 2024                    | Read & prototype simple SSM layer                   |        |
| 39  | **FlashAttention** – Dao 2022          | _Read and make notes only_                          |        |
| 40  | **LoRA** – Hu 2021                     | Implement low‑rank adapters; quick BERT fine‑tune   |        |

### Phase 9 - Generative Models

| #   | Paper (Year)                      | Implementation Goal                                 | Status |
| --- | --------------------------------- | --------------------------------------------------- | ------ |
| 41  | **VAE** – Kingma 2013             | MNIST VAE; latent interpolation demo                |        |
| 42  | **GAN / DCGAN** – Goodfellow 2014 | Train DCGAN on MNIST/FashionMNIST; note instability |        |
| 43  | **Diffusion (DDPM)** – Ho 2020    | Implement toy 32×32 diffusion model                 |        |
| 44  | **WaveNet** – van den Oord 2016   | Skim / prototype 1‑D causal convs for audio         |        |

### Phase 10 - Robotics / Sim2Real

| #   | Paper (Year)                          | Implementation Goal        | Status |
| --- | ------------------------------------- | -------------------------- | ------ |
| 45  | **Hand‑Eye Grasping** – Levine 2016   | _Read and make notes only_ |        |
| 46  | **Domain Randomization** – Tobin 2017 | _Read and make notes only_ |        |
