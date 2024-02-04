# Supplementary code for Sparse Concept Bottleneck Models: Gumbel tricks in Contrastive Learning

## How to install
```bash
git clone https://github.com/icml24/SparseCBM
cd SparseCBM
pip install -r requirements.txt
```
## Repository structure
* <ins>additional_evaluations</ins> contains training examples of CBMs, evaluation of CMS, concepts generation and latent space visualization.
* In <ins>bottleneck</ins> you may find a code for: our model, key objective functions, training utilities and a setup for CMS.
* Use <ins>data</ins> to look through the concept sets and class labels we use.
* Please, run demo.ipynb in <ins>demo_notebooks</ins> to train your own CBM with any hyperparameters you need and any of the datasets provided. We suggest you to use CUB200 examples, for simplicity, and train with a small learning rates both of B/32 and L/14 configurations (CBMs are sensitive to lrs). A simple 10 epochs example with CLIP-ViT-B/32 backbone is already presented in demo, but feel free to adjust more changes.

<ins>Remark:</ins> if you get a ```list index out of range``` error during training just run the training cell again, it means that accuracy is not rounded with 2 decimals. Training process won't stop in this case.

**We believe the details provided are clear enough to reproduce the main findings of our paper.**

