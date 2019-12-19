# stacked-autoencoder-pytorch
Stacked convolutional autoencoder written in Pytorch for some experiments.

[PPT for methods & model architecture](https://docs.google.com/presentation/d/1c8oRVfZMD2TRwqL7cGnUZ8MXTRaEiZk3MdKG9m4lxvE/edit?usp=sharing)

Setup:
- Python 3
- Pytorch 0.3.0 with torchvision

Run `python run.py` to start training.

Args:
- '--gpu': specify the number of gpu(s)
- '--save-dir': directory to save the results

example: `python run.py --save-dir 'stacked_CAE_Dec_19_2' --gpu '0,1,2,3,4,5'`

Observations:
  - Although the loss doesn't propagate through layers the overall quality of the reconstruction improves anyways.
  - When using ReLU, the sparsity of the feature vector at the last layer increases from 50% at the start to 80%+ as you continue training even though sparsity isn't enforced.
  - Augmenting with rotation slows down training and decreases the quality of the feature vector.
