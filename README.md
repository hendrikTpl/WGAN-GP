# WGAN-GP
pytorch implementation of wgan-gp

# Usage
- `python3 train.py <exp_name>`
- Config file must be provided at `./configs/<exp_name>.py`
- experiment logs saved in `./EXP_LOGS/log_<exp_name>.txt`

# Sample Results
## MNIST (100 epochs, batch_size=512)
 - ![IMNIST](results/mnist_100_512.png)

## References
- paper: https://arxiv.org/pdf/1704.00028.pdf
- https://github.com/caogang/