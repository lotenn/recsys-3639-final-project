import torch


def get_device():
    # check if CUDA is available
    train_on_gpu = torch.cuda.is_available()

    device = torch.device('cpu')
    if not train_on_gpu:
        print('CUDA is not available.  Training on CPU ...')
    else:
        print('CUDA is available!  Training on GPU ...')
        device = torch.device('cuda:0')

    return device
