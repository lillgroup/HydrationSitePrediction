import torch


def get_optimizer(model, name='adam', **kwargs):
    """Get optimizer by name and parameters from model"""
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(parameters, **kwargs)
    elif name == 'sgd':
        return torch.optim.SGD(parameters, **kwargs)
    elif name == 'adadelta':
        return torch.optim.Adadelta(parameters, **kwargs)
    elif name == 'adagrad':
        return torch.optim.Adagrad(parameters, **kwargs)
    else:
        raise NotImplementedError('Optimizer not supported: %s' % name)


def get_scheduler(optimizer, name='step', **kwargs):
    """Get scheduler by name and parameters from optimizer"""
    name = name.lower()
    if name == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, **kwargs)
    elif name == 'exp':
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, **kwargs)
    elif name == 'multistep':
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, **kwargs)
    elif name == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
    elif name == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
    else:
        raise NotImplementedError('Scheduler not supported: %s' % name)


def rbf(values, min_val=0.0, max_val=40.0, n_kernels=64, gamma=0.5):
    values = torch.clip(values, min_val, max_val)
    mus = torch.linspace(min_val, max_val, n_kernels, device=values.device)
    return torch.exp(-gamma * torch.square(values.reshape(-1, 1) - mus))


def unload(data):
    """Detach tensor (and collections of tensors) and move to CPU"""
    if type(data) == list:
        return [unload(d) for d in data]
    elif type(data) == tuple:
        return tuple(unload(d) for d in data)
    elif type(data) == dict:
        return {k: unload(v) for k, v in data.items()}
    elif type(data) == torch.Tensor:
        return data.detach().cpu()
    else:
        raise NotImplementedError
