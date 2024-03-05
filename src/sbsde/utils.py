def flatten_dim01(x):
    return x.reshape(-1, *x.shape[2:])
