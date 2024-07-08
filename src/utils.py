def print_sep():
    print("-" * 60)

def load_hparams(hparams_path):
    hparams_values = {}
    for i, line in enumerate(open(hparams_path)):
        if i == 0:
            continue
        key, value = line.strip().split(':')
        hparams_values[key.strip()] = value.strip()
    return hparams_values

def min_max_norm(x):
    if len(x.shape) == 1:
        return (x - min(x)) / (max(x) - min(x))
    else:
        x_min = x.min(-1).values.unsqueeze(-1)
        x_max = x.max(-1).values.unsqueeze(-1)
        return (x - x_min) / (x_max - x_min)
