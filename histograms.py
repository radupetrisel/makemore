from models import *

def activation_dist(layers: list[Linear | Tanh | BatchNorm1d]):
    plt.figure(figsize=(20, 4))
    legends = []
    for i, layer in enumerate(layers):
        if not isinstance(layer, Tanh):
            continue

        out = layer.out
        print(f'layer {i} ({layer.__class__.__name__}): mean: {out.mean():.2f}, std: {out.std():.2f}, saturated: {(out.abs() > 0.97).float().mean() * 100:.2f}%')

        hy, hx = torch.histogram(out, density=True)
        plt.plot(hx[:-1].detach(), hy.detach())
        legends.append(f'layer {i} ({layer.__class__.__name__})')

    plt.legend(legends); plt.title("activation distribution")

def gradient_distribution(layers: list[Linear | Tanh | BatchNorm1d]):
    plt.figure(figsize=(20, 4))
    legends = []

    for i, layer in enumerate(layers):
        if not isinstance(layer, Tanh):
            continue

        out = layer.out.grad
        print(f'Layer {i} ({layer.__class__.__name__}): mean {out.mean():+f} std: {out.std():e}')
        hy, hx = torch.histogram(out, density=True)
        plt.plot(hx[:-1].detach(), hy.detach())
        legends.append(f'Layer {i} ({layer.__class__.__name__})')

    plt.legend(legends); plt.title('gradient distribution')

def weight_gradient_dist(parameters: list[torch.Tensor]):
    plt.figure(figsize=(20, 4))
    legends = []
    for i, p in enumerate(parameters):
        if p.ndim != 2:
            continue

        out = p.grad
        print(f'weight {tuple(p.shape)} | mean: {out.mean():+f} | std: {out.std():e} | grad:data ratio {out.std() / p.std():e}')

        hy, hx = torch.histogram(out, density=True)
        plt.plot(hx[:-1].detach(), hy.detach())
        legends.append(f'{i} weight {tuple(p.shape)}')

    plt.legend(legends); plt.title('weight gradient distribution')

def grad_data_ratio(parameters: list[torch.Tensor], ratios: list[list[float]]):
    plt.figure(figsize=(20, 4))
    legends = []

    for i, p in enumerate(parameters):
        if p.ndim != 2:
            continue

        plt.plot([ratios[j][i] for j in range(len(ratios))])
        legends.append(f'param {i}')

    plt.plot([0, len(ratios)], [-3, -3], 'k')
    plt.legend(legends); plt.title('gradient to data ratio')