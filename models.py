import torch
import string
import matplotlib.pyplot as plt
import torch.nn.functional as F

class Bigram:

    def __init__(self):
        self._P = torch.ones(size=(27, 27))
        self._stoi = {s: i+1 for i, s in enumerate(string.ascii_lowercase)}
        self._stoi['.'] = 0
        self._itos = {i: s for s, i in self._stoi.items()}

    def fit(self, X):
        for word in X:
            input = ['.'] + list(word) + ['.']
            for first, second in zip(input, input[1:]):
                ix1 = self._stoi[first]
                ix2 = self._stoi[second]

                self._P[ix1, ix2] += 1

        row_sum = self._P.sum(dim=1, keepdim=True)
        self._P /= row_sum

    def show_probs(self):
        plt.figure(figsize=(16, 16))
        plt.imshow(self._P, cmap='Blues')
        for i in range(27):
            for j in range(27):
                chstr = self._itos[i] + self._itos[j]
                plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
                plt.text(j, i, f"{self._P[i, j].item():.4f}", ha="center", va="top", color='gray')
        plt.axis('off')

    def make(self, count=5):
        results = []

        generator = torch.Generator()
        generator.manual_seed(1234)

        for _ in range(count):
            out = []
            idx = 0

            while True:
                multinomial = torch.multinomial(self._P[idx], num_samples=1, replacement=True, generator=generator)
                idx = multinomial.item()

                out.append(self._itos[idx])

                if idx == 0:
                    break

            results.append(''.join(out))

        return results

    def loss(self, X):
        log_likelihood = 0.0
        n = 0

        for word in X:
            chars = ['.'] + list(word) + ['.']
            for first, second in zip(chars, chars[1:]):
                ix1 = self._stoi[first]
                ix2 = self._stoi[second]

                p = self._P[ix1, ix2]
                log_likelihood += -torch.log(p)
                n += 1

        return (log_likelihood / n).item()

class NN:

    def __init__(self, generator):
        self._W = torch.randn((27, 27), requires_grad=True, generator=generator)
        self._stoi = {s: i + 1 for i, s in enumerate(string.ascii_lowercase)}
        self._stoi['.'] = 0
        self._itos = {i: s for s, i in self._stoi.items()}

    def fit(self, data, num_epocs=200, learning_rate=50):
        Xs, ys = [], []
        # prepare the dataset
        for word in data:
            chars = ['.'] + list(word) + ['.']
            for x, y in zip(chars, chars[1:]):
                Xs.append(self._stoi[x])
                ys.append(self._stoi[y])

        Xs = torch.tensor(Xs)
        ys = torch.tensor(ys)
        xenc = F.one_hot(Xs, num_classes=27).float()
        num = len(Xs)

        loss = 0.0
        for epoch in range(num_epocs):
            logits = xenc @ self._W
            counts = logits.exp()
            probs = counts / counts.sum(dim=1, keepdim=True)

            loss = -probs[torch.arange(num), ys].log().mean()

            self._W.grad = None
            loss.backward()

            self._W.data += -learning_rate * self._W.grad

        print(f'Loss={loss.item():.4f}')

    def make(self, count=5, generator=torch.Generator()):
        results = []

        for _ in range(count):
            idx = 0
            out = []
            while True:
                xenc = F.one_hot(torch.tensor([idx]), num_classes=27).float()
                logits = xenc @ self._W
                counts = logits.exp()
                probs = counts / counts.sum(dim=1, keepdim=True)

                idx = torch.multinomial(probs, num_samples=1, replacement=True, generator=generator).item()

                out.append(self._itos[idx])
                if idx == 0:
                    break

            results.append(''.join(out))

        return results

class Linear:
    def __init__(self, fan_in, fan_out, generator, gain = 1.0, bias = True):
        self.weight = torch.randn((fan_in, fan_out), generator=generator) * gain / (fan_in ** 0.5)
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight

        if self.bias is not None:
            self.out += self.bias

        return self.out

    def parameters(self):
        return [self.weight] + ([self.bias] if self.bias is not None else [])

class BatchNorm1d:
    def __init__(self, dim, eps = 1e-5, momentum = 0.1):
        self.training = True
        self.eps = eps
        self.momentum = momentum

        self.scale = torch.ones(dim)
        self.shift = torch.zeros(dim)

        self.mean = torch.zeros(dim)
        self.var = torch.ones(dim)

    def __call__(self, x):
        if self.training:
            if x.ndim == 2:
                dim = 0
            elif x.ndim == 3:
                dim = (0, 1)

            mean = x.mean(dim, keepdim=True)
            var = x.var(dim, keepdim=True)
            self.out = self.scale * (x - mean) / torch.sqrt(var + self.eps) + self.shift

            with torch.no_grad():
                self.mean = self.mean * (1 - self.momentum) + mean * self.momentum
                self.var = self.var * (1 - self.momentum) + var * self.momentum

        else:
            self.out = self.scale * (x - self.mean) / torch.sqrt(self.var + self.eps) + self.shift

        return self.out

    def parameters(self):
        return [self.scale, self.shift]

class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []

class Embedding:

    def __init__(self, fan_in, fan_out):
        self.weights = torch.randn((fan_in, fan_out))

    def __call__(self, x):
        self.out = self.weights[x]
        return self.out

    def parameters(self):
        return [self.weights]

class FlattenConsecutive:

    def __init__(self, n):
        self.n = n

    def __call__(self, x):
        B, T, C = x.shape
        self.out = x.view(B, T // self.n, C * self.n)
        if self.out.shape[1] == 1:
            self.out = self.out.squeeze(1)

        return self.out

    def parameters(self):
        return []

class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)

        self.out = x
        return self.out

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]