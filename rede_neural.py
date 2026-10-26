import numpy as np


def sigmoide(x):
    # Função de ativação: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))


class Neuronio:
    def __init__(self, pesos, bias):
        self.pesos = pesos
        self.bias = bias

    def propagacao_direta(self, entradas):
        # Calcula a soma ponderada das entradas e aplica a função de ativação
        soma = np.dot(self.pesos, entradas) + self.bias
        return sigmoide(soma)


pesos = np.array([0, 1])  # w1 = 0, w2 = 1
bias = 4  # b = 4
n = Neuronio(pesos, bias)

X = np.array([2, 3])  # x1 = 2, x2 = 3
print(n.propagacao_direta(X))  # Saída: 0.9990889488055994


class MinhaRedeNeural:
    """
    Uma rede neural com:
        - 2 entradas
        - uma camada oculta com 2 neurônios (h1, h2)
        - uma camada de saída com 1 neurônio (o1)
    Cada neurônio tem o mesmo peso e bias:
        - w1 = 0, w2 = 1
        - b = 0
    """

    def __init__(self):
        pesos = np.array([0, 1])
        bias = 0

        # Classe Neuronio é instanciada para cada neurônio da rede
        self.h1 = Neuronio(pesos, bias)
        self.h2 = Neuronio(pesos, bias)
        self.o1 = Neuronio(pesos, bias)

    def propagacao_direta(self, entradas):
        out_h1 = self.h1.propagacao_direta(entradas)
        out_h2 = self.h2.propagacao_direta(entradas)

        # As entradas para o1 são as saídas de h1 e h2
        out_o1 = self.o1.propagacao_direta(np.array([out_h1, out_h2]))

        return out_o1


network = MinhaRedeNeural()
X = np.array([2, 3])  # x1 = 2, x2 = 3
print(network.propagacao_direta(X))  # Saída: 0.7216325609518421
