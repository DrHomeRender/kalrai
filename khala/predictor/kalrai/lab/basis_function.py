import matplotlib.pyplot as plt
import numpy as np

def make_nonlinear(seed=0):
    np.random.seed(seed)
    n_samples =30
    X =np.sort(np.random.rand(n_samples))
    y= np.sin(2 * np.pi *X) + np.random.rand(n_samples) * 0.1
    X = X[:, np.newaxis]
    return X, y

X, y = make_nonlinear()
plt.scatter(X, y)
plt.xlabel("X")
plt.ylabel("y")
plt.title("Nonlinear function")
plt.show()
# 기저함수
phi_0 = np.polynomial.Polynomial.basis(0)
phi_1 = np.polynomial.Polynomial.basis(1)
phi_2 = np.polynomial.Polynomial.basis(2)
phi_3 = np.polynomial.Polynomial.basis(3)

x = np.linspace(-1, 1, 100)

plt.plot(x, phi_0(x), label="d=0")
plt.plot(x, phi_1(x), label="d=1")
plt.plot(x, phi_2(x), label="d=2")
plt.plot(x, phi_3(x), label="d=3")
plt.legend()
plt.title("bisic polynomial basis function")
plt.show()