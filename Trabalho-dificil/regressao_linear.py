import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_csv(file_path):
    data = pd.read_csv(file_path, header=None)
    x = data[0].values
    fx = data[1].values
    return x, fx

def fit_quadratic(x, fx):
    A = np.vstack([x**2, x, np.ones(len(x))]).T
    alpha = np.linalg.lstsq(A, fx, rcond=None)[0]
    return alpha

def print_solution(alpha):
    print("Coeficientes da função quadrática:")
    print(f"α2 (coeficiente quadrático): {alpha[0]:.6f}")
    print(f"α1 (coeficiente linear): {alpha[1]:.6f}")
    print(f"α0 (constante): {alpha[2]:.6f}")

def plot_results(x, fx, alpha):
    plt.scatter(x, fx, color='red', label='Dados Originais')  
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = alpha[0] * x_fit**2 + alpha[1] * x_fit + alpha[2]  
    plt.plot(x_fit, y_fit, color='blue', label='Função Ajustada: $\\phi(x)$')  
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Ajuste da Função Quadrática')
    plt.legend()
    plt.grid()
    plt.show()

def main(file_path):
    x, fx = read_csv(file_path)
    alpha = fit_quadratic(x, fx)
    print_solution(alpha)
    plot_results(x, fx, alpha)

main('Trabalho-dificil/dados_originais.csv')