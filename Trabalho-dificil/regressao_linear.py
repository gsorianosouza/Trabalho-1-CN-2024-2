# - numpy: Utilizado para operações matemáticas e manipulação de Arrays.
# - pandas:Utilizado para leitura e manipulação de dados em formatos como CSV.
# - matplotlib.pyplot: Utilizado para a criação de gráficos
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Usamos essa função para ler um arquivo CSV e extrair os dados de dentro dele.
def ler_csv(file_path):
    data = pd.read_csv(file_path, header=None)
    x = data[0].values
    fx = data[1].values
    return x, fx

#Essa função serve para ajustar a função quadrática aos dados que foram passados.
def fit_quadratic(x, fx):
    A = np.vstack([x**2, x, np.ones(len(x))]).T
    alpha = np.linalg.lstsq(A, fx, rcond=None)[0]
    return alpha

#Aqui exibimos os coeficientes da função quadrática ajustada.
def mostrar_solucao(alpha):
    print("Coeficientes da função quadrática:")
    print(f"α2 (coeficiente quadrático): {alpha[0]:.6f}")
    print(f"α1 (coeficiente linear): {alpha[1]:.6f}")
    print(f"α0 (constante): {alpha[2]:.6f}")

#Esta função plota e mostra os resultados dos ajustes realizados.
def plotar_resultados(x, fx, alpha):
    #Plota os dados originais que foram passados como pontos vermelhos.
    plt.scatter(x, fx, color='red', label='Dados Originais')
    #Gera os pontos para a curva ajustada (função quadrática).  
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = alpha[0] * x_fit**2 + alpha[1] * x_fit + alpha[2]
    #Plota a função ajustada como uma linha azul.  
    plt.plot(x_fit, y_fit, color='blue', label='Função Ajustada: $\\phi(x)$')  
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Ajuste da Função Quadrática')
    plt.legend()
    plt.grid()
    plt.show()

#Essa é a função principal que integra todas as outras etapas e chama os métodos.
def main(file_path):
    x, fx = ler_csv(file_path)
    alpha = fit_quadratic(x, fx)
    mostrar_solucao(alpha)
    plotar_resultados(x, fx, alpha)
#Caminho do arquivos de dados principais.
main('Trabalho-dificil/dados_originais.csv')
