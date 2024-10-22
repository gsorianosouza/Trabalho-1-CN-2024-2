import numpy as np
import csv


# Função para ler o CSV e converter em uma matriz A e vetor b
def ler_csv(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        dados = [list(map(float, row)) for row in reader]
    
    # Separar a matriz A (coeficientes) e o vetor b (numero apos '=')
    A = np.array([row[:-1] for row in dados]) 
    b = np.array([row[-1] for row in dados]) 

    return A, b, 


def resolver_sistema(A, b):
    return np.linalg.solve(A, b)


def main():
    arquivo_csv = 'sistema_3x3.csv'  

    A, b = ler_csv(arquivo_csv)
    
    solucao = resolver_sistema(A, b)
 
    print(f"\nA solução do sistema é:", ', '.join(f"{x:.0f}" for x in solucao))


if __name__ == '__main__':
    main()
