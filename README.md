# Classificação de Flores Iris com KNN

Este projeto foi criado como resposta a um desafio técnico proposto pelo Professor Doutor Italo Linhares. Ele demonstra a aplicação do algoritmo K-Nearest Neighbors (KNN) para classificar as espécies de flores do famoso dataset Iris. O objetivo é construir e avaliar um modelo que possa prever a espécie de uma flor Iris com base em suas características de sépalas e pétalas.

## 📊 O Dataset Iris

O dataset Iris é um dos mais clássicos na área de machine learning. Ele contém 150 amostras de flores Iris, divididas igualmente entre três espécies:
- **Iris Setosa**
- **Iris Versicolor**
- **Iris Virginica**

Para cada amostra, são fornecidas quatro características em centímetros:
- Comprimento da sépala (`sepal length (cm)`)
- Largura da sépala (`sepal width (cm)`)
- Comprimento da pétala (`petal length (cm)`)
- Largura da pétala (`petal width (cm)`)

## 🧠 O Algoritmo K-Nearest Neighbors (KNN)

KNN é um algoritmo de aprendizado supervisionado não-paramétrico usado para classificação e regressão. Ele funciona encontrando os `K` vizinhos mais próximos de um novo ponto de dados no espaço de características e atribuindo a esse ponto a classe mais comum entre seus vizinhos.

Neste projeto, o KNN é utilizado para classificar as flores Iris.

## 🚀 Estrutura do Projeto

O repositório contém os seguintes arquivos:

├── main.py            # Código principal: exploração, treinamento, avaliação e otimização
├── README.md          # Descrição do projeto
├── requirements.txt   # Lista de dependências do Python
└── .gitignore         # Arquivos e pastas ignorados pelo Git


## 🛠️ Tecnologias Utilizadas

* **Python 3.13**
* **Bibliotecas Python:**
    * `numpy`
    * `pandas`
    * `matplotlib`
    * `seaborn`
    * `scikit-learn`

## ⚙️ Como Rodar o Projeto

Siga os passos abaixo para configurar e executar o projeto em sua máquina local:

1.  **Clone o Repositório:**
    ```bash
    git clone https://github.com/DeveloperKairo/iris-classifier.git
    cd iris-classifier
    ```

2.  **Crie um Ambiente Virtual (Recomendado):**
    ```bash
    py -m venv venv
    ```

3.  **Ative o Ambiente Virtual:**
    * **Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    * **macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

4.  **Instale as Dependências:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Execute o Programa:**
    ```bash
    py main.py
    ```
    Isso rodará todo o programa em seu terminal.

## 📈 Resultados Principais

O projeto demonstra as seguintes etapas e resultados:

1.  **Análise Exploratória de Dados (EDA):**
    * Visualização das primeiras linhas do dataset, informações gerais e estatísticas descritivas.
    * Geração de um `pairplot` para visualizar as relações entre as características e a separabilidade das espécies

2.  **Pré-processamento de Dados:**
    * Divisão dos dados em conjuntos de treinamento e teste.
    * Escalonamento das características usando `StandardScaler`, crucial para algoritmos baseados em distância como o KNN.

3.  **Treinamento e Avaliação do Modelo KNN:**
    * Treinamento de um modelo KNN inicial (K=5).
    * Avaliação de desempenho usando:
        * **Acurácia:** (geralmente alta para este dataset, próxima de 100%).
        * **Matriz de Confusão:** Uma representação visual de acertos e erros por classe.
        * **Relatório de Classificação:** Métricas detalhadas como Precisão, Recall e F1-Score por classe.

4.  **Otimização do Parâmetro K:**
    * Utilização de **Validação Cruzada (Cross-Validation)** para encontrar o valor ótimo de `K` que maximize a acurácia média do modelo.
    * Geração de um gráfico mostrando a acurácia média da validação cruzada para diferentes valores de K

5.  **Avaliação do Modelo Otimizado:**
    * Treinamento do modelo KNN com o `K` ótimo encontrado.
    * Reavaliação no conjunto de teste para confirmar a performance aprimorada ou mantida.

---

## 🤝 Contribuições

Contribuições são bem-vindas! Se você tiver sugestões, melhorias ou encontrar algum problema, por favor, abra uma *issue* ou envie um *pull request*.

---

**Autor:** [Kairo Kaléo/DeveloperKairo]
**Data:** Maio de 2025