![Imagem temática gerada com o chatgpt para ser usada como a capa do MVP](https://github.com/paulalcardoso/MVP_machine_learning/blob/main/Imagem_Capa_MVP_ML.png)

# MVP Machine Learning & Analytics
Este projeto foi desenvolvido como um MVP do módulo Machine Learning & Analytics para **prever qual filme será o vencedor da categoria Best Picture (Melhor Filme) do Oscar**, a partir de dados históricos dos indicados.

Meu desafio foi transformar a pergunta “qual será o filme vencedor do Oscar de Melhor Filme?” em um problema de aprendizado de máquina. Para isso, considerei o problema como uma classificação: em cada ano, existem vários filmes indicados, mas apenas um vencedor. A minha premissa foi que dados de bilheteria poderiam ser bons preditores, pois podem adicionar informações mais relevantes do que quem são os atores ou quais estúdios produziram o filme. 

Um dos dificultadores foi trabalhar com datasets diferentes do Kaggle. De um lado, usei dados históricos das nomeações ao Oscar; de outro, dados sobre os filmes coletados no IMDb. O cruzamento dessas informações exigiu bastante tratamento e, nesse processo, acabei perdendo alguns filmes, que não identifiquei em ambas as bases.

No fim, utilizei dois conjuntos de dados principais:

**full_data.csv** – contém as informações de todas as nomeações ao Oscar, em todas as categorias. Suas colunas são:
**Ceremony** → Número da cerimônia.
**Year** → Ano da premiação.
**Class** → Grande categoria da premiação.
**CanonicalCategory** → Categoria padronizada.
**Category** → Categoria original.
**NomId** → Identificador único da nomeação.
**Film** → Nome do filme indicado.
**FilmId** → Identificador do filme.
**Name** → Nome da pessoa indicada (ator, diretor, etc.), quando aplicável.
**Nominees** → Lista de indicados associados.
**NomineeIds** → Identificadores dos indicados.
**Winner** → Indica se venceu (True/False).
**Detail** → Informação adicional sobre o papel ou contribuição.
**Note** → Observações adicionais.
**Citation** → Citação associada, se houver.
**MultifilmNomination** → Indica se a nomeação está associada a mais de um filme.

**cleaned_data_from_1920_to_2025.csv** – Informações tratadas de bilheteria. Suas colunas são:
**id** → Identificador IMDb do filme.
**title** → Título do filme.
**duration** → Duração.
**MPA** → Classificação indicativa.
**rating** → Nota média no IMDb.
**votes** → Número de votos no IMDb.
**meta_score** → Avaliação de críticos.
**description** → Sinopse do filme.
**movie_link** → Link IMDb.
**writers** → Lista de roteiristas.
**directors** → Lista de diretores.
**stars** → Atores principais.
**budget** → Orçamento estimado.
**opening_weekend_gross** → Receita de estreia.
**gross_worldwide** → Bilheteria mundial.
**gross_us_canada** → Bilheteria nos EUA e Canadá.
**release_date** → Data de lançamento.
**countries_origin** → País(es) de origem.
**filming_locations** → Locais de filmagem.
**production_companies** → Estúdios/produtoras.
**awards_content** → Resumo de prêmios e indicações.
**genres** → Gêneros.
**languages** → Idiomas.

Antes de separar os dados para treino e teste, precisei realizar diversos ajustes, como corrigir o ano de lançamento e normalizar o título dos filmes. Utilizei uma chave composta de título-ano de lançamento para unir os datasets e defini uma regra de tolerância de até 3 anos (baseada em estatísticas) para aumentar a quantidade de filmes corretamente combinados. Após a junção, tratei linhas duplicadas mantendo apenas a linha mais completa (com menos valores nulos) e utilizei uma técnica de fuzzy matching para identificar pares de títulos que ainda não haviam sido associados. Com esse processo, consegui identificar 608 dos 611 filmes indicados. 

Uma vez com o dataset unificado, reduzi a sua dimensão eliminando variáveis irrelevantes para a predição, como links para páginas do IMDb, ficando com 9 atributos a serem usados na predição. Este MVP é um problema de classificação binária, em que a variável target é a coluna Winner, oriunda do dataset dos filmes indicados, com valor True para os vencedores e nulo para os demais indicados.

Na sequência, dividi os dados em treino (80%) e teste (20%) de forma estratificada, garantindo que a proporção entre as classes fosse preservada. Inicialmente identifiquei que faria sentido utilizar validação cruzada estratificada (k=5), já que o dataset é desbalanceado, mas optei por só aplicar essa técnica no modelo selecionado como candidato final.

Além disso, apliquei uma série de transformações de dados para tornar o conjunto mais consistente, informativo e pré-processado para o treinamento e teste:

- Converti colunas financeiras que estavam em formato object, convertendo-as para valores numéricos (float), e desconsiderei valores que não são em dólar.

- Criei colunas auxiliares (_missing) para sinalizar valores ausentes antes de tratá-los, de forma a não perder e confundir informação.

- Apliquei One-Hot Encoding para transformar variáveis categóricas, como países e gêneros, em variáveis numéricas.

- Criei uma coluna com a quantidade de indicações do filme, como um possível indicador de relevância na premiação.

- Testei a normalização das variáveis numéricas, gerando versões escaladas para que atributos em diferentes ordens de grandeza não dominassem o aprendizado.

Para avaliar o impacto dessas transformações, criei diferentes visões do dataset em 4 pipelines de pré-processamento: 

1. sem normalização e sem feature selection;
   
2. sem normalização e com feature selection;
   
3. com normalização e sem feature selection;
   
4. com normalização e com feature selection.

Esse processo me permitiu analisar como cada transformação influenciava os resultados e chegar a um conjunto enxuto e consistente de atributos para alimentar os modelos.

Para estabelecer uma referência mínima de desempenho, utilizei como baseline o DummyClassifier, configurado com a estratégia most_frequent. Além dele, avaliei dois modelos supervisionados bastante utilizados em problemas de classificação. O primeiro foi a Logistic Regression, configurando class_weight="balanced", que ajusta o peso de cada classe de acordo com a sua frequência — atribuindo maior peso para a classe minoritária (vencedores) e menor para a classe majoritária (demais indicados). O segundo modelo foi o XGBoost (algoritmo baseado em gradient boosting com boa capacidade de lidar com desbalanceamento), com o parâmetro scale_pos_weight, definido como a razão entre a quantidade de instâncias negativas (indicados) e positivas (vencedores) no dataset.

Ambos os modelos de regressão logística e XGBoost foram devidamente treinados nos quatro diferentes pipelines de pré-processamento apresentados anteriormente. Na comparação entre eles, tanto a regressão logística quanto o XGBoost superaram o baseline do DummyClassifier. A regressão logística se destacou por ser simples e rápida, além de ter obtido um recall mais alto, conseguindo identificar mais vencedores. No entanto, essa vantagem veio acompanhada de uma queda na precisão, o que reduziu o equilíbrio geral e resultou em um F1 inferior. O XGBoost, por sua vez, apresentou um desempenho mais consistente em termos de F1 ponderado, mesmo com recall menor. Além disso, atingiu bons resultados sem depender de normalização ou seleção de atributos, o que reforça sua robustez. Esses fatores levaram à escolha do XGBoost otimizado como modelo a ter seus hiperparâmetros otimizados e resultados de treino e teste avaliados.

Na etapa seguinte, ao aplicar a validação cruzada estratificada (StratifiedKFold) junto com o RandomizedSearchCV, percebi que o XGBoost (escolhido na etapa anterior) apresentou overfitting, devido a diferença entre os resultados de treino e validação. Para mitigar esse problema, ampliei o conjunto de hiperparâmetros a serem otimizados. Além de ajustar o número de estimadores (n_estimators), a profundidade máxima das árvores (max_depth) e a taxa de aprendizado (learning_rate), passei a incluir também parâmetros como min_child_weight, subsample e colsample_bytree. O objetivo foi encontrar um equilíbrio na complexidade do modelo, buscando reduzir falsos positivos e, ao mesmo tempo, priorizar o acerto da classe dos vencedores, que é a mais difícil de identificar nesse cenário desbalanceado.

Como o meu dataset é fortemente desbalanceado (apenas um vencedor por ano contra vários indicados), a acurácia não é uma métrica adequada: mesmo um modelo que sempre prevê a classe majoritária (não vencedor) alcança valores altos de acurácia, sem de fato resolver o problema. Por isso, para avaliar se a otimização do XGBoost foi eficaz, considerei principalmente as métricas Precisão, Recall e F1-Score da classe vencedora, além do F1 ponderado para o equilíbrio geral do modelo. Com a ampliação dos hiperparâmetros, o XGBoost foi retreinado comto o conjunto de treinoe o resultado foi um modelo mais equilibrado, mas e o trade-off ficou claro: ao buscar reduzir o overfitting e evitar tantos falsos positivos, obtive uma perda em recall, mas os resultados de validação ficaram mais próximos dos obtidos no treino, mostrando que o modelo passou a generalizar melhor.

Para finalizar, testei também um ensemble homogêneo (Voting Classifier) combinando o XGBoost otimizado com a regressão logística balanceada escolhida anteriormente. A intenção foi para reduzir falsos positivos sem comprometer o recall. No entanto, apesar de trazer algum ganho localizado em precisão, o ensemble não superou o desempenho do XGBoost otimizado em termos de equilíbrio geral das métricas. Com isso, concluí que o XGBoost otimizado permaneceu como o melhor candidato, oferecendo o trade-off mais adequado entre precisão e recall e apresentando maior capacidade de generalização em comparação às demais abordagens.


# 🎬 MVP Machine Learning & Analytics

Este projeto foi desenvolvido como um MVP do módulo Machine Learning & Analytics para **prever qual filme será o vencedor da categoria Best Picture (Melhor Filme) do Oscar**, a partir de dados históricos dos indicados.

---

## 1. Definição do Problema  

Meu desafio foi transformar a pergunta **“qual será o filme vencedor do Oscar de Melhor Filme?”** em um problema de aprendizado de máquina. Para isso, considerei o problema como uma **classificação**: em cada ano, existem vários filmes indicados, mas apenas um vencedor.  

A minha premissa foi que **dados de bilheteria poderiam ser bons preditores**, pois podem adicionar informações mais relevantes do que apenas quem são os atores ou quais estúdios produziram o filme.  

Um dos dificultadores foi trabalhar com **datasets diferentes** do Kaggle. De um lado, usei dados históricos das nomeações ao Oscar; de outro, dados sobre os filmes coletados no IMDb. O cruzamento dessas informações exigiu bastante tratamento e, nesse processo, acabei perdendo alguns filmes que não identifiquei em ambas as bases.  

No fim, utilizei dois conjuntos de dados principais:  

**`full_data.csv`** – contém as informações de todas as nomeações ao Oscar, em todas as categorias. Suas colunas são:  
- **Ceremony** → Número da cerimônia.  
- **Year** → Ano da premiação.  
- **Class** → Grande categoria da premiação.  
- **CanonicalCategory** → Categoria padronizada.  
- **Category** → Categoria original.  
- **NomId** → Identificador único da nomeação.  
- **Film** → Nome do filme indicado.  
- **FilmId** → Identificador do filme.  
- **Name** → Nome da pessoa indicada (ator, diretor, etc.), quando aplicável.  
- **Nominees** → Lista de indicados associados.  
- **NomineeIds** → Identificadores dos indicados.  
- **Winner** → Indica se venceu (True/False).  
- **Detail** → Informação adicional sobre o papel ou contribuição.  
- **Note** → Observações adicionais.  
- **Citation** → Citação associada, se houver.  
- **MultifilmNomination** → Indica se a nomeação está associada a mais de um filme.  

**`cleaned_data_from_1920_to_2025.csv`** – Informações tratadas de bilheteria. Suas colunas são:  
- **id** → Identificador IMDb do filme.  
- **title** → Título do filme.  
- **duration** → Duração.  
- **MPA** → Classificação indicativa.  
- **rating** → Nota média no IMDb.  
- **votes** → Número de votos no IMDb.  
- **meta_score** → Avaliação de críticos.  
- **description** → Sinopse do filme.  
- **movie_link** → Link IMDb.  
- **writers** → Lista de roteiristas.  
- **directors** → Lista de diretores.  
- **stars** → Atores principais.  
- **budget** → Orçamento estimado.  
- **opening_weekend_gross** → Receita de estreia.  
- **gross_worldwide** → Bilheteria mundial.  
- **gross_us_canada** → Bilheteria nos EUA e Canadá.  
- **release_date** → Data de lançamento.  
- **countries_origin** → País(es) de origem.  
- **filming_locations** → Locais de filmagem.  
- **production_companies** → Estúdios/produtoras.  
- **awards_content** → Resumo de prêmios e indicações.  
- **genres** → Gêneros.  
- **languages** → Idiomas.  

---

## 2. Preparação de Dados  

Antes de separar os dados para treino e teste, precisei realizar diversos ajustes, como corrigir o ano de lançamento e normalizar o título dos filmes. Utilizei uma chave composta de **título + ano de lançamento** para unir os datasets e defini uma **regra de tolerância de até 3 anos** (baseada em estatísticas) para aumentar a quantidade de filmes corretamente combinados.  

Após a junção, tratei **linhas duplicadas**, mantendo apenas a mais completa (com menos valores nulos), e utilizei **fuzzy matching** para identificar pares de títulos que ainda não haviam sido associados. Com esse processo, consegui identificar **608 dos 611 filmes indicados**.  

Em seguida, reduzi a dimensão do dataset eliminando variáveis irrelevantes (como links para páginas do IMDb), ficando com **9 atributos** a serem usados na predição. Este MVP é, portanto, um problema de **classificação binária**, em que a variável target é a coluna **Winner**, com valor **True** para vencedores e **nulo** para demais indicados.  

Na sequência, dividi os dados em **treino (80%) e teste (20%) de forma estratificada**, garantindo que a proporção entre as classes fosse preservada. Inicialmente considerei utilizar validação cruzada estratificada (k=5) já nesta etapa, mas optei por aplicar essa técnica apenas no modelo candidato final.  

Além disso, apliquei uma série de **transformações de dados**:  
- **Converter** colunas financeiras de `object` para `float`, desconsiderando valores que não estavam em dólar.  
- **Criar** colunas auxiliares (`_missing`) para sinalizar valores ausentes antes do tratamento.  
- **Aplicar** One-Hot Encoding em variáveis categóricas (países, gêneros etc.).  
- **Criar** coluna com a quantidade de indicações do filme em outras categorias.  
- **Normalizar** variáveis numéricas, gerando versões escaladas para comparação.  

Para avaliar o impacto dessas transformações, criei **quatro pipelines de pré-processamento**:  
1. sem normalização e sem feature selection;  
2. sem normalização e com feature selection;  
3. com normalização e sem feature selection;  
4. com normalização e com feature selection.  

---

## 3. Modelagem e Treinamento  

Para estabelecer uma referência mínima de desempenho, utilizei como baseline o **DummyClassifier** (*most_frequent*). Esse modelo sempre prevê a classe majoritária (não vencedor) e, portanto, apresenta recall nulo para vencedores.  

Além do baseline, avaliei dois modelos supervisionados:  
- **Logistic Regression**, configurada com `class_weight="balanced"`, atribuindo maior peso para a classe minoritária (vencedores).  
- **XGBoost**, com o parâmetro `scale_pos_weight`, definido como a razão entre instâncias negativas (indicados) e positivas (vencedores).  

Ambos os modelos foram treinados nos quatro pipelines de pré-processamento. Tanto a Regressão Logística quanto o XGBoost superaram o baseline. A Regressão Logística obteve **recall mais alto**, identificando mais vencedores, mas à custa de menor precisão e F1. Já o XGBoost apresentou **desempenho mais consistente em termos de F1 ponderado**, mesmo com recall menor, além de ser robusto a diferentes transformações.  

Na etapa seguinte, apliquei **validação cruzada estratificada (StratifiedKFold)** junto com o **RandomizedSearchCV** para otimizar o XGBoost. Inicialmente observei **overfitting**, já que as métricas no treino estavam muito acima das métricas de validação. Para mitigar, ampliei o espaço de hiperparâmetros: além de `n_estimators`, `max_depth` e `learning_rate`, inclui `min_child_weight`, `subsample` e `colsample_bytree`, além de regularizações. O objetivo foi equilibrar a complexidade do modelo, reduzindo falsos positivos e priorizando a identificação da classe vencedora.  

---

## 4. Avaliação de Resultados  

Como o dataset é fortemente desbalanceado, a **acurácia não é uma métrica adequada**: um modelo que sempre prevê a classe majoritária já alcançaria acurácia alta sem resolver o problema. Por isso, concentrei a avaliação em **Precisão, Recall e F1-Score da classe vencedora**, além do **F1 ponderado** para medir o equilíbrio global.  

Com a otimização, o XGBoost foi **re-treinado em toda a base de treino** (automático pelo `RandomizedSearchCV`) e só depois avaliado no conjunto de teste. O trade-off observado foi claro: houve **pequena perda em recall**, mas em troca os resultados de validação ficaram mais próximos dos de treino, indicando que o modelo passou a **generalizar melhor**.  

Também testei um **ensemble homogêneo (Voting Classifier)** combinando o XGBoost otimizado e a Regressão Logística balanceada. Embora tenha aumentado o recall, perdeu em precisão e não superou o XGBoost otimizado em termos de F1 global.  

#### Comparação das Métricas  

| Modelo               | Precision (classe vencedora) | Recall (classe vencedora) | F1 (classe vencedora) | F1 Ponderado |
|----------------------|------------------------------|----------------------------|------------------------|--------------|
| **DummyClassifier**  | 0.00                         | 0.00                       | 0.00                   | 0.77         |
| **XGBoost otimizado**| 0.33                         | 0.32                       | 0.32                   | 0.79         |
| **Ensemble (Voting)**| 0.26                         | 0.37                       | 0.30                   | 0.76         |

📌 **Interpretação:**  
- O Dummy confirmou a dificuldade do problema, servindo apenas como baseline.  
- O XGBoost otimizado apresentou o melhor equilíbrio entre precisão, recall e F1, superando amplamente o baseline.  
- O Voting aumentou recall, mas perdeu em precisão e não superou o XGBoost no equilíbrio geral.  

Esses resultados fazem sentido diante da forte desproporção entre vencedores e não vencedores: **aumentar recall implica arriscar mais falsos positivos**, e o desafio é justamente encontrar o ponto de equilíbrio.  

Assim, concluí que o **XGBoost otimizado** é a melhor solução encontrada, por apresentar o trade-off mais adequado entre precisão e recall e maior capacidade de generalização.  

---
