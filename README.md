![Imagem temática gerada com o chatgpt para ser usada como a capa do MVP](https://github.com/paulalcardoso/MVP_machine_learning/blob/main/Imagem_Capa_MVP_ML.png)

# 🎬 MVP Machine Learning & Analytics

---

## 1. Definição do Problema  

Este projeto foi desenvolvido como um MVP do módulo Machine Learning & Analytics para **prever qual filme será o vencedor da categoria Best Picture (Melhor Filme) do Oscar**, a partir de dados históricos dos indicados. Meu desafio foi transformar a pergunta **“qual será o filme vencedor do Oscar de Melhor Filme?”** em um problema de aprendizado de máquina. Para isso, considerei o problema como uma **classificação**: em cada ano, existem vários filmes indicados, mas apenas um vencedor. A minha premissa foi que **dados de bilheteria poderiam ser bons preditores**, pois podem adicionar informações mais relevantes do que apenas quem são os atores ou quais estúdios produziram o filme. 

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

Em seguida, reduzi a dimensão do dataset eliminando variáveis irrelevantes (como links para páginas do IMDb), ficando com **9 atributos** a serem usados na predição. Neste MVP  a variável target é a coluna **Winner**, com valor **True** para vencedores e **nulo** para demais indicados.  

Na sequência, dividi os dados em **treino (80%) e teste (20%) de forma estratificada**, garantindo que a proporção entre as classes fosse preservada. IA validação cruzada estratificada (k=5) será aplicada apenas no modelo candidato final.  

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

Ambos os modelos foram treinados nos quatro pipelines de pré-processamento, criando quatro visões diferentes. Tanto a Regressão Logística quanto o XGBoost superaram o baseline. A Regressão Logística obteve **recall mais alto**, identificando mais vencedores, mas à custa de menor precisão e F1. Já o XGBoost apresentou **desempenho mais consistente em termos de F1 ponderado**, mesmo com recall menor, além de ser robusto a diferentes transformações.  

Na etapa seguinte, apliquei **validação cruzada estratificada (StratifiedKFold)** junto com o **RandomizedSearchCV** para otimizar o XGBoost. Inicialmente observei **overfitting**, já que as métricas no treino estavam muito acima das métricas de validação. Para mitigar, aumentei a lista de hiperparâmetros: além de `n_estimators`, `max_depth` e `learning_rate`, inclui `min_child_weight`, `subsample` e `colsample_bytree`, além de regularizações. O objetivo foi equilibrar a complexidade do modelo, reduzindo falsos positivos e priorizando a identificação da classe vencedora.  

---

## 4. Avaliação de Resultados  

Como o dataset é fortemente desbalanceado, a **acurácia não é uma métrica adequada**: um modelo que sempre prevê a classe majoritária já alcançaria acurácia alta sem resolver o problema. Por isso, concentrei a avaliação em **Precisão, Recall e F1-Score da classe vencedora**, além do **F1 ponderado** para medir o equilíbrio global.  

Com a otimização, o XGBoost foi **re-treinado em toda a base de treino** (automático pelo `RandomizedSearchCV`) e só depois avaliado no conjunto de teste. O trade-off observado foi claro: houve **pequena perda em recall**, mas em troca os resultados de validação ficaram mais próximos dos de treino, indicando que o modelo passou a **generalizar melhor**, apesar de ainda apresentar overfit.  

Também testei um **ensemble heterogêneo (Voting Classifier)** combinando o XGBoost otimizado e a Regressão Logística balanceada. Embora tenha aumentado o recall, perdeu em precisão e não superou o XGBoost otimizado em termos de F1 global.  

#### Comparação das Métricas  

| Modelo               | Precision (classe vencedora) | Recall (classe vencedora) | F1 (classe vencedora) | F1 Ponderado |
|----------------------|------------------------------|----------------------------|------------------------|--------------|
| **DummyClassifier**  | 0.00                         | 0.00                       | 0.00                   | 0.77         |
| **XGBoost otimizado**| 0.33                         | 0.32                       | 0.32                   | 0.79         |
| **Ensemble (Voting)**| 0.26                         | 0.37                       | 0.30                   | 0.76         | 

---
## 5. Conclusões e Próximos Passos  

### Conclusões  
- O problema de previsão dos vencedores do Oscar foi tratado como uma **classificação** em dados tabulares, com forte desbalanceamento entre indicados (classe majoritária) e vencedores (classe minoritária).  
- O **Dummy Classifier** serviu como baseline, confirmando que qualquer modelo útil precisaria superar o simples acerto da classe majoritária.  
- Entre os modelos testados, o **XGBoost** apresentou desempenho mais consistente que a Regressão Logística, especialmente no equilíbrio entre *recall* e *precision*.  
- O **Voting Ensemble** (heterogêneo, combinando Regressão Logística e XGBoost) foi avaliado, mas não trouxe ganhos em relação ao XGBoost otimizado, reforçando a escolha deste como modelo final.  
- O ajuste de hiperparâmetros via *RandomizedSearchCV* melhorou o desempenho do XGBoost, aumentando sua capacidade de generalização, mas aumentou significativamente o tempo de treinamento (vide notebook).  
- O trabalho reforçou a importância de métricas como *recall* e *F1-score* para a classe de vencedores, uma vez que a simples acurácia é enganosa no contexto de classes desbalanceadas.
- O desafio do MVP foi encontrar o ponto de equilíbrio entre *aumentar recall e gerar mais falsos positivos*.

### Trade-offs observados  
- O aumento do *recall* veio em detrimento da *precision*: ao identificar mais vencedores, o modelo também trouxe mais falsos positivos.  
- O uso de ensemble ampliou a cobertura (maior *recall*), mas reduziu a confiança nas previsões (queda na *precision* e no *F1*).  

### Próximos passos  
- **Dados**: incorporar mais atributos dos filmes (ex.: indicações em outras categorias do Oscar, bilheteria detalhada) para enriquecer a base. Também considerar prever vencedores como um todo das premiações ao Oscar, e não só de *Best Picture*.  
- **Modelos**: testar métodos mais avançados de balanceamento de classes (ex.: SMOTE, ajustes adicionais em *class weights*).  
- **Tuning**: experimentar *Bayesian Optimization* ou outros métodos mais eficientes de busca para otimização dos hiperparâmetros, reduzindo o custo de tempo de treinamento.  
