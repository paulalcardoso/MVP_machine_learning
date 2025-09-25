![Imagem temática gerada com o chatgpt para ser usada como a capa do MVP](https://github.com/paulalcardoso/MVP_machine_learning/blob/main/Imagem_Capa_MVP_ML.png)

# MVP Machine Learning & Analytics
Este projeto foi desenvolvido como um MVP do módulo Machine Learning & Analytics para **prever qual filme será o vencedor da categoria Best Picture (Melhor Filme) do Oscar**, a partir de dados históricos dos indicados.

Meu desafio foi transformar a pergunta “qual será o filme vencedor do Oscar de Melhor Filme?” em um problema de aprendizado de máquina. Para isso, considerei o problema como uma classificação: em cada ano, existem vários filmes indicados, mas apenas um vencedor. A minha premissa foi que dados de bilheteria poderiam ser bons preditores, pois podem adicionar informações mais relevantes do que quem são os atores ou quais estúdios produziram o filme. 

Um dos dificultadores foi trabalhar com datasets diferentes do Kaggle. De um lado, usei dados históricos das nomeações ao Oscar; de outro, dados sobre os filmes coletados no IMDb. O cruzamento dessas informações exigiu bastante tratamento e, nesse processo, acabei perdendo alguns filmes, que não identifiquei em ambas as bases.

No fim, utilizei dois conjuntos de dados principais:

full_data.csv – contém as informações de todas as nomeações ao Oscar, em todas as categorias. Suas colunas são:
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

cleaned_data_from_1920_to_2025.csv – Informações tratadas de bilheteria. Suas colunas são:
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
