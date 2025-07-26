# app-exploracao.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# 🎨 Estilo Seaborn
sns.set_theme(style="whitegrid")


# Título e descrição
st.title("🎬 Análise Exploratória da Base de Bilheteria Mundial (1977–2025)")
st.markdown("""
Esta aplicação explora dados de filmes lançados entre 1977 e 2025 com receitas ajustadas pela inflação (base 2024). 
As informações incluem receita doméstica, internacional, abertura, duração, rating e distribuidor.
""")

# Carregar os dados tratados
df = pd.read_parquet("dados_tratados.parquet")  # substitua conforme sua exportação

# Filtros interativos
st.sidebar.header("🎛️ Filtros")
anos = sorted(df["release_year"].dropna().unique())
ano_selecionado = st.sidebar.slider("Ano de lançamento", int(min(anos)), int(max(anos)), (2000, 2025))

generos = df["genres"].dropna().unique().tolist()
generos_selecionados = st.sidebar.multiselect("Gêneros", generos, default=generos)

distribuidores = df["distributor_group"].dropna().unique().tolist()
distribuidores_selecionados = st.sidebar.multiselect("Grupos de Distribuidores", distribuidores, default=distribuidores)

df["release_year"] = df["release_year"].astype(int)

df_filtrado = df[
    (df["release_year"].between(ano_selecionado[0], ano_selecionado[1])) &
    (df["genres"].isin(generos_selecionados)) &
    (df["distributor_group"].isin(distribuidores_selecionados))
]

# Métricas principais
st.subheader("📊 Métricas Gerais")
col1, col2, col3 = st.columns(3)
col1.metric("Total de Filmes", f"{df_filtrado.shape[0]}")
col2.metric("Receita Total Ajustada (US$)", f"{df_filtrado['total_gross_adjusted'].sum():,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
col3.metric("Distribuidores Únicos", df_filtrado["domestic_distributor"].nunique())

# Gráfico: Receita por ano
st.subheader("📈 Receita Total Ajustada por Ano")
receita_por_ano = df_filtrado.groupby("release_year")["total_gross_adjusted"].sum().reset_index()
fig = px.line(receita_por_ano, x="release_year", y="total_gross_adjusted", title="Receita Ajustada ao Longo dos Anos")
fig.update_layout(yaxis_title="Total Receita (US$)", xaxis_title="Ano")
st.plotly_chart(fig, use_container_width=True)

# Gráfico: Top 10 filmes por receita ajustada
st.subheader("🏆 Top 10 Filmes por Receita Ajustada")
top_filmes = df_filtrado.sort_values("total_gross_adjusted", ascending=False).head(10)
fig2 = px.bar(top_filmes, x="total_gross_adjusted", y="movie_title", orientation='h', 
              title="Top 10 Filmes (Receita Ajustada)", color="distributor_group")
st.plotly_chart(fig2, use_container_width=True)

# Gráfico: Distribuição por grupo de distribuidor
st.subheader("🏢 Participação dos Grupos de Distribuidores")
grupo_dist = df_filtrado["distributor_group"].value_counts().reset_index()
fig3 = px.pie(grupo_dist, values="count", names="distributor_group", title="Distribuição dos Filmes por Grupo de Distribuidor")
st.plotly_chart(fig3, use_container_width=True)

# Gráfico: Receita por categoria de duração
st.subheader("⏱️ Receita por Duração do Filme")
duracao_df = df_filtrado.groupby("duration_category")["total_gross_adjusted"].sum().reset_index()
fig4 = px.bar(duracao_df, x="duration_category", y="total_gross_adjusted", title="Receita por Categoria de Duração")
st.plotly_chart(fig4, use_container_width=True)

# Tabela de amostra
st.subheader("🧾 Amostra dos Dados")
st.dataframe(df_filtrado[[
    "movie_title", "release_year", "genres", "indicative_rating", "running_time",
    "total_gross_adjusted", "domestic_opening_adjusted", "distributor_group"
]].sort_values(by="total_gross_adjusted").head(10000))

# Rodapé
st.markdown("---")
st.markdown("📌 Dados ajustados com base no CPI de 2024 (valor base = 313.69)")
