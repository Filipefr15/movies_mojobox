# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# 🎨 Estilo Seaborn
sns.set_theme(style="whitegrid")

# 🔧 Configuração da página
st.set_page_config(page_title="🎬 Comparação de Modelos ML", layout="wide")

# ========================================
# 1. Título e descrição
# ========================================
st.title("🎬 Comparação de Modelos para Previsão de Receita de Filmes")

st.markdown("""
Este app compara os resultados de dois modelos de regressão treinados com PySpark:

- 🌳 **Random Forest Regressor**
- 🔥 **Gradient Boosted Trees (GBT) Regressor**

📁 Carregue os arquivos CSV exportados do PySpark contendo as predições de cada modelo.
""")

# ========================================
# 2. Upload dos arquivos
# ========================================
col1, col2 = st.columns(2)

with col1:
    rf_file = st.file_uploader("📂 Random Forest - CSV de predições", type="csv", key="rf")
with col2:
    gbt_file = st.file_uploader("📂 GBT - CSV de predições", type="csv", key="gbt")

# ========================================
# 3. Função de análise
# ========================================
def avaliar_modelo(df, real_col="total_gross_adjusted", pred_col="prediction"):
    df["abs_error"] = abs(df[real_col] - df[pred_col])
    df["smape"] = 100 * df["abs_error"] / ((abs(df[real_col]) + abs(df[pred_col])) / 2)
    mae = df["abs_error"].mean()
    smape = df["smape"].mean()
    r2 = 1 - ((df["abs_error"] ** 2).sum() / ((df[real_col] - df[real_col].mean()) ** 2).sum())
    return mae, smape, r2, df

# ========================================
# 4. Avaliação e comparação
# ========================================
if rf_file and gbt_file:
    df_rf = pd.read_csv(rf_file)
    df_gbt = pd.read_csv(gbt_file)

    pred_col_rf = "prediction_exp" if "prediction_exp" in df_rf.columns else "prediction"
    pred_col_gbt = "prediction_exp" if "prediction_exp" in df_gbt.columns else "prediction"

    mae_rf, smape_rf, r2_rf, df_rf = avaliar_modelo(df_rf, pred_col=pred_col_rf)
    mae_gbt, smape_gbt, r2_gbt, df_gbt = avaliar_modelo(df_gbt, pred_col=pred_col_gbt)

    st.subheader("📊 Comparativo de Métricas")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MAE RF", f"${mae_rf:,.0f}")
    col2.metric("MAE GBT", f"${mae_gbt:,.0f}", delta=f"${mae_rf - mae_gbt:,.0f}")
    col3.metric("R² RF", f"{r2_rf:.4f}")
    col4.metric("R² GBT", f"{r2_gbt:.4f}", delta=f"{r2_gbt - r2_rf:.4f}")

    col5, col6 = st.columns(2)
    col5.metric("SMAPE RF", f"{smape_rf:.2f}%")
    col6.metric("SMAPE GBT", f"{smape_gbt:.2f}%", delta=f"{smape_rf - smape_gbt:.2f}%")

    st.markdown("---")
    st.subheader("📈 Real vs Previsto (Scatter)")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x="total_gross_adjusted", y=pred_col_rf, data=df_rf, label="Random Forest", alpha=0.5)
    sns.scatterplot(x="total_gross_adjusted", y=pred_col_gbt, data=df_gbt, label="GBT", alpha=0.5)
    plt.plot([df_rf["total_gross_adjusted"].min(), df_rf["total_gross_adjusted"].max()],
             [df_rf["total_gross_adjusted"].min(), df_rf["total_gross_adjusted"].max()],
             'r--', label="Linha ideal")
    plt.xlabel("Receita Real Ajustada")
    plt.ylabel("Receita Prevista")
    plt.title("Comparação Real x Previsto")
    plt.legend()
    st.pyplot(fig)

    st.subheader("📊 Erro Absoluto por Modelo")

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.histplot(df_rf["abs_error"], bins=50, kde=True, label="Random Forest", color="blue", alpha=0.5)
    sns.histplot(df_gbt["abs_error"], bins=50, kde=True, label="GBT", color="orange", alpha=0.5)
    plt.xlabel("Erro Absoluto ($)")
    plt.ylabel("Frequência")
    plt.title("Distribuição do Erro Absoluto")
    plt.legend()
    st.pyplot(fig2)

    st.subheader("🎯 Ranking de Erros (Top 10)")

    col1, col2 = st.columns(2)
    col1.write("**Random Forest**")
    col1.dataframe(df_rf[["movie_title", "distributor_group", "total_gross_adjusted", pred_col_rf, "abs_error"]]
                   .sort_values(by="abs_error", ascending=False).head(10))
    col2.write("**GBT**")
    col2.dataframe(df_gbt[["movie_title", "distributor_group", "total_gross_adjusted", pred_col_gbt, "abs_error"]]
                   .sort_values(by="abs_error", ascending=False).head(10))
    
    st.subheader("🎯 Ranking de Acertos (Top 100)")

    col1, col2 = st.columns(2)
    col1.write("**Random Forest**")
    col1.dataframe(df_rf[["movie_title", "distributor_group", "total_gross_adjusted", pred_col_rf, "abs_error"]]
                   .sort_values(by="abs_error", ascending=True).head(100))
    col2.write("**GBT**")
    col2.dataframe(df_gbt[["movie_title", "distributor_group", "total_gross_adjusted", pred_col_gbt, "abs_error"]]
                   .sort_values(by="abs_error", ascending=True).head(100))
    
    # st.subheader("🎯 Ranking de Erros (Top 1000) (Distribuidores ≠ 'Major Studio')")

    # # Filtrar os DataFrames para excluir 'Major Studio'
    # df_rf_filtrado = df_rf[df_rf["abs_error"] < 10000000]
    # df_gbt_filtrado = df_gbt[df_gbt["abs_error"] < 10000000]

    # col1, col2 = st.columns(2)
    # col1.write("**Random Forest**")
    # col1.dataframe(
    #     df_rf_filtrado[["movie_title", "distributor_group", "total_gross_adjusted", pred_col_rf, "abs_error"]]
    #     .sort_values(by="abs_error", ascending=True)
    #     .head(1000)
    # )

    # col2.write("**GBT**")
    # col2.dataframe(
    #     df_gbt_filtrado[["movie_title", "distributor_group", "total_gross_adjusted", pred_col_gbt, "abs_error"]]
    #     .sort_values(by="abs_error", ascending=True)
    #     .head(1000)
    # )
    
    st.subheader("📅 MAE por Mês de Lançamento")

    # Verifica se existe a coluna release_month em ambos
    if "release_month" in df_rf.columns and "release_month" in df_gbt.columns:
        rf_monthly = df_rf.groupby("release_month")["abs_error"].mean().reset_index(name="MAE_RF")
        gbt_monthly = df_gbt.groupby("release_month")["abs_error"].mean().reset_index(name="MAE_GBT")

        monthly = pd.merge(rf_monthly, gbt_monthly, on="release_month")

        fig, ax = plt.subplots(figsize=(10,6))
        monthly.set_index("release_month")[["MAE_RF", "MAE_GBT"]].plot(kind="bar", ax=ax)
        plt.ylabel("Erro Médio Absoluto ($)")
        plt.title("Erro Médio por Mês de Lançamento")
        plt.xticks(rotation=0)
        st.pyplot(fig)

else:
    st.info("👈 Carregue os dois arquivos CSV (Random Forest e GBT) para iniciar a comparação.")
