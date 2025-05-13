import streamlit as st
import pandas as pd
from utils import load_and_clean_csv
from categorize import load_mapping, assign_categories
from visualize import plot_by_category

st.set_page_config(page_title="HFG - Analyse bancaire", layout="wide")
st.title("📊 Analyse consolidée des flux bancaires")

# Upload des deux fichiers
col1, col2 = st.columns(2)
with col1:
    qonto_file = st.file_uploader("📥 Fichier Qonto (.csv)", type="csv", key="qonto")
with col2:
    revolut_file = st.file_uploader("📥 Fichier Revolut (.csv)", type="csv", key="revolut")

if qonto_file and revolut_file:
    try:
        # Charger les fichiers avec identification de la source
        df_qonto = load_and_clean_csv(qonto_file)
        df_qonto["Source"] = "QONTO"

        df_revolut = load_and_clean_csv(revolut_file)
        df_revolut["Source"] = "REVOLUT"

        # Fusion
        df = pd.concat([df_qonto, df_revolut], ignore_index=True)

        # Catégorisation
        mapping_df = load_mapping("categories_mapping.csv")
        df = assign_categories(df, mapping_df)

        # Affichage
        st.success(f"{len(df)} opérations consolidées")
        st.write(f"📅 Période : {df['Date'].min().date()} ➡ {df['Date'].max().date()}")

        st.subheader("🔍 Aperçu des données")
        st.dataframe(df)

        st.subheader("📊 Répartition par catégorie")
        plot_by_category(df)

        # Export CSV
        csv = df.to_csv(index=False, sep=';').encode('utf-8')
        st.download_button(
            label="💾 Télécharger le CSV consolidé",
            data=csv,
            file_name="mouvements_categorises.csv",
            mime='text/csv'
        )

    except Exception as e:
        st.error(f"❌ Erreur lors du traitement : {e}")
else:
    st.info("Veuillez téléverser les deux fichiers bancaires (Qonto et Revolut).")
