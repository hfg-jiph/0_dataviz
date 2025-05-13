import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
from datetime import date

st.set_page_config(page_title="💰 Analyse du cash et du patrimoine", layout="wide")

# === Fichiers requis ===
flux_path = "output/mouvements_categorises.csv"
prev_path = "data/previsions_v2.csv"

if not os.path.exists(flux_path) or not os.path.exists(prev_path):
    st.error("❌ Fichier de flux ou de prévisions introuvable.")
    st.stop()


# === Onglets ===
tab_chargement, tab1, tab2 = st.tabs(["📥 Chargement", "💵 Cash", "📈 Patrimoine"])


with tab_chargement:
    st.title("📥 Chargement des extraits bancaires Qonto et Revolut")

    col1, col2 = st.columns(2)
    with col1:
        file_qonto = st.file_uploader("Extrait Qonto (.csv)", type="csv", key="qonto")
    with col2:
        file_revolut = st.file_uploader("Extrait Revolut (.csv)", type="csv", key="revolut")

    if file_qonto and file_revolut:
        try:
            from utils import load_and_clean_csv
            from categorize import load_mapping, assign_categories
            import pandas as pd, os

            # Chargement et concaténation
            df_qonto = load_and_clean_csv(file_qonto)
            df_qonto["Source"] = "Qonto"

            df_revolut = load_and_clean_csv(file_revolut)
            df_revolut["Source"] = "Revolut"

            df = pd.concat([df_qonto, df_revolut], ignore_index=True).sort_values("Date")
            mapping_df = load_mapping()
            df = assign_categories(df, mapping_df)

            st.success(f"{len(df)} opérations importées")
            st.write(f"Période couverte : {df['Date'].min().date()} ➡ {df['Date'].max().date()}")

            st.subheader("Aperçu des données importées")
            st.dataframe(df)

            # Sauvegarde
            os.makedirs("output", exist_ok=True)
            df.to_csv("output/mouvements_categorises.csv", sep=";", index=False)
            st.success("✅ Données sauvegardées dans 'output/mouvements_categorises.csv'")

        except Exception as e:
            st.error(f"❌ Erreur lors de l'import : {e}")

    else:
        st.info("Veuillez importer les deux fichiers Qonto et Revolut pour générer le fichier consolidé.")





# === ONGLET 1 : CASH ===
with tab1:

# === 🎛 Interface : pas de temps ===
    freq_options = {
        "Jour": "D",
        "Mois": "M"
    }
    selected_freq_label = st.selectbox("🕒 Pas de temps", list(freq_options.keys()), index=1)
    selected_freq = freq_options[selected_freq_label]

    st.title("💵 Évolution du cash réel & prévisionnel")
    df = pd.read_csv(flux_path, sep=';')
    df['Date'] = pd.to_datetime(df['Date'])
    df["Flux cash"] = df["Montant (€)"]
    df = df.sort_values("Date")

    df_prev = pd.read_csv(prev_path, sep=';')
    df_prev["Date prevue"] = pd.to_datetime(df_prev["Date prevue"], dayfirst=True)
    df_prev["Montant attendu"] = df_prev["Montant attendu"].astype(str).str.replace(",", ".").astype(float)
    df_prev["Flux cash prévu"] = df_prev.apply(
        lambda row: row["Montant attendu"] if row["Type de flux"].startswith("in_") else -row["Montant attendu"], axis=1
    )
    df_prev = df_prev.rename(columns={"Date prevue": "Date"})
    df_prev = df_prev.sort_values("Date")

    # Chargement des flux réels (pour récupérer les FdS)
    df_flux = pd.read_csv(flux_path, sep=';')
    df_flux['Date'] = pd.to_datetime(df_flux['Date'])
    df_fds = df_flux[df_flux["Catégorie"] == "FdS"].copy()
    df_fds["Période"] = df_fds["Date"].dt.to_period(selected_freq).dt.to_timestamp()
    fds_par_periode = df_fds.groupby("Période")["Montant (€)"].sum().to_dict()


    df_real = df.copy()
    df_real["Période"] = df_real["Date"].dt.to_period(selected_freq).dt.to_timestamp()
    df_prev["Période"] = df_prev["Date"].dt.to_period(selected_freq).dt.to_timestamp()

    # === 1. Histogramme cash réel ===
    agg_real = df_real.groupby("Période")["Flux cash"].sum().reset_index()
    agg_real["Cash cumulé"] = agg_real["Flux cash"].cumsum()
    agg_real["Cash cumulé (k€)"] = agg_real["Cash cumulé"] / 1000

    st.subheader("📊 1. Évolution du cash réel")
    fig1 = go.Figure()
    fig1.add_bar(x=agg_real["Période"], y=agg_real["Cash cumulé (k€)"], name="Cash réel")
    fig1.update_layout(
        yaxis_title="Montants (k€)", hovermode="x unified",
        xaxis_title="Date", title="Cash cumulé sans prévisionnel"
    )
    if selected_freq == "M":
        fig1.update_xaxes(
            tickformat="%b %Y",
            tickvals=[d for d in agg_real["Période"] if d.month in [3, 6, 9, 12]]
        )
    st.plotly_chart(fig1, use_container_width=True)

    # === 2. Histogramme FdS ===
    df_fds = df_real[df_real["Catégorie"] == "FdS"]
    agg_fds = df_fds.groupby("Période")["Montant (€)"].sum().reset_index()
    agg_fds["Montant (k€)"] = agg_fds["Montant (€)"] / 1000

    st.subheader("📊 2. Dépenses FdS (par période)")
    fig2 = go.Figure()
    fig2.add_bar(x=agg_fds["Période"], y=agg_fds["Montant (k€)"], name="FdS", marker_color='orange')
    fig2.update_layout(
        yaxis_title="Montants (k€)", hovermode="x unified",
        xaxis_title="Date", title="Frais de Structure dépensés"
    )
    if selected_freq == "M":
        fig2.update_xaxes(
            tickformat="%b %Y",
            tickvals=[d for d in agg_fds["Période"] if d.month in [3, 6, 9, 12]]
        )
    st.plotly_chart(fig2, use_container_width=True)

    # === 3. Combiné : cash réel + projection ===
    today = pd.to_datetime(date.today())
    agg_real["Cash réel (k€)"] = agg_real["Cash cumulé"] / 1000
    agg_real_filtered = agg_real[agg_real["Période"] <= today]

    agg_sim = df_prev.groupby("Période")["Flux cash prévu"].sum().reset_index()
    agg_sim["Période"] = pd.to_datetime(agg_sim["Période"])
    agg_sim_filtered = agg_sim[agg_sim["Période"] > today].copy()

    cash_today = agg_real_filtered["Cash cumulé"].iloc[-1] if not agg_real_filtered.empty else 0
    agg_sim_filtered["Cash prévisionnel"] = cash_today + agg_sim_filtered["Flux cash prévu"].cumsum()
    agg_sim_filtered["Cash prévisionnel (k€)"] = agg_sim_filtered["Cash prévisionnel"] / 1000

    st.subheader("📊 3. Cash réel + projection")
    fig3 = go.Figure()
    fig3.add_bar(
        x=agg_real_filtered["Période"],
        y=agg_real_filtered["Cash réel (k€)"],
        name="Cash réel",
        marker_color="steelblue"
    )
    fig3.add_trace(go.Scatter(
        x=agg_sim_filtered["Période"],
        y=agg_sim_filtered["Cash prévisionnel (k€)"],
        mode="lines+markers",
        name="Projection",
        line=dict(color="firebrick", width=3)
    ))
    fig3.update_layout(
        height=650,
        yaxis_title="Montants (k€)",
        xaxis_title="Date",
        hovermode="x unified",
        title="Cash réel (jusqu’à aujourd’hui) + projection future (connectée)"
    )
    if selected_freq == "M":
        fig3.update_xaxes(
            tickformat="%b %Y",
            tickvals=[d for d in pd.to_datetime(df_real["Période"].unique()) if d.month in [3, 6, 9, 12]]
        )
    st.plotly_chart(fig3, use_container_width=True)



# === ONGLET 2 : PATRIMOINE ===
with tab2:
    st.title("📈 Évolution du patrimoine global")

    # Option pour inclure ou non les flux futurs
    inclure_futur = st.checkbox("Inclure les flux futurs (au-delà d'aujourd'hui)", value=False)


    # Chargement et nettoyage des prévisions
    df_prev = pd.read_csv(prev_path, sep=';')
    df_prev["Date"] = pd.to_datetime(df_prev["Date prevue"], dayfirst=True)
    df_prev["Montant attendu"] = df_prev["Montant attendu"].astype(str).str.replace(",", ".").astype(float)
    df_prev["Type de flux"] = df_prev["Type de flux"].str.strip()
    df_prev["Classe d actif"] = df_prev["Classe d actif"].str.strip()
    # df_prev = df_prev[df_prev["Date"] <= pd.to_datetime(date.today())].copy()
    if not inclure_futur:
        df_prev = df_prev[df_prev["Date"] <= pd.to_datetime(date.today())].copy()

    df_prev = df_prev.sort_values("Date")

    # Interface de pas de temps
    freq_options = {
        "Mois": "M",
        "Trimestre": "Q",
        "Année": "Y"
    }
    selected_freq_label = st.selectbox("📊 Pas de temps (onglet patrimoine)", list(freq_options.keys()), index=0)
    selected_freq = freq_options[selected_freq_label]

    df_prev["Période"] = df_prev["Date"].dt.to_period(selected_freq).dt.to_timestamp()

    # Composantes et couleurs
    composantes = ["Cash", "Crowdlending", "Crypto", "Finance", "PE", "SCPI", "VC"]
    couleurs = {
        "Cash": "#1f77b4",
        "Crowdlending": "#aec7e8",
        "Crypto": "#ff9896",
        "Finance": "#9467bd",
        "PE": "#d62728",
        "SCPI": "#98df8a",
        "VC": "#2ca02c"
    }

    # Initialisation du patrimoine
    patrimoine = {col: 0.0 for col in composantes}
    patrimoine["Cash"] = 0  # Montant initial 4_000_000
    cumul = []

    for _, row in df_prev.iterrows():
        flux = row["Type de flux"]
        classe = row["Classe d actif"]
        montant = row["Montant attendu"]
        periode = row["Période"]

        if flux in ["in_facture", "in_loyer_scpi", "in_versement_interet"]:
            patrimoine["Cash"] += montant

        elif flux == "in_participation":
            patrimoine["Cash"] += montant

        elif flux == "in_remboursement_capital":
            patrimoine["Cash"] += montant
            if classe in composantes:
                patrimoine[classe] -= montant

        elif flux == "out_paiement":
            patrimoine["Cash"] -= montant

        elif flux in ["out_investissement", "out_renforcement"]:
            patrimoine["Cash"] -= montant
            if classe in composantes:
                patrimoine[classe] += montant

        cumul.append({"Période": periode, **patrimoine.copy()})


    df_cumul = pd.DataFrame(cumul).drop_duplicates("Période", keep="last")
    df_cumul.set_index("Période", inplace=True)

    # Remplir les périodes manquantes
    # Correction du warning pandas avec freq
    freq_pandas = "ME" if selected_freq == "M" else selected_freq
    all_periods = pd.date_range(start=df_cumul.index.min(), end=df_cumul.index.max(), freq=freq_pandas)
    df_cumul = df_cumul.reindex(all_periods, method="ffill")
    df_grouped_k = df_cumul / 1000  # conversion en k€

    # Création du tooltip global pour chaque période
    tooltips = []
    for idx, row in df_grouped_k.iterrows():
        total = row.sum()
        lines = [f"<b> {int(total):,}k€</b>"]
        for comp in composantes:
            montant = row[comp]
            pct = f"{(montant / total * 100):.0f}%" if total > 0 else "0%"
            lines.append(f"<span style='color:{couleurs[comp]}'>{comp} : {int(montant):,}k€ ({pct})</span>")
        html_tooltip = "<br>".join(lines).replace(",", ".")
        tooltips.append(html_tooltip)

    # Création du graphique empilé
    import plotly.graph_objects as go
    fig = go.Figure()

    for comp in composantes:
        fig.add_bar(
            x=df_grouped_k.index,
            y=df_grouped_k[comp],
            name=comp,
            marker_color=couleurs[comp],
            hoverinfo="skip"  # On désactive les infobulles par composante
        )

    # Ajout d'une trace invisible avec tout le texte
    fig.add_trace(go.Scatter(
        x=df_grouped_k.index,
        y=df_grouped_k.sum(axis=1),
        mode="markers",
        marker=dict(opacity=0),
        hovertemplate="%{customdata}<extra></extra>",
        customdata=tooltips,
        showlegend=False
    ))

    # Mise en page
    fig.update_layout(
        barmode="stack",
        yaxis_title="Montants (k€)",
        xaxis_title="Date",
        hovermode="x unified",
        height=500,
        title="Patrimoine global cumulé par classe d'actif",
        font=dict(size=14)
    )

    # Ticks sur les trimestres pour "M"
    if selected_freq == "M":
        fig.update_xaxes(
            tickformat="%b %Y",
            tickvals=[d for d in df_grouped_k.index if d.month in [3, 6, 9, 12]]
        )

    # Adaptation dynamique de l'axe Y
    max_val = df_grouped_k.sum(axis=1).max()
    fig.update_yaxes(range=[0, max_val * 1.1])

    st.plotly_chart(fig, use_container_width=True)
