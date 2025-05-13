import pandas as pd

def load_and_clean_csv(filepath):
    """
    Charge et nettoie un fichier CSV bancaire, en détectant automatiquement le format :
    - Format HFG (Qonto) : Date de la valeur (UTC), Nom de la contrepartie, Montant total (TTC)
    - Format Revolut : Date completed (UTC), Description, Amount, Type
    """
    try:
        df = pd.read_csv(filepath, sep=None, engine='python')  # auto-détection du séparateur
    except Exception as e:
        raise ValueError(f"Erreur lors du chargement du fichier : {e}")

    # Format HFG (Qonto)
    if {'Date de la valeur (UTC)', 'Nom de la contrepartie', 'Montant total (TTC)'}.issubset(df.columns):
        df['Date'] = pd.to_datetime(df['Date de la valeur (UTC)'], format="%d-%m-%Y %H:%M:%S")
        df['Montant (€)'] = df['Montant total (TTC)'].astype(str).str.replace(',', '.').astype(float)
        df['Contrepartie'] = df['Nom de la contrepartie'].str.strip()
        df['Mois'] = df['Date'].dt.to_period('M')
        return df[['Date', 'Mois', 'Contrepartie', 'Montant (€)']]

    # Format Revolut ou équivalent
    elif {'Date completed (UTC)', 'Description', 'Amount'}.issubset(df.columns):
        df['Date'] = pd.to_datetime(df['Date completed (UTC)'])
        df['Montant (€)'] = df['Amount'].astype(float)
        df['Contrepartie'] = df['Description'].astype(str).str.strip()
        df['Mois'] = df['Date'].dt.to_period('M')

        if 'Type' in df.columns:
            df['Type'] = df['Type'].astype(str).str.strip()
            return df[['Date', 'Mois', 'Contrepartie', 'Montant (€)', 'Type']]
        else:
            return df[['Date', 'Mois', 'Contrepartie', 'Montant (€)']]

    # Aucun format reconnu
    raise ValueError("Format de fichier non reconnu. Veuillez vérifier les colonnes.")
