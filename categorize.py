import pandas as pd
import re

def load_mapping(filepath="categories_mapping.csv"):
    """
    Charge le fichier de mapping des mots-clés vers les catégories, avec support d'une colonne optionnelle 'type_condition'.
    """
    try:
        mapping = pd.read_csv(filepath, sep=';')
        mapping = mapping.dropna(subset=['catégorie'])  # on garde les lignes avec une catégorie définie
        return mapping
    except Exception as e:
        raise ValueError(f"Erreur lors du chargement du fichier de mapping : {e}")

def assign_categories(df, mapping):
    """
    Ajoute les colonnes :
    - 'Catégorie' (comme avant)
    - 'Matched mot_clé' : le mot-clé ayant permis le match
    - 'Matched type_condition' : le filtre de type utilisé (si applicable)
    """

    df['Catégorie'] = 'Inconnu'
    df['Matched mot_clé'] = ''
    df['Matched type_condition'] = ''

    has_type_column = 'Type' in df.columns

    for _, row in mapping.iterrows():
        mot_clé = str(row['mot_clé']) if not pd.isna(row['mot_clé']) else ''
        catégorie = str(row['catégorie']).strip()
        type_condition = str(row['type_condition']).strip() if 'type_condition' in row and pd.notna(row['type_condition']) else None

        if mot_clé:
            mask = df['Contrepartie'].str.lower().str.contains(re.escape(mot_clé.lower()), na=False)
        else:
            mask = pd.Series([True] * len(df))

        if has_type_column and type_condition and type_condition != '':
            mask = mask & (df['Type'].astype(str).str.strip() == type_condition)

        # Appliquer uniquement sur les lignes encore 'Inconnu' ou pas encore annotées
        update_mask = mask & (df['Catégorie'] == 'Inconnu')

        df.loc[update_mask, 'Catégorie'] = catégorie
        df.loc[update_mask, 'Matched mot_clé'] = mot_clé
        df.loc[update_mask, 'Matched type_condition'] = type_condition if type_condition else ''

    return df

