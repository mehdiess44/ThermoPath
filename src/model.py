import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ── Liste explicite des features retenues ──────────────────────────────
FEATURE_COLS: list[str] = [
    "thermal_shipper_temp_reading",
    "g_force",
    "temp_mean",
    "temp_std",
    "g_force_mean",
    "g_force_std",
    "temp_velocity",
]


def prepare_and_split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler]:
    """Prépare les données pour le ML : séparation, scission chronologique et scaling.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame nettoyé et enrichi (features rolling déjà calculées).
    test_size : float, optional
        Proportion du jeu de test (défaut 0.2).

    Returns
    -------
    tuple
        (X_train_scaled, X_test_scaled, y_train, y_test, scaler)
    """

    # ── Étape A : Séparation Features / Target ─────────────────────────
    X: pd.DataFrame = df[FEATURE_COLS]
    y: pd.Series = df["is_shock"]

    # ── Étape B : Scission Chronologique ───────────────────────────────
    # shuffle=False : OBLIGATOIRE pour respecter l'ordre temporel.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    # ── Étape C : Mise à l'échelle (StandardScaler) ────────────────────
    scaler = StandardScaler()

    # fit_transform sur le train UNIQUEMENT (pas de Data Leakage)
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )

    # transform sur le test
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    # ── Étape D : Retour ───────────────────────────────────────────────
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_evaluate_and_save(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    scaler: StandardScaler,
    model_dir: str = "models",
) -> IsolationForest:
    """Entraîne un IsolationForest, évalue sur le test set et sérialise les artefacts.

    Parameters
    ----------
    X_train : pd.DataFrame
        Features d'entraînement (déjà scalées).
    X_test : pd.DataFrame
        Features de test (déjà scalées).
    y_test : pd.Series
        Vérité terrain du jeu de test.
    scaler : StandardScaler
        Scaler déjà fitté (à sérialiser avec le modèle).
    model_dir : str, optional
        Répertoire de sortie pour les fichiers .pkl (défaut "models").

    Returns
    -------
    IsolationForest
        Le modèle entraîné.
    """

    # ── Étape A : Entraînement ─────────────────────────────────────────
    model = IsolationForest(contamination=0.0065, random_state=42)
    model.fit(X_train)

    # ── Étape B : Prédiction et Re-mapping ─────────────────────────────
    y_pred_raw: np.ndarray = model.predict(X_test)
    # IsolationForest : -1 = anomalie, 1 = normal  →  re-map vers 1/0
    y_pred_mapped: np.ndarray = np.where(y_pred_raw == -1, 1, 0)

    # ── Étape C : Évaluation (Métriques industrielles) ─────────────────
    print("\n" + "═" * 55)
    print("  --- ÉVALUATION DU MODÈLE ---")
    print("═" * 55)
    print("\nMatrice de confusion :")
    print(confusion_matrix(y_test, y_pred_mapped))
    print("\nRapport de classification :")
    print(classification_report(y_test, y_pred_mapped))

    # ── Étape D : Sérialisation (Export pour le SIL) ───────────────────
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, f"{model_dir}/model.pkl")
    joblib.dump(scaler, f"{model_dir}/scaler.pkl")
    print(f"✅ Modèle et scaler exportés dans '{model_dir}/' (model.pkl, scaler.pkl)")

    return model


# ── Point d'entrée pour test ───────────────────────────────────────────
if __name__ == "__main__":
    try:
        from src.data_prep import load_and_resample   # python -m src.model
        from src.features import inject_synthetic_faults, create_rolling_features
    except ModuleNotFoundError:
        from data_prep import load_and_resample        # python src/model.py
        from features import inject_synthetic_faults, create_rolling_features

    # 1. Chargement et préparation du dataset
    print("⏳ Chargement et resampling …")
    df = load_and_resample()
    print(f"   DataFrame chargé : {df.shape}")

    # 2. Injection de 3 chocs synthétiques
    print("⚡ Injection de 500 chocs synthétiques…")
    df = inject_synthetic_faults(df, num_shocks=500)

    # 3. Génération des rolling features (fenêtre = 5)
    print("📊 Création des rolling features (window=5)…")
    df = create_rolling_features(df, window_size=5)

    # 4. Préparation et split ML
    print("✂️  Préparation et scission temporelle…")
    X_train, X_test, y_train, y_test, scaler = prepare_and_split_data(df)

    # 5. Affichage des résultats
    print("\n" + "═" * 55)
    print("  📐  RÉSULTATS DU SPLIT")
    print("═" * 55)
    print(f"  X_train shape : {X_train.shape}")
    print(f"  X_test  shape : {X_test.shape}")
    print(f"  y_train shape : {y_train.shape}")
    print(f"  y_test  shape : {y_test.shape}")

    print("\n── 🕐 Vérification de la continuité chronologique ──")
    print(f"  Dernière date  X_train : {X_train.index[-1]}")
    print(f"  Première date  X_test  : {X_test.index[0]}")
    assert X_train.index[-1] < X_test.index[0], (
        "❌ ERREUR : le split n'est PAS chronologique !"
    )
    print("  ✅ Split chronologique confirmé (train < test)")

    # 6. Entraînement, Évaluation et Sauvegarde
    print("\n🧠 Entraînement de l'IA et Évaluation…")
    train_evaluate_and_save(X_train, X_test, y_test, scaler)
