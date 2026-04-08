import numpy as np
import pandas as pd


def inject_synthetic_faults(
    df: pd.DataFrame,
    num_shocks: int = 3,
    seed: int | None = 42,
) -> pd.DataFrame:
    """Injecte des anomalies synthétiques (chocs mécaniques + dérive thermique)
    dans un DataFrame temporel resamplé à la minute.

    Trois étapes successives :
        A. Création d'une baseline de vibrations mécaniques (bruit gaussien).
        B. Injection de *num_shocks* pics G-Force extrêmes (nids-de-poule).
        C. Ajout d'une dérive thermique cumulative (+0.05 °C/min) sur les
           60 minutes suivant chaque choc, simulant un pont thermique.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame indexé par un DatetimeIndex à fréquence 1 min.
        Doit contenir la colonne ``thermal_shipper_temp_reading``.
    num_shocks : int, optional
        Nombre de chocs à injecter (par défaut 3).
    seed : int | None, optional
        Graine pour la reproductibilité (par défaut 42).

    Returns
    -------
    pd.DataFrame
        DataFrame enrichi des colonnes ``g_force`` et ``is_shock``.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    n = len(df)

    # ── Étape A : Baseline mécanique (G-Force) ─────────────────────────
    df["g_force"] = rng.normal(loc=1.0, scale=0.05, size=n)
    df["is_shock"] = 0

    # ── Étape B : Injection des chocs (nids-de-poule) ──────────────────
    # On évite les 60 dernières minutes pour laisser de la place à la
    # dérive thermique de l'étape C.
    safe_zone = max(0, n - 60)
    if safe_zone < num_shocks:
        raise ValueError(
            f"Le DataFrame est trop court ({n} lignes) pour injecter "
            f"{num_shocks} chocs avec une marge de 60 minutes."
        )

    shock_positions: np.ndarray = rng.choice(
        safe_zone, size=num_shocks, replace=False
    )
    shock_positions.sort()

    for pos in shock_positions:
        # Pic G-Force extrême entre 3.0 et 5.0
        df.iloc[pos, df.columns.get_loc("g_force")] = rng.uniform(3.0, 5.0)
        df.iloc[pos, df.columns.get_loc("is_shock")] = 1

    # ── Étape C : Dérive thermique (conséquence physique) ──────────────
    temp_col = df.columns.get_loc("thermal_shipper_temp_reading")
    drift_window = 60  # minutes

    for pos in shock_positions:
        end = min(pos + drift_window, n)
        window_len = end - pos
        # Pénalité progressive : +0.05 °C à la min 1, +0.10 à la min 2, …
        penalty = np.linspace(0.05, 0.05 * window_len, num=window_len)
        df.iloc[pos:end, temp_col] += penalty

    return df


# ── Point d'entrée pour test ───────────────────────────────────────────
if __name__ == "__main__":
    try:
        from src.data_prep import load_and_resample   # python -m src.features
    except ModuleNotFoundError:
        from data_prep import load_and_resample        # python src/features.py

    print("⏳ Chargement et resampling…")
    df = load_and_resample().iloc[:5000]
    print(f"   DataFrame chargé : {df.shape}")

    print("⚡ Injection des failles synthétiques…")
    df = inject_synthetic_faults(df, num_shocks=3)

    total_shocks = df["is_shock"].sum()
    print(f"\n✅ Nombre total de chocs injectés : {total_shocks}")

    shocks = df[df["is_shock"] == 1]
    print("\n── Lignes de chocs ──")
    print(shocks[["g_force", "is_shock", "thermal_shipper_temp_reading"]])
