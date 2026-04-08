import pandas as pd


def load_and_resample(
    file_path: str = "data/raw/input_data.csv",
    batch_id: str = "batch001",
) -> pd.DataFrame:

    # ── 1. Chargement et Filtrage ──────────────────────────────────────
    df: pd.DataFrame = pd.read_csv(file_path)
    df = df[df["batch_id"] == batch_id].copy()

    # ── 2. Parsing Temporel ────────────────────────────────────────────
    df["date"] = pd.to_datetime(df["date"], format="mixed", dayfirst=True)
    df = df.set_index("date").sort_index()
    df = df[~df.index.duplicated(keep='first')]

    # ── 3. Dilatation (Upsampling) à la minute ────────────────────────
    df = df.asfreq("1min")

    # ── 4. Interpolation Physique ──────────────────────────────────────
    # Interpolation linéaire basée sur le temps pour la température
    df["thermal_shipper_temp_reading"] = df[
        "thermal_shipper_temp_reading"
    ].interpolate(method="time")

    # Forward Fill pour les colonnes catégorielles / non continues
    df = df.ffill()

    # ── 5. Garantie Qualité (DoD) ──────────────────────────────────────
    assert (
        df["thermal_shipper_temp_reading"].isna().sum() == 0
    ), "ERREUR QUALITÉ : des valeurs NaN subsistent dans thermal_shipper_temp_reading."

    return df


# ── Point d'entrée pour validation rapide ──────────────────────────────
if __name__ == "__main__":
    result: pd.DataFrame = load_and_resample()
    print("── HEAD ──")
    print(result.head())
    print(f"\n── SHAPE : {result.shape} ──")
