import pandas as pd

INPUT_COLUMNS = [
    "Product Name",
    "Main Ingredients (KR)",
    "Main Ingredients (EN)",
    "Product Form and Packaging",
    "Intended Use",
]
TARGET_COLUMN = "HS Code"
SHEET_NAME = "사례_전처리 완료"

def load_casebook(path, sheet_name=SHEET_NAME):
    df = pd.read_excel(path, sheet_name=sheet_name)
    required = INPUT_COLUMNS + [TARGET_COLUMN]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {missing}")

    df = df.copy()
    for col in INPUT_COLUMNS:
        df[col] = df[col].fillna("").astype(str).str.strip()

    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(str).str.strip()
    df = df[df[TARGET_COLUMN].ne("")].reset_index(drop=True)

    df["model_text"] = (
        "[PRODUCT] " + df["Product Name"] +
        " [INGREDIENTS_KR] " + df["Main Ingredients (KR)"] +
        " [INGREDIENTS_EN] " + df["Main Ingredients (EN)"] +
        " [FORM] " + df["Product Form and Packaging"] +
        " [USE] " + df["Intended Use"]
    )

    df["duplicate_group"] = pd.util.hash_pandas_object(
        df[INPUT_COLUMNS], index=False
    ).astype(str)

    return df

def dataset_audit(df):
    label_counts = df[TARGET_COLUMN].value_counts().rename_axis("HS Code").reset_index(name="count")
    group_counts = (
        df.groupby(TARGET_COLUMN)["duplicate_group"]
        .nunique()
        .rename("unique_input_groups")
        .reset_index()
    )
    label_counts = label_counts.merge(group_counts, on="HS Code", how="left")

    audit = {
        "n_rows": len(df),
        "n_hs_codes": df[TARGET_COLUMN].nunique(),
        "exact_duplicate_rows": int(df.duplicated(subset=INPUT_COLUMNS + [TARGET_COLUMN]).sum()),
        "duplicate_input_rows": int(df.duplicated(subset=INPUT_COLUMNS).sum()),
        "missing_values": df[INPUT_COLUMNS].replace("", pd.NA).isna().sum().to_dict(),
        "label_counts": label_counts,
    }
    return audit
