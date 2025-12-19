import pandas as pd

def read_data(file_path):
    # Kitapla uyumlu kolon isimleri
    column_names = ["user", "activity", "timestamp",
                    "x-axis", "y-axis", "z-axis"]

    df = pd.read_csv(
        file_path,
        header=None,
        names=column_names,
        on_bad_lines="skip"   # bozuk satırları atla
    )

    # Satır sonundaki ';' işaretini temizle + float'a çevir
    df["z-axis"] = df["z-axis"].astype(str).str.replace(";", "", regex=False)
    df["z-axis"] = pd.to_numeric(df["z-axis"], errors="coerce")

    df.dropna(inplace=True)
    return df
