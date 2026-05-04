"""
ENM412 – MAN Türkiye A.Ş.
Modül 1 – Veri Yükleme ve Geleneksel Tahmin
Yazarlar: Büşra ÇİL · İrem ÇELİK · Sevde SÖZDEN
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

FEATURE_COLS = [
    "lag_1","lag_3","lag_6","lag_12",
    "roll_mean_3","roll_mean_6","roll_std_3","roll_std_6","roll_max_3",
    "Ay","Yil","Ceyrek","sin_ay","cos_ay",
    "MPS_Toplam_Arac","MPS_lag_1",
    "MPS_LC12m","MPS_LC18m","MPS_Coach","MPS_Coach2","MPS_Skyliner",
    "ABC_enc","XYZ_enc",
]

# C sınıfı veya Z grubu → geleneksel yöntem
# A ve B sınıfı → ML
def ml_mi_kullan(abc, xyz):
    if abc == "C" or xyz == "Z":
        return False
    return True

TARGET_COL = "Talep"
PARCA_COL  = "Parça_Kodu"
TARIH_COL  = "Tarih"
SPLIT_COL  = "Split"


def veri_yukle(dosya_yolu: str) -> dict:
    xl = pd.ExcelFile(dosya_yolu)

    # ML hazır veri
    ml = pd.read_excel(xl, sheet_name="ML_Hazir_Veri", header=0)
    ml[TARIH_COL] = ml[TARIH_COL].astype(str).str.strip()
    ml[PARCA_COL] = ml[PARCA_COL].astype(str).str.strip()
    for c in FEATURE_COLS:
        if c in ml.columns:
            ml[c] = pd.to_numeric(ml[c], errors="coerce").fillna(0)
    ml[TARGET_COL] = pd.to_numeric(ml[TARGET_COL], errors="coerce").fillna(0)

    # ABC/XYZ
    abc = pd.read_excel(xl, sheet_name="ABC_XYZ_Segmentasyon", header=0)
    abc[PARCA_COL] = abc[PARCA_COL].astype(str).str.strip()

    # Optimizasyon parametreleri
    opt = pd.read_excel(xl, sheet_name="Optimizasyon_Parametreleri", header=0)
    opt[PARCA_COL] = opt[PARCA_COL].astype(str).str.strip()
    opt = opt.rename(columns={
        "Tedarik Süresi (gün)":    "LT_gun",
        "Birim Maliyet (TL)":      "Birim_Maliyet",
        "Sipariş Maliyeti (TL)":   "Siparis_Maliyeti",
        "Elde Tutma (TL/adet/ay)": "h",
        "Stoksuz Maliyet (TL)":    "p",
        "Başlangıç Stok":          "Baslangic_Stok",
    })
    opt["LT_ay"] = opt["LT_gun"] / 30

    # MPS
    mps = pd.read_excel(xl, sheet_name="MPS_Long_Format", header=0)
    mps[TARIH_COL] = mps[TARIH_COL].astype(str).str.strip()

    # Merge ABC ve opt bilgilerini ml'e ekle
    ml = ml.merge(abc[[PARCA_COL,"Ort_Aylik_Talep","Std_Sapma","CV","Segment"]],
                  on=PARCA_COL, how="left", suffixes=("","_abc"))
    ml = ml.merge(opt[[PARCA_COL,"LT_gun","LT_ay","Birim_Maliyet",
                        "Siparis_Maliyeti","h","p","Baslangic_Stok"]],
                  on=PARCA_COL, how="left")

    # ML mi geleneksel mi?
    ml["kullan_ml"] = ml.apply(
        lambda r: ml_mi_kullan(r.get("ABC","C"), r.get("XYZ","Z")), axis=1)

    train_df = ml[ml[SPLIT_COL]=="Train"].copy().reset_index(drop=True)
    test_df  = ml[ml[SPLIT_COL]=="Test"].copy().reset_index(drop=True)
    parcalar = sorted(ml[PARCA_COL].unique().tolist())

    ml_parcalar  = sorted(ml[ml["kullan_ml"]==True][PARCA_COL].unique().tolist())
    gel_parcalar = sorted(ml[ml["kullan_ml"]==False][PARCA_COL].unique().tolist())

    print(f"[Veri] {len(parcalar):,} parça | ML: {len(ml_parcalar):,} | Geleneksel: {len(gel_parcalar):,}")

    return {
        "ml_df":        ml,
        "abc_df":       abc,
        "opt_df":       opt,
        "mps_df":       mps,
        "train_df":     train_df,
        "test_df":      test_df,
        "parcalar":     parcalar,
        "ml_parcalar":  ml_parcalar,
        "gel_parcalar": gel_parcalar,
    }


def parca_verisi(ml_df, parca_kodu):
    pdf = ml_df[ml_df[PARCA_COL]==parca_kodu].sort_values(TARIH_COL)
    train = pdf[pdf[SPLIT_COL]=="Train"]
    test  = pdf[pdf[SPLIT_COL]=="Test"]
    return {
        "pdf": pdf, "train": train, "test": test,
        "ts_train": train[TARGET_COL].values,
        "ts_test":  test[TARGET_COL].values,
        "tarihler": pdf[TARIH_COL].tolist(),
        "kullan_ml": bool(pdf["kullan_ml"].iloc[0]) if "kullan_ml" in pdf.columns else True,
        "abc": str(pdf["ABC"].iloc[0]) if "ABC" in pdf.columns else "C",
        "xyz": str(pdf["XYZ"].iloc[0]) if "XYZ" in pdf.columns else "Z",
    }


def geleneksel_tahmin(ts_train, n_tahmin=6):
    """Hareketli Ort., Üstel Düzeltme, Naif."""
    n = len(ts_train)
    # Hareketli Ort.
    pencere = min(6, n)
    ho = [float(np.mean(ts_train[-pencere:]))] * n_tahmin
    # Üstel
    alpha = 0.3
    ustel_val = float(ts_train[0]) if n > 0 else 0.
    for v in ts_train:
        ustel_val = alpha*float(v) + (1-alpha)*ustel_val
    ustel = [ustel_val] * n_tahmin
    # Naif
    naif = [float(ts_train[-1]) if n > 0 else 0.] * n_tahmin
    return {"hareketli_ort": ho, "ustel": ustel, "naif": naif}


def metrik_hesapla(y_true, y_pred, model_adi=""):
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mask = y_true > 0
    mape = float(np.mean(np.abs((y_true[mask]-y_pred[mask])/y_true[mask]))*100) if mask.sum()>0 else np.nan
    return {"model": model_adi, "MAE": mae, "RMSE": rmse, "MAPE": mape}
