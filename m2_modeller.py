"""
ENM412 – MAN Türkiye A.Ş.
Modül 2 – 4 ML Modeli (RF·XGB·LGB·CAT) + Multi-Output Tahmin
Yazarlar: Büşra ÇİL · İrem ÇELİK · Sevde SÖZDEN

Kural:
    A/B sınıfı → RF · XGBoost · LightGBM · CatBoost + Optuna
    C sınıfı veya Z grubu → Geleneksel (Hareketli Ort.)

Multi-output: 6 aylık tahmini tek seferde üretir (iteratif değil)
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
import optuna
import warnings
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

from m1_veri import FEATURE_COLS, TARGET_COL, PARCA_COL, metrik_hesapla, geleneksel_tahmin

# Opsiyonel kütüphaneler
try:
    import xgboost as xgb; XGB_OK=True
except: XGB_OK=False

try:
    import lightgbm as lgb; LGB_OK=True
except: LGB_OK=False

try:
    from catboost import CatBoostRegressor; CAT_OK=True
except: CAT_OK=False


def _get_Xy(df, hedef_cols=None):
    feat = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feat].fillna(0).values
    if hedef_cols:
        y = df[hedef_cols].fillna(0).values
    else:
        y = df[TARGET_COL].values
    return X, y, feat


def _tscv_rmse(model, X, y, n_splits=4):
    """Zaman serisi CV ile RMSE hesapla."""
    from sklearn.metrics import mean_squared_error
    n = len(X)
    fold = n // (n_splits + 1)
    scores = []
    for i in range(1, n_splits+1):
        tr_end = i * fold
        val_end = min(tr_end + fold, n)
        if val_end <= tr_end: continue
        try:
            m = type(model)(**model.get_params())
            m.fit(X[:tr_end], y[:tr_end])
            pred = m.predict(X[tr_end:val_end])
            scores.append(float(np.sqrt(mean_squared_error(
                y[tr_end:val_end].flatten(), pred.flatten()))))
        except: pass
    return float(np.mean(scores)) if scores else 1e9


def _multi_hedef_olustur(df, n_ay=6):
    """
    Multi-output için hedef sütunları oluştur:
    t+1, t+2, ..., t+6 talebi.
    """
    parca_dfs = []
    for pid, grp in df.groupby(PARCA_COL):
        grp = grp.sort_values("Tarih").copy()
        talep = grp[TARGET_COL].values
        for i in range(1, n_ay+1):
            grp[f"hedef_{i}"] = np.roll(talep, -i)
            grp[f"hedef_{i}"].iloc[-i:] = np.nan
        parca_dfs.append(grp)
    return pd.concat(parca_dfs).dropna(subset=[f"hedef_{i}" for i in range(1, n_ay+1)])


def segment_modelleri_egit(train_df, test_df, n_trials=30, n_ay=6):
    """
    Her segment için 4 ML modeli eğitir.
    A/B → ML, C/Z → geleneksel.
    Multi-output: 6 aylık tahmin tek seferde.
    """
    # Multi-output hedef sütunları oluştur
    hedef_cols = [f"hedef_{i}" for i in range(1, n_ay+1)]
    train_mo = _multi_hedef_olustur(train_df, n_ay)

    segmentler = train_df["Segment"].dropna().unique() if "Segment" in train_df.columns else ["ALL"]
    sonuclar = {}

    for seg in segmentler:
        # ABC/XYZ kuralı
        abc_s = seg[0] if len(seg) > 0 else "C"
        xyz_s = seg[1] if len(seg) > 1 else "Z"
        kullan_ml = (abc_s in ["A","B"]) and (xyz_s != "Z")

        tr = train_df[train_df["Segment"]==seg] if "Segment" in train_df.columns else train_df
        te = test_df[test_df["Segment"]==seg]   if "Segment" in test_df.columns  else test_df

        if len(tr) < 10:
            continue

        # Test gerçek değerleri (son 6 ay, parça bazlı ortalama)
        y_te_vals = te.groupby(PARCA_COL)[TARGET_COL].apply(list).to_dict()

        if not kullan_ml:
            # Geleneksel yöntem
            print(f"  [{seg}] C/Z sınıfı → Hareketli Ortalama")
            sonuclar[seg] = {
                "tip": "geleneksel",
                "sampiyon": "Hareketli Ort.",
                "feat_cols": [],
                "modeller": {},
                "n_ay": n_ay,
            }
            continue

        # Multi-output train
        tr_mo = train_mo[train_mo["Segment"]==seg] if "Segment" in train_mo.columns else train_mo
        if len(tr_mo) < 20:
            continue

        feat = [c for c in FEATURE_COLS if c in tr_mo.columns]
        X_tr = tr_mo[feat].fillna(0).values
        y_tr = tr_mo[hedef_cols].fillna(0).values

        X_te, y_te_raw, _ = _get_Xy(te)

        modeller  = {}
        metrikler = {}

        # ── RF ─────────────────────────────────────────────────
        print(f"  [{seg}] RF (multi-output)...")
        def rf_obj(trial):
            p = {"n_estimators": trial.suggest_int("n_estimators",50,200),
                 "max_depth":    trial.suggest_int("max_depth",3,12),
                 "max_features": trial.suggest_float("max_features",0.3,1.0),
                 "random_state":42,"n_jobs":-1}
            m = MultiOutputRegressor(RandomForestRegressor(**p))
            return _tscv_rmse(RandomForestRegressor(**p), X_tr, y_tr[:,0])

        s = optuna.create_study(direction="minimize",sampler=optuna.samplers.TPESampler(seed=42))
        s.optimize(rf_obj, n_trials=n_trials, show_progress_bar=False)
        p = {**s.best_params,"random_state":42,"n_jobs":-1}
        rf_m = MultiOutputRegressor(RandomForestRegressor(**p))
        rf_m.fit(X_tr, y_tr)
        rf_pred = np.maximum(rf_m.predict(X_te), 0)
        modeller["RF"] = rf_m
        metrikler["RF"] = metrik_hesapla(y_te_raw, rf_pred[:,0], "RF") if rf_pred.ndim>1 else metrik_hesapla(y_te_raw, rf_pred, "RF")

        # ── XGBoost ────────────────────────────────────────────
        print(f"  [{seg}] XGBoost (multi-output)...")
        def xgb_obj(trial):
            p = {"n_estimators":  trial.suggest_int("n_estimators",50,300),
                 "max_depth":     trial.suggest_int("max_depth",2,8),
                 "learning_rate": trial.suggest_float("learning_rate",0.01,0.3,log=True),
                 "subsample":     trial.suggest_float("subsample",0.5,1.0),
                 "random_state":42}
            base = xgb.XGBRegressor(**p,verbosity=0,n_jobs=-1) if XGB_OK else GradientBoostingRegressor(**{k:v for k,v in p.items() if k!="subsample"})
            return _tscv_rmse(base, X_tr, y_tr[:,0])

        s = optuna.create_study(direction="minimize",sampler=optuna.samplers.TPESampler(seed=42))
        s.optimize(xgb_obj, n_trials=n_trials, show_progress_bar=False)
        if XGB_OK:
            base = xgb.XGBRegressor(**s.best_params,random_state=42,verbosity=0,n_jobs=-1)
        else:
            p2 = {k:v for k,v in s.best_params.items() if k in ["n_estimators","max_depth","learning_rate"]}
            base = GradientBoostingRegressor(**p2,random_state=42)
        xgb_m = MultiOutputRegressor(base)
        xgb_m.fit(X_tr, y_tr)
        xgb_pred = np.maximum(xgb_m.predict(X_te), 0)
        modeller["XGBoost"] = xgb_m
        metrikler["XGBoost"] = metrik_hesapla(y_te_raw, xgb_pred[:,0], "XGBoost") if xgb_pred.ndim>1 else metrik_hesapla(y_te_raw, xgb_pred, "XGBoost")

        # ── LightGBM ───────────────────────────────────────────
        print(f"  [{seg}] LightGBM (multi-output)...")
        def lgb_obj(trial):
            p = {"n_estimators":  trial.suggest_int("n_estimators",50,300),
                 "max_depth":     trial.suggest_int("max_depth",2,10),
                 "learning_rate": trial.suggest_float("learning_rate",0.01,0.3,log=True),
                 "num_leaves":    trial.suggest_int("num_leaves",15,63),
                 "random_state":42}
            base = lgb.LGBMRegressor(**p,verbose=-1,n_jobs=-1) if LGB_OK else GradientBoostingRegressor(**{k:v for k,v in p.items() if k not in ["num_leaves"]})
            return _tscv_rmse(base, X_tr, y_tr[:,0])

        s = optuna.create_study(direction="minimize",sampler=optuna.samplers.TPESampler(seed=42))
        s.optimize(lgb_obj, n_trials=n_trials, show_progress_bar=False)
        if LGB_OK:
            base = lgb.LGBMRegressor(**s.best_params,random_state=42,verbose=-1,n_jobs=-1)
        else:
            p2 = {k:v for k,v in s.best_params.items() if k in ["n_estimators","max_depth","learning_rate"]}
            base = GradientBoostingRegressor(**p2,random_state=42)
        lgb_m = MultiOutputRegressor(base)
        lgb_m.fit(X_tr, y_tr)
        lgb_pred = np.maximum(lgb_m.predict(X_te), 0)
        modeller["LightGBM"] = lgb_m
        metrikler["LightGBM"] = metrik_hesapla(y_te_raw, lgb_pred[:,0], "LightGBM") if lgb_pred.ndim>1 else metrik_hesapla(y_te_raw, lgb_pred, "LightGBM")

        # ── CatBoost ───────────────────────────────────────────
        print(f"  [{seg}] CatBoost (multi-output)...")
        def cat_obj(trial):
            p = {"iterations":    trial.suggest_int("iterations",50,300),
                 "depth":         trial.suggest_int("depth",2,8),
                 "learning_rate": trial.suggest_float("learning_rate",0.01,0.3,log=True),
                 "random_seed":42}
            base = CatBoostRegressor(**p,verbose=0) if CAT_OK else GradientBoostingRegressor(n_estimators=p["iterations"],max_depth=p["depth"],learning_rate=p["learning_rate"],random_state=42)
            return _tscv_rmse(base, X_tr, y_tr[:,0])

        s = optuna.create_study(direction="minimize",sampler=optuna.samplers.TPESampler(seed=42))
        s.optimize(cat_obj, n_trials=n_trials, show_progress_bar=False)
        if CAT_OK:
            base = CatBoostRegressor(**s.best_params,random_seed=42,verbose=0)
        else:
            p2 = {k:v for k,v in s.best_params.items() if k in ["iterations","depth","learning_rate"]}
            base = GradientBoostingRegressor(n_estimators=p2.get("iterations",100),max_depth=p2.get("depth",3),learning_rate=p2.get("learning_rate",0.1),random_state=42)
        cat_m = MultiOutputRegressor(base)
        cat_m.fit(X_tr, y_tr)
        cat_pred = np.maximum(cat_m.predict(X_te), 0)
        modeller["CatBoost"] = cat_m
        metrikler["CatBoost"] = metrik_hesapla(y_te_raw, cat_pred[:,0], "CatBoost") if cat_pred.ndim>1 else metrik_hesapla(y_te_raw, cat_pred, "CatBoost")

        # Şampiyon (en düşük RMSE)
        sampiyon = min(metrikler, key=lambda k: metrikler[k]["RMSE"])
        rmse_str = " | ".join([f"{k}={metrikler[k]['RMSE']:,.0f}" for k in metrikler])
        print(f"  [{seg}] {rmse_str} → Şampiyon: {sampiyon}")

        sonuclar[seg] = {
            "tip":       "ml",
            "modeller":  modeller,
            "metrikler": metrikler,
            "sampiyon":  sampiyon,
            "feat_cols": feat,
            "hedef_cols":hedef_cols,
            "n_ay":      n_ay,
        }

    return sonuclar


def parca_tahmin(parca_kodu, ml_df, seg_modelleri, n_ay=6):
    """
    Tek parça tahmini.
    A/B → ML (multi-output), C/Z → Geleneksel.
    Test seti üzerinde tüm modeller + geleneksel karşılaştırılır.
    """
    from m1_veri import parca_verisi

    pv  = parca_verisi(ml_df, parca_kodu)
    abc = pv["abc"]
    xyz = pv["xyz"]
    seg = str(pv["train"]["Segment"].iloc[0]) if "Segment" in pv["train"].columns else "CZ"
    kullan_ml = pv["kullan_ml"]

    ts_train   = pv["ts_train"]
    ts_test    = pv["ts_test"]
    test_df    = pv["test"]

    if seg not in seg_modelleri:
        seg = list(seg_modelleri.keys())[0]

    seg_res   = seg_modelleri[seg]
    feat_cols = seg_res["feat_cols"]
    hedef_cols= seg_res.get("hedef_cols",[f"hedef_{i}" for i in range(1,n_ay+1)])

    # Geleneksel tahminler (her zaman hesapla - karşılaştırma için)
    gel = geleneksel_tahmin(ts_train, n_tahmin=max(len(ts_test), n_ay))
    gel_metrikler = {
        "Hareketli Ort.": metrik_hesapla(ts_test, gel["hareketli_ort"][:len(ts_test)], "Hareketli Ort."),
        "Üstel Düzeltme": metrik_hesapla(ts_test, gel["ustel"][:len(ts_test)],          "Üstel Düzeltme"),
        "Naif":           metrik_hesapla(ts_test, gel["naif"][:len(ts_test)],            "Naif"),
    }

    if not kullan_ml or seg_res["tip"] == "geleneksel":
        # C/Z → Hareketli Ortalama kullan
        tahminler    = gel["hareketli_ort"][:n_ay]
        ml_metrikler = {}
        sampiyon     = "Hareketli Ort."
        tum_ml_pred  = {}
        y_pred_test  = gel["hareketli_ort"][:len(ts_test)]
    else:
        # A/B → ML (multi-output)
        X_te = test_df[[c for c in feat_cols if c in test_df.columns]].fillna(0).values
        if X_te.shape[1] < len(feat_cols):
            full = np.zeros((len(test_df), len(feat_cols)))
            for i, c in enumerate(feat_cols):
                if c in test_df.columns:
                    full[:,i] = test_df[c].fillna(0).values
            X_te = full

        ml_metrikler = {}
        tum_ml_pred  = {}
        sampiyon     = seg_res["sampiyon"]

        for m_adi, m_obj in seg_res["modeller"].items():
            try:
                pred = np.maximum(m_obj.predict(X_te), 0)
                # Multi-output → ilk sütun t+1 tahmini
                pred_t1 = pred[:,0] if pred.ndim > 1 else pred
                ml_metrikler[m_adi] = metrik_hesapla(ts_test, pred_t1, m_adi)
                tum_ml_pred[m_adi]  = pred_t1.tolist()
            except Exception as e:
                print(f"  [!] {m_adi} tahmin hatası: {e}")

        # Gelecek n_ay tahmini (şampiyon modelden, son test satırı)
        samp_model = seg_res["modeller"].get(sampiyon)
        if samp_model is not None:
            son_satir = test_df.iloc[-1:].copy()
            X_son = son_satir[[c for c in feat_cols if c in son_satir.columns]].fillna(0).values
            if X_son.shape[1] < len(feat_cols):
                full = np.zeros((1, len(feat_cols)))
                for i, c in enumerate(feat_cols):
                    if c in son_satir.columns:
                        full[0,i] = float(son_satir[c].fillna(0).values[0])
                X_son = full
            pred_gelecek = np.maximum(samp_model.predict(X_son), 0)
            # Multi-output → n_ay tahmin direkt çıkıyor
            if pred_gelecek.ndim > 1 and pred_gelecek.shape[1] >= n_ay:
                tahminler = pred_gelecek[0, :n_ay].tolist()
            else:
                tahminler = gel["hareketli_ort"][:n_ay]
        else:
            tahminler = gel["hareketli_ort"][:n_ay]

        y_pred_test = tum_ml_pred.get(sampiyon, gel["hareketli_ort"][:len(ts_test)])

    return {
        "tahminler":     tahminler,
        "y_test":        ts_test.tolist(),
        "y_pred_test":   y_pred_test if isinstance(y_pred_test, list) else list(y_pred_test),
        "ts_train":      ts_train.tolist(),
        "sampiyon":      sampiyon,
        "segment":       seg,
        "abc":           abc,
        "xyz":           xyz,
        "kullan_ml":     kullan_ml,
        "ml_metrikler":  ml_metrikler,
        "gel_metrikler": gel_metrikler,
        "tum_ml_pred":   tum_ml_pred,
    }


def batch_tahmin(ml_df, seg_modelleri, parcalar, n_ay=6):
    rows = []
    for pid in parcalar:
        try:
            res = parca_tahmin(pid, ml_df, seg_modelleri, n_ay)
            rec = {"Parça_Kodu": pid, "Sampiyon": res["sampiyon"],
                   "Segment": res["segment"], "ABC": res["abc"], "XYZ": res["xyz"],
                   "Kullan_ML": res["kullan_ml"]}
            for i, t in enumerate(res["tahminler"],1):
                rec[f"Tahmin_Ay_{i}"] = round(t,1)
            # En iyi ML metriği
            tum_met = {**res["ml_metrikler"], **res["gel_metrikler"]}
            samp_met = tum_met.get(res["sampiyon"], {})
            rec["MAE"]  = round(samp_met.get("MAE",0),2)
            rec["RMSE"] = round(samp_met.get("RMSE",0),2)
            mape = samp_met.get("MAPE",np.nan)
            rec["MAPE"] = round(mape,2) if not np.isnan(mape) else None
            rows.append(rec)
        except Exception as e:
            print(f"  [!] {pid}: {e}")
    return pd.DataFrame(rows)
