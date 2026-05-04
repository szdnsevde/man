"""
ENM412 – MAN Türkiye A.Ş.
Modül 3 – Grid Search + Optuna + SimPy + EOQ vs ML Karşılaştırması
Yazarlar: Büşra ÇİL · İrem ÇELİK · Sevde SÖZDEN

Metodoloji:
    1. Grid Search  → geniş aralıkta maliyet haritası
    2. Optuna (TPE) → hassas Q*, r* optimizasyonu
    3. SimPy        → Q*, r* ile simülasyon doğrulaması
    4. EOQ klasik   → aynı maliyet parametreleriyle karşılaştırma
    
Maliyet katsayıları (h, p, S) sabittir - Excel'den gelir, değişmez.
Karşılaştırma: EOQ (statik μ) vs ML+Optuna (tahmin μ)
"""
import numpy as np
import pandas as pd
import optuna
import simpy
from scipy.stats import norm
import warnings
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")


def lt_parametreleri(mu, sigma, LT_ay):
    """Teslim süresi boyunca talep parametreleri. LT belirsizliği %20."""
    mu_L    = mu * LT_ay
    sigma_L = np.sqrt(sigma**2 * LT_ay + (mu * LT_ay * 0.20)**2)
    return float(mu_L), float(max(sigma_L, 1e-6))


def tc_hesapla(Q, r, mu, mu_L, sigma_L, h, p, S):
    """
    Analitik toplam maliyet (TL/ay).
    h, p, S SABİTTİR — sadece Q ve r değişir.
    """
    Q = max(float(Q), 1.0)
    z    = (r - mu_L) / sigma_L if sigma_L > 0 else 0
    E_so = max(sigma_L * (norm.pdf(z) - z*(1-norm.cdf(z))), 0.0)
    SS   = max(0.0, r - mu_L)
    et   = h * (Q/2.0 + SS)          # Elde tutma
    si   = S * (mu / Q)              # Sipariş (sabit S, değişen sıklık)
    sk   = p * (mu / Q) * E_so       # Stoksuz kalma
    return {"elde_tutma": et, "siparis": si, "stoksuz": sk, "toplam": et+si+sk}


# ══════════════════════════════════════════════════════════════════
# ADIM 1 – GRID SEARCH
# ══════════════════════════════════════════════════════════════════

def grid_search(mu, sigma, LT_ay, h, p, S, Q_ref, grid_adim=15):
    mu_L, sigma_L = lt_parametreleri(mu, sigma, LT_ay)
    Q_ref = max(Q_ref, 1.0)
    Q_grid = np.unique(np.linspace(max(1., Q_ref*0.3), Q_ref*4, grid_adim).astype(int))
    r_grid = np.unique(np.linspace(max(0., mu_L-sigma_L), mu_L+3*sigma_L, grid_adim).astype(int))
    best_tc, best_Q, best_r = np.inf, Q_ref, mu_L+1.65*sigma_L
    for Qv in Q_grid:
        for rv in r_grid:
            tc = tc_hesapla(Qv, rv, mu, mu_L, sigma_L, h, p, S)["toplam"]
            if tc < best_tc:
                best_tc, best_Q, best_r = tc, float(Qv), float(rv)
    return best_Q, best_r, mu_L, sigma_L


# ══════════════════════════════════════════════════════════════════
# ADIM 2 – OPTUNA
# ══════════════════════════════════════════════════════════════════

def optuna_optimize(mu, sigma, LT_ay, h, p, S, best_Qg, best_rg, n_trials=50, MOQ=1):
    mu_L, sigma_L = lt_parametreleri(mu, sigma, LT_ay)
    Q_lb = max(float(MOQ), best_Qg*0.4)
    Q_ub = max(best_Qg*2.5, Q_lb+1.)
    r_lb = max(0., best_rg - sigma_L)
    r_ub = best_rg + sigma_L

    def obj(t):
        Q = t.suggest_float("Q", Q_lb, Q_ub)
        r = t.suggest_float("r", r_lb, r_ub)
        return tc_hesapla(Q, r, mu, mu_L, sigma_L, h, p, S)["toplam"]

    s = optuna.create_study(direction="minimize",
                            sampler=optuna.samplers.TPESampler(seed=42))
    s.optimize(obj, n_trials=n_trials, show_progress_bar=False)

    opt_Q  = max(int(round(s.best_params["Q"])), MOQ)
    opt_r  = max(0, int(round(s.best_params["r"])))
    opt_SS = max(0, int(round(1.65*sigma_L)))
    hizmet_analitik = float(norm.cdf((opt_r-mu_L)/sigma_L)) if sigma_L > 0 else 1.
    komp   = tc_hesapla(opt_Q, opt_r, mu, mu_L, sigma_L, h, p, S)
    return {"optimal_Q": opt_Q, "optimal_r": opt_r, "optimal_SS": opt_SS,
            "hizmet_analitik": hizmet_analitik,
            "analitik_maliyet": komp["toplam"],
            "et_analitik": komp["elde_tutma"],
            "si_analitik": komp["siparis"],
            "sk_analitik": komp["stoksuz"]}


# ══════════════════════════════════════════════════════════════════
# ADIM 3 – SİMPY DOĞRULAMA
# ══════════════════════════════════════════════════════════════════

class _Stok:
    def __init__(self, env, Q, r, LT_ay, h, p, S, talep, MOQ=1):
        self.env=env; self.Q=max(int(Q),MOQ); self.r=int(r)
        self.LT_ay=LT_ay; self.h=h; self.p=p; self.S=S; self.talep=talep
        self.stok=self.r+self.Q; self.yolda=0
        self.et=self.si=self.sk=0.; self.kar=self.tot=0; self.ay=0
        env.process(self._talep()); env.process(self._kontrol())

    def _talep(self):
        for t in self.talep:
            yield self.env.timeout(1)
            t=max(0,round(t)); self.tot+=t
            if self.stok>=t: self.stok-=t; self.kar+=t
            else:
                self.kar+=self.stok; self.sk+=(t-self.stok)*self.p; self.stok=0
            self.et+=self.stok*self.h; self.ay+=1

    def _kontrol(self):
        while True:
            yield self.env.timeout(0.01)
            if (self.stok+self.yolda)<=self.r and self.yolda==0:
                self.env.process(self._siparis())

    def _siparis(self):
        self.si+=self.S; self.yolda+=self.Q
        lt=max(0.1, np.random.normal(self.LT_ay, self.LT_ay*0.20))
        yield self.env.timeout(lt)
        self.stok+=self.Q; self.yolda-=self.Q


def simpy_dogrula(Q, r, mu, LT_ay, h, p, S, talep_serisi, n_rep=20, seed=42):
    rng = np.random.default_rng(seed)
    ts  = list(talep_serisi) if talep_serisi else [mu]*12
    while len(ts) < 24: ts = ts + ts
    ts = ts[:36]

    maliyetler=[]; hizmetler=[]; et_l=[]; si_l=[]; sk_l=[]
    for _ in range(n_rep):
        g   = rng.normal(1.0, 0.05, len(ts))
        t_r = [max(0, t*gi) for t,gi in zip(ts,g)]
        env = simpy.Environment()
        sim = _Stok(env, Q=Q, r=r, LT_ay=LT_ay, h=h, p=p, S=S, talep=t_r)
        env.run(until=len(t_r)+1)
        ay  = max(sim.ay, 1)
        maliyetler.append((sim.et+sim.si+sim.sk)/ay)
        hizmetler.append(sim.kar/max(sim.tot,1))
        et_l.append(sim.et/ay); si_l.append(sim.si/ay); sk_l.append(sim.sk/ay)

    return {
        "sim_maliyet": float(np.mean(maliyetler)),
        "sim_hizmet":  float(np.mean(hizmetler)),
        "sim_et":      float(np.mean(et_l)),
        "sim_si":      float(np.mean(si_l)),
        "sim_sk":      float(np.mean(sk_l)),
        "sim_std":     float(np.std(maliyetler)),
    }


# ══════════════════════════════════════════════════════════════════
# ANA FONKSİYON
# ══════════════════════════════════════════════════════════════════

def parca_optimize(parca_kodu, opt_df, abc_df, tahmin_listesi,
                   grid_adim=12, n_trials=50, n_rep=20):
    """
    EOQ (statik μ) vs ML+Optuna (tahmin μ) karşılaştırması.
    
    - h, p, S SABİT (Excel'den)
    - EOQ: geçmiş 30 ay ortalaması (μ_hist)
    - ML:  ML tahmini ortalaması (μ_tahmin)
    - İki sistem aynı formülle hesaplanır, sadece μ farklı
    - SimPy ML+Optuna'nın Q*, r*'ını doğrular
    """
    opt_row = opt_df[opt_df["Parça_Kodu"]==parca_kodu]
    abc_row = abc_df[abc_df["Parça_Kodu"]==parca_kodu]
    if opt_row.empty: raise ValueError(f"{parca_kodu} bulunamadı")

    o   = opt_row.iloc[0]
    lt  = float(o.get("LT_ay", o.get("LT_gun",20)/30))
    # SABİT maliyet katsayıları
    h   = float(o["h"])
    p   = float(o["p"])
    S   = float(o["Siparis_Maliyeti"])

    # Tarihsel μ (30 aylık eğitim) - EOQ için kullanılır
    if not abc_row.empty:
        mu_hist  = float(abc_row.iloc[0]["Ort_Aylik_Talep"])
        sig_hist = float(abc_row.iloc[0]["Std_Sapma"])
    else:
        mu_hist  = float(np.mean(tahmin_listesi)) if tahmin_listesi else 1.
        sig_hist = float(np.std(tahmin_listesi))  if tahmin_listesi else 1.

    # ML tahmini μ - Optuna için kullanılır
    mu_tahmin = float(np.mean([t for t in tahmin_listesi if t>0])) if tahmin_listesi else mu_hist

    # ── EOQ KLASİK (tarihsel μ ile) ─────────────────────────────
    mu_L_h, sigma_L_h = lt_parametreleri(mu_hist, sig_hist, lt)
    Q_eoq  = max(int(round(np.sqrt(2*S*max(mu_hist,1)/max(h,1e-9)))), 1)
    r_eoq  = int(round(mu_L_h + 1.65*sigma_L_h))
    SS_eoq = int(round(1.65*sigma_L_h))
    eoq    = tc_hesapla(Q_eoq, r_eoq, mu_hist, mu_L_h, sigma_L_h, h, p, S)

    # ── ML + OPTUNA (tahmin μ ile) ───────────────────────────────
    mu_L_t, sigma_L_t = lt_parametreleri(mu_tahmin, sig_hist, lt)
    Q_ref = max(np.sqrt(2*S*max(mu_tahmin,1)/max(h,1e-9)), 1.)
    best_Qg, best_rg, _, _ = grid_search(mu_tahmin, sig_hist, lt, h, p, S, Q_ref, grid_adim)
    opt = optuna_optimize(mu_tahmin, sig_hist, lt, h, p, S, best_Qg, best_rg, n_trials)

    # ── SİMPY DOĞRULAMA (ML Q*, r* ile) ─────────────────────────
    sim = simpy_dogrula(
        Q=opt["optimal_Q"], r=opt["optimal_r"],
        mu=mu_tahmin, LT_ay=lt, h=h, p=p, S=S,
        talep_serisi=tahmin_listesi, n_rep=n_rep
    )

    # ── ML ANALİTİK MALİYET (adil karşılaştırma için) ───────────
    # Hem EOQ hem ML aynı analitik formülle hesaplanır
    # SimPy sadece hizmet düzeyi doğrulaması için kullanılır
    ml_analitik = tc_hesapla(opt["optimal_Q"], opt["optimal_r"],
                              mu_tahmin, mu_L_t, sigma_L_t, h, p, S)

    tasarruf_tl   = eoq["toplam"] - ml_analitik["toplam"]
    tasarruf_oran = (tasarruf_tl / eoq["toplam"] * 100) if eoq["toplam"] > 0 else 0.

    return {
        # Optimal politika (ML+Optuna)
        "optimal_Q":   opt["optimal_Q"],
        "optimal_r":   opt["optimal_r"],
        "optimal_SS":  opt["optimal_SS"],
        # SimPy → hizmet düzeyi doğrulaması
        "sim_maliyet": sim["sim_maliyet"],
        "sim_hizmet":  sim["sim_hizmet"],
        "sim_std":     sim["sim_std"],
        "sim_et":      sim["sim_et"],
        "sim_si":      sim["sim_si"],
        "sim_sk":      sim["sim_sk"],
        # Analitik ML maliyeti (EOQ ile adil karşılaştırma)
        "ml_maliyet":  ml_analitik["toplam"],
        "ml_et":       ml_analitik["elde_tutma"],
        "ml_si":       ml_analitik["siparis"],
        "ml_sk":       ml_analitik["stoksuz"],
        # EOQ klasik
        "Q_eoq":   Q_eoq,   "r_eoq":  r_eoq, "SS_eoq": SS_eoq,
        "tc_eoq":  eoq["toplam"],
        "et_eoq":  eoq["elde_tutma"],
        "si_eoq":  eoq["siparis"],
        "sk_eoq":  eoq["stoksuz"],
        # Tasarruf (EOQ analitik vs ML analitik — adil karşılaştırma)
        "tasarruf_tl":   tasarruf_tl,
        "tasarruf_oran": tasarruf_oran,
        # Parametreler
        "mu_hist":    mu_hist,
        "mu_tahmin":  mu_tahmin,
        "sigma_L":    sigma_L_t,
        "mu_L":       mu_L_t,
        "h": h, "p": p, "S": S,
    }


def aksiyon_uyarisi(opt_res, tahminler):
    Q   = opt_res["optimal_Q"]
    r   = opt_res["optimal_r"]
    SS  = opt_res["optimal_SS"]
    hiz = opt_res["sim_hizmet"]
    trend = ((tahminler[-1]-tahminler[0])/max(tahminler[0],1)*100
             if len(tahminler)>=2 else 0.)
    if hiz < 0.85:
        return {"renk":"kirmizi","trend":trend,
                "mesaj":f"🔴 KRİTİK: Hizmet düzeyi düşük (%{hiz*100:.0f}). Acil sipariş önerilir. Q={Q:,} adet verin."}
    elif trend > 20:
        return {"renk":"sari","trend":trend,
                "mesaj":f"🟡 UYARI: Talep %{trend:.0f} artış bekleniyor. Q={Q:,} adet sipariş verin, r={r:,} eşiğini izleyin."}
    elif trend < -20:
        return {"renk":"mavi","trend":trend,
                "mesaj":f"🔵 BİLGİ: Talep %{abs(trend):.0f} azalış bekleniyor. Sipariş öncesi stoku kontrol edin."}
    else:
        return {"renk":"yesil","trend":trend,
                "mesaj":f"🟢 NORMAL: Q={Q:,} adet sipariş verin. r={r:,} eşiğinde sipariş tetikleyin. SS={SS:,} bulundurun."}
