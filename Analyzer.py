#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Audio Quality Contest – Analisi tecnica deterministica con scoring assoluto

Panoramica
Strumento per valutare in modo riproducibile e comparabile la qualità tecnica di file audio.
Assegna uno score assoluto (0–100) con indicatori semaforici (✅/⚠️/❌) e produce un
report dettagliato orientato a esigenze “audiophile”.

Cosa fa
- Scansione di una cartella, supporto PCM/DSD con pre‑conversione FFmpeg (Hi‑Res o CD).
- Deduplica tecnica: raggruppa i file audio‑identici tramite MD5 bit‑exact.
- Misure avanzate: true‑peak/ISP (oversampling), DC offset, noise floor robusto con
  selettore multi‑scala (relaxed/strict/strict+notch), rilevazione HUM 50/60 Hz e notch
  combinato per stimare il broadband, spur, LUFS/LRA, PLR (integrato/attivo/effettivo),
  DR (TT), short‑term IQR, bilanciamento spettrale, ampiezza stereo IQR, transienti,
  diagnostica HF su SR elevati; THD+N/jitter quando sono presenti toni di test.
- Punteggio assoluto: componente H (igiene) e Q (qualità) con pesi adattivi in base
  alla confidenza del rumore; soglie fisse “2.4.1‑audiophile”. Profilo “strict”
  attivabile via AQC_PROFILE=strict per criteri più severi.
- Gestione del rumore e affidabilità: quando la stima del noise floor è inaffidabile
  (contenuto sempre pieno/incoerenza elevata), il valore viene comunque calcolato e
  mostrato a fini diagnostici ma può essere escluso dallo score. Applicato “sanity cap”
  coerente con il livello musicale (tipicamente p90−6/−8/−10 dB con floor), con
  motivazione esplicitata nel log.
- Classifica e tie‑break: ordinamento “hygiene‑first”, raffinamento TP/ISP sui top,
  confronto per bande con allineamento e gain‑match per distinguere versioni dello
  stesso master da semplici differenze di volume.
- Report e azioni: log trasparente con soglie, pesi effettivi e motivazioni; annuncio
  del vincitore; spostamento del file vincitore e dei suoi duplicati MD5; pulizia dei
  temporanei.
- Prestazioni e determinismo: parallelismo opzionale, pin dei thread e variabili
  d’ambiente per esecuzioni riproducibili.

Nota
In assenza di passaggi realmente “vuoti”, la stima del noise floor resta prudente:
per non falsare lo score può essere esclusa, ma viene sempre riportata a titolo diagnostico.

Versione policy: 2.4.1‑audiophile
"""

import soundfile as sf
import numpy as np
import sys
import os
import subprocess
import warnings
import math
import tempfile
import shutil
from scipy.fft import fft
from scipy.stats import pearsonr
import pyloudnorm as pyln
import re
from scipy.signal import butter, sosfilt, hilbert
import datetime

# --- Impostazioni Globali Fisse ---
# (Queste definiscono il comportamento tecnico base)
BLOCK_SIZE_SECONDS = 3.0
RMS_SILENCE_THRESHOLD_DBFS = -50.0
TEMP_FLAC_QUALITY = 5
TEMP_FLAC_SR = 176400
TEMP_FLAC_BITS = 24
CD_EDGE_HZ = 22050.0
# --- Soglie Semaforiche e Punteggio Assoluto ---
# (Queste definiscono la valutazione della qualità - TARABILI)
# Struttura: 'metric_key': {
#   'good': valore eccellente (✅),
#   'warn': valore limite accettabile (⚠️ ),
#   'bad': valore problematico (❌),
#   'higher_is_better': True/False,
#   'weight': peso nello score finale (W)
# }
# Le soglie 'good' e 'bad' definiscono la scala 0-100 (o 1-0).
# 'warn' è usato solo per il simbolo semaforico.

def design_hum_notch_comb(sr, f0, max_harm=12):
    sos_list = []
    nyq = 0.5 * sr
    from scipy.signal import iirnotch, tf2sos
    for h in range(1, max_harm + 1):
        fk = f0 * h
        if fk >= nyq * 0.98:
            break
        if h <= 3:
            Q = 80.0
        elif h <= 8:
            Q = 50.0
        else:
            Q = 35.0
        b, a = iirnotch(w0=fk, Q=Q, fs=sr)
        sos = tf2sos(b, a)
        sos_list.append(sos)
    return sos_list

def apply_sos_chain_filtfilt(x, sos_chain):
    y = np.asarray(x, dtype=np.float64)
    if sos_chain is None:
        return y
    if isinstance(sos_chain, (list, tuple)):
        for sos in sos_chain:
            if sos is None:
                continue
            try:
                arr = np.asarray(sos, dtype=np.float64)
                if arr.ndim == 1 and arr.size == 6:
                    arr = arr.reshape(1, 6)
                if arr.ndim != 2 or arr.shape[1] != 6 or arr.shape[0] < 1:
                    continue
                try:
                    y = sosfiltfilt(arr, y)
                except Exception:
                    y = sosfilt(arr, y)
            except Exception:
                continue
        return y
    else:
        arr = np.asarray(sos_chain, dtype=np.float64)
        if arr.ndim == 1 and arr.size == 6:
            arr = arr.reshape(1, 6)
        if arr.ndim != 2 or arr.shape[1] != 6 or arr.shape[0] < 1:
            return y
        try:
            return sosfiltfilt(arr, y)
        except Exception:
            try:
                return sosfilt(arr, y)
            except Exception:
                return y

def automaster_process(filepath, verbose=True):
    try:
        data, sr = sf.read(filepath, dtype="float64", always_2d=True)
    except Exception as e:
        raise RuntimeError(f"Impossibile leggere '{filepath}': {e}")
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    L = data[:, 0]
    R = data[:, 1]
    sr_work = choose_work_sr(sr)
    if verbose:
        fam = classify_family(sr)
        print(f"[AutoMaster] SR in: {sr} Hz (famiglia {fam}) → SR lavoro: {sr_work} Hz")
    y_in = np.column_stack([L, R])
    y_work, _ = resample_to(sr_work, y_in, sr)
    Lw = y_work[:, 0]
    Rw = y_work[:, 1]
    sos_hp = highpass_dc_sos(sr_work, fc=20.0, order=2)
    Lw = apply_sos_chain_filtfilt(Lw, [sos_hp])
    Rw = apply_sos_chain_filtfilt(Rw, [sos_hp])
    hum_f0, hum_score = detect_hum_f0_and_strength(0.5*(Lw+Rw), sr_work)
    if hum_f0 is not None and hum_score >= 4.0:
        sos_notches = design_hum_notch_comb(sr_work, hum_f0, max_harm=12)
        Lw = apply_sos_chain_filtfilt(Lw, sos_notches)
        Rw = apply_sos_chain_filtfilt(Rw, sos_notches)
    Lw, Rw = gentle_tilt_if_needed(Lw, Rw, sr_work)
    Lw, Rw = ms_width_adjust(Lw, Rw, target_db_range=(-10.0, +6.0), max_gain_change=0.10)
    y_work2 = limiter_truepeak_stereo(np.column_stack([Lw, Rw]), sr_work, tp_target_db=-1.0)
    prog_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_hr = os.path.join(prog_dir, f"automasterHR_{sr_work//1000}k_{ts}.wav")
    sf.write(out_hr, np.asarray(y_work2, dtype=np.float64), sr_work, subtype="PCM_24", format="WAV")
    if verbose:
        tp_hr = true_peak_estimate_stereo(y_work2, sr_work, os_factor=8)
        print(f"Salvato Hi‑SR: {out_hr}  (TP≈{tp_hr:.2f} dBTP)")
    y_441, _ = downsample_to_44100(y_work2, sr_work)
    x16 = dither_tpdf_16bit(y_441)
    out_cd = os.path.join(prog_dir, f"automasterCD_{ts}.wav")
    sf.write(out_cd, x16, 44100, subtype="PCM_16", format="WAV")
    if verbose:
        x16f = x16.astype(np.float64) / 32768.0
        if x16f.ndim == 1:
            x16f = x16f.reshape(-1, 1)
        tp_cd = true_peak_estimate_stereo(x16f, 44100, os_factor=8)
        print(f"Salvato CD:   {out_cd}  (44.1 kHz / 16 bit, TP≈{tp_cd:.2f} dBTP)")
    return out_hr, out_cd

def get_quality_policy():
    policy = {
        "version": "2.4.1-audiophile",
        "thresholds": {
            "isp_margin_db":        {"good": 1.0,  "warn": 0.5,  "bad": 0.1,  "higher_is_better": True},
            "dc_offset_dbfs":       {"good": -80.0,"warn": -70.0,"bad": -60.0,"higher_is_better": False},
            "noise_spur_db":        {"good": 6.0,  "warn": 12.0, "bad": 18.0, "higher_is_better": False},
            "noise_floor_dbfs_16":  {"good": -90.0,"warn": -88.0,"bad": -86.0,"higher_is_better": False},
            "noise_floor_dbfs_24":  {"good": -105.0,"warn": -100.0,"bad": -98.0,"higher_is_better": False},
            "dr_tt_avg":            {"good": 14.0, "warn": 8.0,  "bad": 6.0,  "higher_is_better": True},
            "plr_est":              {"good": 12.0, "warn": 9.0,  "bad": 6.0,  "higher_is_better": True},
            "lra_est":              {"good": 12.0, "warn": 6.0,  "bad": 4.0,  "higher_is_better": True},
            "st_lufs_iqr_db":       {"good": 6.0,  "warn": 4.0,  "bad": 2.0,  "higher_is_better": True},
            "spectral_balance_dev_db": {"good": 2.0, "warn": 4.0, "bad": 6.0, "higher_is_better": False},
            "stereo_width_iqr_db":  {"good": 6.0,  "warn": 3.0,  "bad": 1.0,  "higher_is_better": True},
            "jitter_ppm":           {"good": 50.0, "warn": 150.0,"bad": 250.0,"higher_is_better": False},
            "hf_rms_var_db":        {"good": 1.0,  "warn": 2.5,  "bad": 4.0,  "higher_is_better": False},
            "stereo_correlation":   {"good": 0.90, "warn": 0.95, "bad": 1.00, "higher_is_better": False},
        },
        "weights": {
            "H": {
                "noise_floor_dbfs": 0.40,
                "noise_spur_db":    0.30,
                "isp_margin_db":    0.20,
                "dc_offset_dbfs":   0.10,
            },
            "Q": {
                "dr_tt_avg":              0.30,
                "plr_est":                0.20,
                "lra_est":                0.15,
                "st_lufs_iqr_db":         0.15,
                "spectral_balance_dev_db":0.15,
                "stereo_width_iqr_db":    0.04,
            }
        },
        "score_mix": {"H": 0.80, "Q": 0.20},
        "clip_classes": {
            "hard_tp_over": 0.10
        },
        "confidence_weights": {
            "Alta":  1.00,
            "Media": 0.70,
            "Bassa": 0.40
        }
    }
    return policy

# --- Utility di calcolo ---
def dbfs(value):
    if value is None:
        return -np.inf
    try:
        v = float(value)
    except (TypeError, ValueError):
        return -np.inf
    if not np.isfinite(v):
        return -np.inf
    v = abs(v)
    if v <= 1e-20:
        return -np.inf
    return 20.0 * math.log10(v)

def _true_peak_internal(audio, sr, os_factor=8):
    if audio is None or sr is None or sr <= 0:
        return None
    if not isinstance(audio, np.ndarray) or audio.size == 0:
        return -np.inf
    x = audio.astype(np.float64, copy=False)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    n = x.shape[0]
    chs = x.shape[1]
    try:
        from scipy.signal import resample_poly
        use_sp = True
    except Exception:
        use_sp = False
    osf = int(max(16, os_factor if isinstance(os_factor, (int, float)) and os_factor > 0 else 16))
    abs_samp = np.max(np.abs(x), axis=1) if chs > 1 else np.abs(x[:, 0])
    dur_s = n / float(sr)
    K = int(max(16, min(2000, math.ceil(dur_s * 4.0))))
    if abs_samp.size <= K:
        cand_idx = np.arange(abs_samp.size, dtype=int)
    else:
        part = np.argpartition(abs_samp, -K)[-K:]
        cand_idx = np.sort(part)
    win_half = int(max(1, round(0.010 * sr)))
    starts = np.maximum(0, cand_idx - win_half)
    ends = np.minimum(n, cand_idx + win_half)
    intervals = []
    if starts.size:
        s0 = int(starts[0]); e0 = int(ends[0])
        for s, e in zip(starts[1:], ends[1:]):
            s = int(s); e = int(e)
            if s <= e0:
                e0 = max(e0, e)
            else:
                intervals.append((s0, e0))
                s0, e0 = s, e
        intervals.append((s0, e0))
    peak = 0.0
    if intervals:
        for s, e in intervals:
            seg = x[s:e, :]
            if seg.size == 0:
                continue
            if use_sp:
                try:
                    y = resample_poly(seg, osf, 1, axis=0, padtype='constant', window=('kaiser', 12.0))
                except Exception:
                    y = resample_poly(seg, osf, 1, axis=0, padtype='constant')
                if y.size:
                    p = float(np.max(np.abs(y)))
                    if p > peak:
                        peak = p
            else:
                m = seg.shape[0]
                xp = np.arange(m, dtype=np.float64)
                xos = np.linspace(0.0, m - 1.0, m * osf, endpoint=False, dtype=np.float64)
                if chs == 1:
                    seg_os = np.interp(xos, xp, seg[:, 0])
                    if seg_os.size:
                        p = float(np.max(np.abs(seg_os)))
                        if p > peak:
                            peak = p
                else:
                    for c in range(chs):
                        ch_os = np.interp(xos, xp, seg[:, c])
                        if ch_os.size:
                            p = float(np.max(np.abs(ch_os)))
                            if p > peak:
                                peak = p
        if peak > 0:
            return dbfs(peak)
    try:
        step_in = 262144
        overlap_in = max(4096, int(0.02 * sr))
        i = 0
        while i < n:
            start = i
            end = min(n, i + step_in)
            if start > 0:
                start = max(0, start - overlap_in)
            seg = x[start:end]
            if seg.size == 0:
                if i == 0:
                    i += step_in
                else:
                    i += step_in - overlap_in
                continue
            if use_sp:
                try:
                    seg_os = resample_poly(seg, osf, 1, axis=0, padtype='constant', window=('kaiser', 12.0))
                except Exception:
                    seg_os = resample_poly(seg, osf, 1, axis=0, padtype='constant')
                if seg_os.size:
                    p = float(np.max(np.abs(seg_os)))
                    if p > peak:
                        peak = p
            else:
                xp = np.arange(seg.shape[0], dtype=np.float64)
                xos = np.linspace(0.0, seg.shape[0] - 1.0, seg.shape[0] * osf, endpoint=False, dtype=np.float64)
                if chs == 1:
                    seg_os = np.interp(xos, xp, seg[:, 0])
                    if seg_os.size:
                        p = float(np.max(np.abs(seg_os)))
                        if p > peak:
                            peak = p
                else:
                    for c in range(chs):
                        ch_os = np.interp(xos, xp, seg[:, c])
                        if ch_os.size:
                            p = float(np.max(np.abs(ch_os)))
                            if p > peak:
                                peak = p
            if i == 0:
                i += step_in
            else:
                i += step_in - overlap_in
    except Exception:
        pass
    return dbfs(peak) if peak > 0 else -np.inf

def calculate_rms(samples):
    if samples is None:
        return 0.0
    x = np.asarray(samples, dtype=np.float64)
    if x.size == 0:
        return 0.0
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    s = float(np.max(np.abs(x)))
    if s <= 0.0 or not np.isfinite(s):
        return 0.0
    y = x / s
    sumsq = float(np.dot(y, y))
    if not np.isfinite(sumsq) or y.size == 0:
        return 0.0
    mean_sq = sumsq / float(y.size)
    if mean_sq <= 0.0 or not np.isfinite(mean_sq):
        return 0.0
    return float(s * math.sqrt(mean_sq))

def estimate_effective_bits(signal_float: np.ndarray, nominal_bits: int) -> int:
    if signal_float is None or len(signal_float) == 0 or nominal_bits is None or nominal_bits <= 0:
        return 0
    sig = np.clip(np.asarray(signal_float, dtype=np.float64), -1.0, 1.0)
    max_int = (1 << (nominal_bits - 1)) - 1
    ints = np.round(sig * max_int).astype(np.int64, copy=False)
    if ints.size == 0:
        return 0
    vals = np.abs(ints)
    if not np.any(vals):
        return 0
    n = vals.size
    step = 131072
    eff_list = []
    for i in range(0, n, step):
        seg = vals[i:i + step]
        if seg.size == 0:
            continue
        or_all = int(np.bitwise_or.reduce(seg))
        if or_all == 0:
            eff_win = 0
        else:
            t = 0
            x = or_all
            while (x & 1) == 0 and t < nominal_bits:
                t += 1
                x >>= 1
            eff_win = nominal_bits - t
            if eff_win < 0:
                eff_win = 0
            if eff_win > nominal_bits:
                eff_win = nominal_bits
            if eff_win > 0:
                idx_bit = eff_win - 1
                mask = (1 << idx_bit)
                count_ones = int(np.count_nonzero((seg & mask) != 0))
                thr_min = max(32, int(0.001 * seg.size))
                if count_ones < thr_min:
                    while eff_win > 0:
                        eff_win -= 1
                        if eff_win == 0:
                            break
                        idx_bit = eff_win - 1
                        mask = (1 << idx_bit)
                        count_ones = int(np.count_nonzero((seg & mask) != 0))
                        if count_ones >= thr_min:
                            break
        eff_list.append(eff_win)
    if not eff_list:
        return 0
    eff = int(np.median(np.array(eff_list, dtype=float)))
    if eff < 0:
        eff = 0
    if eff > nominal_bits:
        eff = nominal_bits
    return eff

def tt_bandpass_filter(samples: np.ndarray, sr: int) -> np.ndarray:
    if samples is None or sr is None or sr <= 0:
        return np.array([], dtype=float)
    x = np.asarray(samples, dtype=np.float64)
    if x.size == 0:
        return np.array([], dtype=float)
    nyq = sr * 0.5
    low_hz = 20.0
    high_hz = min(20000.0, nyq * 0.98)
    if high_hz <= low_hz or nyq <= 0:
        return x
    low = low_hz / nyq
    high = high_hz / nyq
    low = min(max(low, 1e-7), 0.99999)
    high = min(max(high, low + 1e-6), 0.999999)
    if low >= high:
        return x
    try:
        from scipy.signal import butter, sosfiltfilt, sosfilt
    except Exception:
        return x
    try:
        if not hasattr(tt_bandpass_filter, "_sos_cache"):
            tt_bandpass_filter._sos_cache = {}
        key = (int(sr), round(low, 7), round(high, 7))
        sos = tt_bandpass_filter._sos_cache.get(key)
        if sos is None:
            sos = butter(4, [low, high], btype='band', output='sos')
            tt_bandpass_filter._sos_cache[key] = sos
        try:
            y = sosfiltfilt(sos, x)
        except Exception:
            y = sosfilt(sos, x)
        if not np.all(np.isfinite(y)):
            return x
        return y.astype(np.float64, copy=False)
    except Exception:
        return x

def _safe_float(v):
    try:
        f = float(v)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None

def _estimate_noise_floor(samples: np.ndarray, sr: int, win_ms: float = 100.0, percentile: float = 5.0) -> float | None:
    if samples is None or sr is None or sr <= 0:
        return None
    if not isinstance(samples, np.ndarray) or samples.size == 0:
        return None
    win_len = max(1, int(sr * win_ms / 1000.0))
    if samples.size < win_len:
        rms = calculate_rms(samples.astype(np.float64, copy=False))
        if rms <= 0:
            return None
        val = 20.0 * np.log10(rms)
        return float(val) if np.isfinite(val) else None
    trimmed_len = (samples.size // win_len) * win_len
    if trimmed_len <= 0:
        return None
    trimmed = samples[:trimmed_len].astype(np.float64, copy=False)
    frames = trimmed.reshape(-1, win_len)
    vals = []
    try:
        from scipy.signal import welch
        for seg in frames:
            if not np.any(np.isfinite(seg)):
                continue
            try:
                nperseg = min(seg.size, 16384)
                if nperseg < 1024:
                    nperseg = max(256, nperseg)
                noverlap = nperseg // 2
                freqs_w, Pxx = welch(seg, fs=sr, window='hann', nperseg=nperseg, noverlap=noverlap, scaling='spectrum', detrend='constant')
                band = (freqs_w >= 20.0) & (freqs_w <= 20000.0)
                if not np.any(band):
                    rms = calculate_rms(seg)
                    if rms > 0:
                        vals.append(20.0 * np.log10(rms))
                    continue
                power_band = np.sum(Pxx[band])
                nf_rms_w = math.sqrt(power_band) if power_band > 0 else 0.0
                if nf_rms_w > 0:
                    vals.append(20.0 * np.log10(nf_rms_w))
            except Exception:
                rms = calculate_rms(seg)
                if rms > 0:
                    vals.append(20.0 * np.log10(rms))
    except Exception:
        for seg in frames:
            rms = calculate_rms(seg)
            if rms > 0:
                vals.append(20.0 * np.log10(rms))
    vals = np.array([v for v in vals if np.isfinite(v)], dtype=float)
    if vals.size == 0:
        return None
    p = float(np.percentile(vals, percentile))
    return p if np.isfinite(p) else None

def _spectral_usage(sr, bw):
    sr_f, bw_f = _safe_float(sr), _safe_float(bw)
    if sr_f and bw_f and sr_f > CD_EDGE_HZ:
        nyq   = sr_f * 0.5
        extra = nyq - CD_EDGE_HZ          # banda oltre il CD
        if extra <= 0:
            return 0.0                    # niente parte “extra”, nessun bonus
        used  = max(0.0, bw_f - CD_EDGE_HZ)
        return min(1.0, used / extra)     # 0–1 solo sulla parte extra
    return 0.0

def compute_isp_stats(audio, sr, os_factor=8, win_s=0.05, hop_s=0.025):
    if audio is None or sr is None or sr <= 0:
        return None
    if not isinstance(audio, np.ndarray) or audio.size == 0:
        return None
    x = audio.astype(np.float64, copy=False)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    n = x.shape[0]
    chs = x.shape[1]

    try:
        from scipy.signal import resample_poly
        use_sp = True
    except Exception:
        use_sp = False

    osf = int(max(16, float(os_factor)))

    def _oversample_chunk(seg):
        if use_sp:
            try:
                y = resample_poly(seg, osf, 1, axis=0, padtype='constant', window=('kaiser', 12.0))
            except Exception:
                y = resample_poly(seg, osf, 1, axis=0, padtype='constant')
            return y
        else:
            m = seg.shape[0]
            xp = np.arange(m, dtype=np.float64)
            xos = np.linspace(0.0, m - 1.0, m * osf, endpoint=False, dtype=np.float64)
            if chs == 1:
                yos = np.interp(xos, xp, seg[:, 0]).reshape(-1, 1)
            else:
                yos = np.zeros((xos.size, chs), dtype=np.float64)
                for c in range(chs):
                    yos[:, c] = np.interp(xos, xp, seg[:, c])
            return yos

    def _margins_for(win_sec, hop_sec):
        win_raw = max(32, int(round(win_sec * sr)))
        hop_raw_nom = max(1, int(round(hop_sec * sr)))
        hop_raw = int(min(hop_raw_nom, max(1, win_raw // 4)))
        if n < win_raw or hop_raw <= 0:
            y = _oversample_chunk(x)
            if y.size == 0:
                return None
            abs_os = np.max(np.abs(y), axis=1)
            p = float(np.max(abs_os)) if abs_os.size else 0.0
            tp_db = dbfs(p) if p > 0 else -np.inf
            if not np.isfinite(tp_db):
                return None
            return np.array([-tp_db], dtype=float)

        win_os = int(max(8, round(win_sec * sr * osf)))
        hop_os = int(max(1, round(hop_sec * sr * osf)))
        overlap_raw = max(win_raw, int(0.02 * sr))
        chunk_raw = 262144
        if chunk_raw <= overlap_raw:
            chunk_raw = overlap_raw * 2

        margins = []
        st = 0
        while st < n:
            en = min(n, st + chunk_raw)
            seg = x[st:en, :]
            if seg.shape[0] <= 0:
                break
            y = _oversample_chunk(seg)
            if y.size == 0 or y.shape[0] < win_os:
                st = en if st == 0 else st + max(1, chunk_raw - overlap_raw)
                continue
            abs_os = np.max(np.abs(y), axis=1)
            overlap_os = overlap_raw * osf
            valid_len = int(max(0, abs_os.size - win_os - overlap_os))
            if valid_len < 0:
                valid_len = 0
            if valid_len >= 0:
                stop = valid_len
                if stop >= 0:
                    idx0 = 0
                    if stop >= 0:
                        count = (stop // hop_os) + 1 if stop >= 0 else 0
                        if count > 0 and abs_os.size >= (count - 1) * hop_os + win_os:
                            stride = abs_os.strides[0]
                            shape = (count, win_os)
                            strides = (hop_os * stride, stride)
                            view = np.lib.stride_tricks.as_strided(abs_os, shape=shape, strides=strides, writeable=False)
                            wmax = np.max(view, axis=1)
                            if wmax.size:
                                m = -20.0 * np.log10(np.maximum(wmax, 1e-30))
                                margins.append(m.astype(np.float64, copy=False))
            if en >= n:
                break
            st = st + max(1, chunk_raw - overlap_raw)

        if not margins:
            y = _oversample_chunk(x)
            if y.size == 0:
                return None
            abs_os = np.max(np.abs(y), axis=1)
            if abs_os.size < win_os:
                p = float(np.max(abs_os)) if abs_os.size else 0.0
                tp_db = dbfs(p) if p > 0 else -np.inf
                if not np.isfinite(tp_db):
                    return None
                return np.array([-tp_db], dtype=float)
            count = (abs_os.size - win_os) // hop_os + 1
            if count <= 0:
                return None
            stride = abs_os.strides[0]
            shape = (count, win_os)
            strides = (hop_os * stride, stride)
            view = np.lib.stride_tricks.as_strided(abs_os, shape=shape, strides=strides, writeable=False)
            wmax = np.max(view, axis=1)
            m = -20.0 * np.log10(np.maximum(wmax, 1e-30))
            return m.astype(np.float64, copy=False)

        arr = np.concatenate(margins) if len(margins) > 1 else margins[0]
        if arr.size == 0:
            return None
        return np.asarray(arr, dtype=np.float64)

    m20 = _margins_for(0.020, 0.010)
    m50 = _margins_for(0.050, 0.025)

    if m20 is None and m50 is None:
        tp = _true_peak_internal(x, sr, os_factor=osf)
        if tp is None or not np.isfinite(tp):
            return None
        margin = -tp
        return {
            "isp_window_count": 1,
            "isp_margin_p05_db": margin,
            "isp_margin_p50_db": margin,
            "isp_margin_p95_db": margin,
            "isp_under_1db_count": int(margin < 1.0),
            "isp_under_05db_count": int(margin < 0.5),
            "isp_under_02db_count": int(margin < 0.2),
            "isp_under_1db_frac": float(int(margin < 1.0)),
            "isp_under_05db_frac": float(int(margin < 0.5)),
            "isp_under_02db_frac": float(int(margin < 0.2)),
            "isp_window_count_20ms": 1,
            "isp_margin_p05_db_20ms": margin,
            "isp_margin_p50_db_20ms": margin,
            "isp_margin_p95_db_20ms": margin,
            "isp_under_1db_count_20ms": int(margin < 1.0),
            "isp_under_05db_count_20ms": int(margin < 0.5),
            "isp_under_02db_count_20ms": int(margin < 0.2),
            "isp_under_1db_frac_20ms": float(int(margin < 1.0)),
            "isp_under_05db_frac_20ms": float(int(margin < 0.5)),
            "isp_under_02db_frac_20ms": float(int(margin < 0.2)),
            "isp_window_count_50ms": 1,
            "isp_margin_p05_db_50ms": margin,
            "isp_margin_p50_db_50ms": margin,
            "isp_margin_p95_db_50ms": margin,
            "isp_under_1db_count_50ms": int(margin < 1.0),
            "isp_under_05db_count_50ms": int(margin < 0.5),
            "isp_under_02db_count_50ms": int(margin < 0.2),
            "isp_under_1db_frac_50ms": float(int(margin < 1.0)),
            "isp_under_05db_frac_50ms": float(int(margin < 0.5)),
            "isp_under_02db_frac_50ms": float(int(margin < 0.2)),
            "isp_multi_ratio_05": 1.0,
            "isp_multi_ratio_02": 1.0,
            "isp_multi_consistency": "safe"
        }

    def stats(arr):
        k = arr.size
        p05 = float(np.percentile(arr, 5))
        p50 = float(np.percentile(arr, 50))
        p95 = float(np.percentile(arr, 95))
        c1 = int(np.sum(arr < 1.0))
        c05 = int(np.sum(arr < 0.5))
        c02 = int(np.sum(arr < 0.2))
        f1 = float(c1) / float(k)
        f05 = float(c05) / float(k)
        f02 = float(c02) / float(k)
        return k, p05, p50, p95, c1, c05, c02, f1, f05, f02

    if m20 is None:
        m20 = m50.copy()
    if m50 is None:
        m50 = m20.copy()

    k20, p05_20, p50_20, p95_20, c1_20, c05_20, c02_20, f1_20, f05_20, f02_20 = stats(m20)
    k50, p05_50, p50_50, p95_50, c1_50, c05_50, c02_50, f1_50, f05_50, f02_50 = stats(m50)

    ratio_05 = (f05_20 / f05_50) if f05_50 > 1e-12 else (float('inf') if f05_20 > 0 else 1.0)
    ratio_02 = (f02_20 / f02_50) if f02_50 > 1e-12 else (float('inf') if f02_20 > 0 else 1.0)
    if f05_20 < 0.02 and f05_50 < 0.02:
        tag = "safe"
    elif f05_20 >= 0.05 and f05_50 >= 0.05:
        tag = "dense"
    elif ratio_05 >= 3.0 and f05_20 >= 0.05 and f05_50 < 0.02:
        tag = "spiky"
    else:
        tag = "mixed"

    return {
        "isp_window_count": int(k50),
        "isp_margin_p05_db": p05_50,
        "isp_margin_p50_db": p50_50,
        "isp_margin_p95_db": p95_50,
        "isp_under_1db_count": c1_50,
        "isp_under_05db_count": c05_50,
        "isp_under_02db_count": c02_50,
        "isp_under_1db_frac": f1_50,
        "isp_under_05db_frac": f05_50,
        "isp_under_02db_frac": f02_50,
        "isp_window_count_20ms": int(k20),
        "isp_margin_p05_db_20ms": p05_20,
        "isp_margin_p50_db_20ms": p50_20,
        "isp_margin_p95_db_20ms": p95_20,
        "isp_under_1db_count_20ms": c1_20,
        "isp_under_05db_count_20ms": c05_20,
        "isp_under_02db_count_20ms": c02_20,
        "isp_under_1db_frac_20ms": f1_20,
        "isp_under_05db_frac_20ms": f05_20,
        "isp_under_02db_frac_20ms": f02_20,
        "isp_window_count_50ms": int(k50),
        "isp_margin_p05_db_50ms": p05_50,
        "isp_margin_p50_db_50ms": p50_50,
        "isp_margin_p95_db_50ms": p95_50,
        "isp_under_1db_count_50ms": c1_50,
        "isp_under_05db_count_50ms": c05_50,
        "isp_under_02db_count_50ms": c02_50,
        "isp_under_1db_frac_50ms": f1_50,
        "isp_under_05db_frac_50ms": f05_50,
        "isp_under_02db_frac_50ms": f02_50,
        "isp_multi_ratio_05": float(ratio_05) if np.isfinite(ratio_05) else float('inf'),
        "isp_multi_ratio_02": float(ratio_02) if np.isfinite(ratio_02) else float('inf'),
        "isp_multi_consistency": tag
    }

def compute_transient_metrics(signal: np.ndarray, sr: int):
    import os
    if signal is None or sr is None or sr <= 0:
        return None, None
    x = np.asarray(signal, dtype=np.float64)
    if x.size < int(0.2 * sr):
        return None, None
    sr_eff = int(sr)
    try:
        native = os.environ.get("AQC_NATIVE_SR", "").strip().lower() in ("1", "true", "on", "yes", "y")
    except Exception:
        native = False
    if not native:
        try:
            if sr >= 96000:
                from scipy.signal import resample_poly
                x = resample_poly(x, 1, 2)
                sr_eff = int(round(sr / 2))
        except Exception:
            sr_eff = int(sr)
    try:
        nyq = sr_eff * 0.5
        hp = 120.0 / nyq
        if 0 < hp < 1:
            sos_hp = butter(2, hp, btype='highpass', output='sos')
            x = sosfilt(sos_hp, x)
    except Exception:
        pass
    bands = []
    nyq_hz = sr_eff * 0.5
    hi3 = min(12000.0, nyq_hz * 0.9)
    bdef = [(150.0, 800.0), (800.0, 4000.0), (4000.0, hi3)]
    for lo, hi in bdef:
        if hi <= lo:
            continue
        if hi >= nyq_hz * 0.98:
            hi = nyq_hz * 0.98
        if lo >= hi or hi <= 0:
            continue
        wl = max(1e-6, lo / nyq_hz)
        wh = min(0.999, hi / nyq_hz)
        if wl < wh:
            try:
                sos = butter(3, [wl, wh], btype='band', output='sos')
                bands.append(sos)
            except Exception:
                pass
    if not bands:
        bands = [None]
    win = int(round(0.020 * sr_eff))
    hop = max(1, int(round(0.005 * sr_eff)))
    if win <= 0 or hop <= 0 or x.size <= win:
        return None, None
    try:
        wwin = np.hanning(win)
    except Exception:
        wwin = np.ones(win, dtype=np.float64)
    flux_bands = []
    nfft = int(2 ** math.ceil(math.log2(win)))
    for sos in bands:
        if sos is None:
            y = x
        else:
            try:
                from scipy.signal import sosfiltfilt as _sff
                y = _sff(sos, x)
            except Exception:
                y = sosfilt(sos, x)
        specs = []
        for i in range(0, x.size - win + 1, hop):
            seg = y[i:i + win]
            if seg.size < win:
                break
            spec = np.abs(np.fft.rfft(seg * wwin, n=nfft))
            specs.append(spec)
        if len(specs) <= 1:
            continue
        specs = np.array(specs, dtype=np.float64)
        diff = np.diff(specs, axis=0)
        diff = np.maximum(0.0, diff)
        flux = np.sum(diff, axis=1)
        med = float(np.median(flux)) if flux.size else 0.0
        mad = float(np.median(np.abs(flux - med))) if flux.size else 0.0
        denom = mad if mad > 1e-12 else (np.std(flux) if np.std(flux) > 1e-12 else 1.0)
        flux_norm = (flux - med) / denom
        flux_norm = np.maximum(0.0, flux_norm)
        flux_bands.append(flux_norm)
    if not flux_bands:
        return None, None
    min_len = min(f.size for f in flux_bands)
    if min_len <= 1:
        return None, None
    fb = [f[:min_len] for f in flux_bands]
    comb = np.sum(np.vstack(fb), axis=0)
    med = float(np.median(comb))
    mad = float(np.median(np.abs(comb - med))) if comb.size else 0.0
    thr = med + 3.0 * (mad if mad > 1e-12 else np.std(comb))
    p75 = float(np.percentile(comb, 75)) if comb.size > 1 else thr
    if not np.isfinite(thr) or thr < p75:
        thr = p75
    onset_idx = np.where(comb > thr)[0] + 1
    if onset_idx.size == 0:
        return None, None
    valid = []
    min_gap = int(round(0.050 * sr_eff / hop))
    last = None
    for oi in onset_idx:
        if last is None or (oi - last) >= max(1, min_gap):
            valid.append(oi)
            last = oi
    onset_indices = np.array(valid, dtype=int)
    if onset_indices.size < 5:
        return None, None
    crest_list = []
    rise_list = []
    try:
        analytic = hilbert(x)
        env_full = np.abs(analytic)
    except Exception:
        env_full = np.abs(x)
    seg_len = int(round(0.020 * sr_eff))
    pre = int(round(0.005 * sr_eff))
    for of in onset_indices:
        onset_sample = int(of * hop)
        start = max(0, onset_sample - pre)
        end = min(x.size, start + seg_len)
        seg = x[start:end]
        if seg.size < int(0.005 * sr_eff):
            continue
        peak = float(np.max(np.abs(seg)))
        rms = calculate_rms(seg)
        if peak <= 1e-9 or rms <= 1e-12:
            continue
        crest = peak / rms
        if np.isfinite(crest) and 1.0 < crest < 50.0:
            crest_list.append(float(crest))
        env = env_full[start:end]
        if env.size < 4:
            continue
        p = float(np.max(env))
        if p <= 1e-12:
            continue
        e10 = 0.10 * p
        e90 = 0.90 * p
        i10_all = np.where(env >= e10)[0]
        i90_all = np.where(env >= e90)[0]
        if i10_all.size == 0 or i90_all.size == 0:
            continue
        t10 = int(i10_all[0])
        i90_v = i90_all[i90_all >= t10]
        if i90_v.size == 0:
            continue
        t90 = int(i90_v[0])
        if t90 > t10:
            rise_ms = (t90 - t10) / sr_eff * 1000.0
            if 0.05 < rise_ms < 60.0:
                rise_list.append(float(rise_ms))
    median_crest = float(np.nanmedian(np.array(crest_list, dtype=float))) if len(crest_list) >= 3 else None
    median_rise = float(np.nanmedian(np.array(rise_list, dtype=float))) if len(rise_list) >= 3 else None
    return median_crest, median_rise

def _detect_nominal_bits(sf_file):
    st = (getattr(sf_file, "subtype", None) or "").upper()
    if "FLOAT" in st or "DOUBLE" in st:
        return None
    subtype_map = {
        'PCM_U8': 8, 'PCM_S8': 8, 'PCM_8': 8,
        'PCM_16': 16, 'PCM_S16': 16, 'PCM_16_IN32': 16,
        'PCM_20': 20,
        'PCM_24': 24, 'PCM_S24': 24, 'PCM_24_IN32': 24,
        'PCM_32': 32, 'PCM_S32': 32
    }
    if st in subtype_map:
        return subtype_map[st]
    try:
        info = sf.info(sf_file.name)
        sub_info = (info.subtype_info or "").lower()
        fmt_info = (info.format_info or "").lower()
        if "float" in sub_info or "double" in sub_info:
            return None
        m = re.search(r'(\d+)\s*bit', sub_info)
        if not m:
            m = re.search(r'(\d+)\s*bit', fmt_info)
        if m:
            b = int(m.group(1))
            if b in (8, 16, 20, 24, 32):
                return b
        m2 = re.search(r'pcm[_\-]s?(\d+)', sub_info)
        if not m2:
            m2 = re.search(r'pcm[_\-]s?(\d+)', fmt_info)
        if m2:
            b = int(m2.group(1))
            if b in (8, 16, 20, 24, 32):
                return b
    except Exception:
        pass
    try:
        path = getattr(sf_file, "name", None)
        if path and os.path.isfile(path):
            out = subprocess.run(
                ['ffprobe', '-v', 'error', '-select_streams', 'a:0', '-show_entries', 'stream=bits_per_sample', '-of', 'default=noprint_wrappers=1:nokey=1', path],
                check=False, capture_output=True, text=True, encoding='utf-8', errors='replace'
            )
            txt = (out.stdout or "").strip()
            if txt.isdigit():
                b = int(txt)
                if b in (8, 16, 20, 24, 32):
                    return b
    except Exception:
        pass
    return None

def compute_dr_tt_true(left: np.ndarray, right: np.ndarray, sr: int, block_sec: float = 3.0, gate_db: float = -60.0):
    if left is None or right is None or len(left) == 0 or len(right) == 0 or sr is None or sr <= 0:
        return 0.0, 0.0, np.array([], dtype=float)

    l_f = tt_bandpass_filter(left, sr)
    r_f = tt_bandpass_filter(right, sr)
    if l_f.size == 0 or r_f.size == 0:
        return 0.0, 0.0, np.array([], dtype=float)

    n = min(len(l_f), len(r_f))
    if n <= 0:
        return 0.0, 0.0, np.array([], dtype=float)
    l_f = l_f[:n].astype(np.float64, copy=False)
    r_f = r_f[:n].astype(np.float64, copy=False)

    def _frame(signal, frame_len, hop_len):
        if signal.size < frame_len or frame_len <= 0 or hop_len <= 0:
            return np.empty((0, frame_len), dtype=np.float64)
        n_frames = 1 + (signal.size - frame_len) // hop_len
        stride = signal.strides[0]
        return np.lib.stride_tricks.as_strided(signal, shape=(n_frames, frame_len), strides=(hop_len * stride, stride))

    block = int(max(1, round(block_sec * sr)))
    hop_b = max(1, block // 2)
    if block > n:
        block = n
        hop_b = max(1, block // 2)

    L_frames = _frame(l_f, block, hop_b)
    R_frames = _frame(r_f, block, hop_b)

    if L_frames.size == 0 or R_frames.size == 0:
        return 0.0, 0.0, np.array([], dtype=float)

    rmsL = np.sqrt(np.mean(L_frames * L_frames, axis=1))
    rmsR = np.sqrt(np.mean(R_frames * R_frames, axis=1))

    with np.errstate(divide='ignore'):
        rmsL_db = 20.0 * np.log10(np.maximum(rmsL, 1e-30))
        rmsR_db = 20.0 * np.log10(np.maximum(rmsR, 1e-30))

    selL = rmsL[rmsL_db > gate_db]
    selR = rmsR[rmsR_db > gate_db]

    if selL.size > 0:
        keep_l = max(1, int(round(0.20 * selL.size)))
        topL = np.sort(selL)[-keep_l:]
        loud_l = float(np.sqrt(np.mean(topL * topL)))
    else:
        loud_l = 0.0
    if selR.size > 0:
        keep_r = max(1, int(round(0.20 * selR.size)))
        topR = np.sort(selR)[-keep_r:]
        loud_r = float(np.sqrt(np.mean(topR * topR)))
    else:
        loud_r = 0.0

    return loud_l, loud_r, np.array([], dtype=float)

def _calc_spectral_balance_dev(mag, xf, ref_slope_db_oct=-3.0):
    if mag is None or xf is None:
        return None
    if not isinstance(mag, np.ndarray) or not isinstance(xf, np.ndarray):
        return None
    if mag.size != xf.size or mag.size < 32:
        return None
    mag = np.asarray(mag, dtype=np.float64)
    xf = np.asarray(xf, dtype=np.float64)
    finite = np.isfinite(mag) & np.isfinite(xf) & (mag > 0) & (xf > 0)
    if not np.any(finite):
        return None
    mag = mag[finite]
    xf = xf[finite]
    nyq_approx = float(np.max(xf))
    f_min = 31.25
    f_max = min(16000.0, nyq_approx * 0.98)
    if f_max <= f_min * 1.1:
        return None
    centers = []
    fc = f_min
    k_step = 2.0 ** (1.0 / 3.0)
    while fc <= f_max:
        centers.append(fc)
        fc *= k_step
    centers = np.array(centers, dtype=np.float64)
    if centers.size < 3:
        return None
    xs = []
    ys = []
    ws = []
    eps = 1e-30
    for fc in centers:
        fl = fc / (2.0 ** (1.0 / 6.0))
        fh = fc * (2.0 ** (1.0 / 6.0))
        mask = (xf >= fl) & (xf < fh)
        if not np.any(mask):
            continue
        band_mag = mag[mask]
        if band_mag.size < 2:
            continue
        band_mag = np.maximum(band_mag, eps)
        band_db = 20.0 * np.log10(band_mag)
        med_db = float(np.median(band_db[np.isfinite(band_db)]))
        if not np.isfinite(med_db):
            continue
        nb = band_mag.size
        energy = float(np.sum(band_mag * band_mag))
        w = max(1.0, math.log1p(nb)) * math.sqrt(max(energy, eps))
        xs.append(np.log2(fc))
        ys.append(med_db)
        ws.append(w)
    if not xs or not ys or not ws or len(xs) < 3:
        return None
    xs = np.array(xs, dtype=np.float64)
    ys = np.array(ys, dtype=np.float64)
    ws = np.array(ws, dtype=np.float64)
    t = ys - ref_slope_db_oct * xs
    def _weighted_mean(v, w):
        wt = np.sum(w)
        if wt <= 0:
            return float(np.mean(v))
        return float(np.sum(v * w) / wt)
    def _huber_location(v, w, c=1.5, iters=3):
        k = _weighted_mean(v, w)
        for _ in range(iters):
            r = v - k
            s = 1.4826 * float(np.median(np.abs(r))) + 1e-12
            if not np.isfinite(s) or s <= 0:
                break
            w_h = np.ones_like(r)
            mask = np.abs(r) > c * s
            w_h[mask] = (c * s) / (np.abs(r[mask]) + 1e-30)
            w_eff = w * w_h
            k = _weighted_mean(v, w_eff)
        return float(k)
    k = _huber_location(t, ws, c=1.5, iters=4)
    y_pred = k + ref_slope_db_oct * xs
    dev = np.abs(ys - y_pred)
    def _weighted_median(v, w):
        order = np.argsort(v)
        v_s = v[order]
        w_s = w[order]
        cw = np.cumsum(w_s)
        tot = float(np.sum(w_s))
        if tot <= 0:
            return float(np.median(v))
        idx = int(np.searchsorted(cw, 0.5 * tot, side='left'))
        idx = max(0, min(idx, v_s.size - 1))
        return float(v_s[idx])
    dev_med = _weighted_median(dev, ws)
    r = dev - dev_med
    s = 1.4826 * float(np.median(np.abs(r))) + 1e-12
    w_h = np.ones_like(dev)
    mask = np.abs(r) > 1.5 * s
    w_h[mask] = (1.5 * s) / (np.abs(r[mask]) + 1e-30)
    dev_mean = _weighted_mean(dev, ws * w_h)
    if not np.isfinite(dev_mean):
        return None
    return float(max(0.0, min(dev_mean, 30.0)))


def _short_term_lufs_iqr(audio_f32, sr):
    if audio_f32 is None or sr is None or sr <= 0:
        return None
    try:
        if isinstance(audio_f32, np.ndarray):
            if audio_f32.ndim == 2:
                if audio_f32.shape[0] < audio_f32.shape[1] and audio_f32.shape[1] in (1, 2):
                    x_mono = audio_f32.mean(axis=1)
                elif audio_f32.shape[0] in (1, 2):
                    x_mono = audio_f32.T.mean(axis=1)
                else:
                    axis = 1 if audio_f32.shape[1] in (1, 2) else 0
                    x_mono = audio_f32.mean(axis=axis)
            else:
                x_mono = audio_f32
        else:
            return None
        x_mono = np.asarray(x_mono, dtype=np.float32)
    except Exception:
        return None
    n = x_mono.size
    if n < int(0.4 * sr):
        return None
    try:
        import pyloudnorm as pyln
        meter = pyln.Meter(sr, block_size=0.400, filter_class="K-weighting")
        lt = meter.loudness_time_series(x_mono)
        if lt is None:
            raise RuntimeError("lt none")
        lt = np.array(lt, dtype=float)
        lt = lt[np.isfinite(lt)]
        if lt.size < 5:
            raise RuntimeError("lt short")
        med = float(np.median(lt))
        gate = max(-70.0, med - 10.0)
        ltg = lt[lt >= gate]
        if ltg.size < max(5, int(0.05 * lt.size)):
            idx = np.argsort(lt)[-max(5, int(0.05 * lt.size)):]
            ltg = lt[idx]
        p95 = float(np.percentile(ltg, 95))
        p5 = float(np.percentile(ltg, 5))
        d = p95 - p5
        if not np.isfinite(d) or d <= 0:
            return None
        return float(min(max(d, 0.0), 30.0))
    except Exception:
        try:
            win = int(round(0.4 * sr))
            hop = max(1, int(round(0.1 * sr)))
            if win <= 0 or hop <= 0 or n < win:
                return None
            vals = []
            for i in range(0, n - win + 1, hop):
                seg = x_mono[i:i + win].astype(np.float64, copy=False)
                rms = calculate_rms(seg)
                if rms > 1e-12:
                    v = dbfs(rms)
                    if np.isfinite(v):
                        vals.append(v)
            if len(vals) < 5:
                return None
            arr = np.array(vals, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size < 5:
                return None
            med = float(np.median(arr))
            gate = max(-70.0, med - 10.0)
            arrg = arr[arr >= gate]
            if arrg.size < max(5, int(0.05 * arr.size)):
                idx = np.argsort(arr)[-max(5, int(0.05 * arr.size)):]
                arrg = arr[idx]
            p95 = float(np.percentile(arrg, 95))
            p5 = float(np.percentile(arrg, 5))
            d = p95 - p5
            if not np.isfinite(d) or d <= 0:
                return None
            return float(min(max(d, 0.0), 30.0))
        except Exception:
            return None

def _reverb_tail_ratio_db(mono, sr):
    import os
    if mono is None or sr is None or sr <= 0:
        return None
    x = np.asarray(mono, dtype=np.float64)
    n = x.size
    if n < int(0.1 * sr):
        return None
    sr_eff = int(sr)
    try:
        native = os.environ.get("AQC_NATIVE_SR", "").strip().lower() in ("1", "true", "on", "yes", "y")
    except Exception:
        native = False
    if not native:
        try:
            if sr > 48000:
                from scipy.signal import resample_poly
                g = math.gcd(int(sr), 48000)
                up = 48000 // g
                down = int(sr) // g
                if up > 0 and down > 0:
                    x = resample_poly(x, up, down)
                    sr_eff = int(round(sr * (up / down)))
                else:
                    sr_eff = int(sr)
            else:
                sr_eff = int(sr)
        except Exception:
            sr_eff = int(sr)
    try:
        nyq = sr_eff * 0.5
        lo = 100.0 / nyq
        hi = min(8000.0, nyq * 0.98) / nyq
        if hi > lo and hi < 1.0:
            sos = butter(2, [lo, hi], btype='band', output='sos')
            try:
                from scipy.signal import sosfiltfilt as _sff
                x = _sff(sos, x)
            except Exception:
                x = sosfilt(sos, x)
        elif 0 < lo < 1.0:
            sos = butter(2, lo, btype='highpass', output='sos')
            try:
                from scipy.signal import sosfiltfilt as _sff
                x = _sff(sos, x)
            except Exception:
                x = sosfilt(sos, x)
    except Exception:
        pass
    try:
        env = np.abs(hilbert(x))
    except Exception:
        env = np.abs(x)
    win_s = 0.020
    wlen = int(max(3, round(win_s * sr_eff)))
    if wlen % 2 == 0:
        wlen += 1
    if wlen > 1 and env.size >= wlen:
        w = np.hanning(wlen)
        w = w / np.sum(w)
        try:
            from scipy.signal import fftconvolve
            env = fftconvolve(env, w, mode='same')
        except Exception:
            env = np.convolve(env, w, mode='same')
    delays_ms = [50, 100, 200, 300]
    ratios_all = []
    for dms in delays_ms:
        d = int(round((dms / 1000.0) * sr_eff))
        if d <= 0 or env.size <= 2 * d:
            continue
        step = d
        early_rms_list = []
        late_rms_list = []
        for st in range(0, env.size - 2 * d + 1, step):
            e = env[st:st + d]
            l = env[st + d:st + 2 * d]
            re = calculate_rms(e)
            rl = calculate_rms(l)
            if np.isfinite(re) and np.isfinite(rl):
                early_rms_list.append(re)
                late_rms_list.append(rl)
        if len(early_rms_list) < 8:
            continue
        ea = np.array(early_rms_list, dtype=np.float64)
        la = np.array(late_rms_list, dtype=np.float64)
        ea = ea[np.isfinite(ea)]
        la = la[np.isfinite(la)]
        if ea.size < 8 or la.size < 8:
            continue
        p75 = float(np.percentile(ea, 75))
        if not np.isfinite(p75) or p75 <= 0:
            continue
        mask = ea >= p75
        ea_s = ea[mask]
        la_s = la[mask]
        if ea_s.size < 4 or la_s.size < 4:
            continue
        valid = (ea_s > 1e-12) & (la_s > 1e-12)
        if not np.any(valid):
            continue
        ea_s = ea_s[valid]
        la_s = la_s[valid]
        r_db = 20.0 * np.log10(la_s) - 20.0 * np.log10(ea_s)
        r_db = r_db[np.isfinite(r_db)]
        if r_db.size >= 3:
            ratios_all.append(float(np.median(r_db)))
    if not ratios_all:
        return None
    return float(np.median(np.array(ratios_all, dtype=np.float64)))

def _isp_margin_db(audio, sr, os_factor=8):
    if audio is None or sr is None or sr <= 0:
        return None
    try:
        ofs = int(os_factor) if isinstance(os_factor, (int, float)) else 16
        ofs = max(16, ofs)
    except Exception:
        ofs = 16
    tp_db = _true_peak_internal(audio, sr, os_factor=ofs)
    if tp_db is None or not np.isfinite(tp_db):
        return None
    margin = -float(tp_db)
    if not np.isfinite(margin):
        return None
    return float(round(margin * 100.0) / 100.0)


def _dc_offset_dbfs(mono):
    if mono is None:
        return None
    x = np.asarray(mono, dtype=np.float64)
    if x.size == 0:
        return None
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None
    n = x.size
    if n < 8:
        m = float(np.mean(x))
        dc = abs(m)
        if dc < 1e-15:
            return -np.inf
        return dbfs(dc)
    block = 65536
    hop = block // 2
    means = []
    if n <= block:
        means.append(float(np.mean(x)))
    else:
        for st in range(0, n - block + 1, hop):
            seg = x[st:st + block]
            if seg.size > 0:
                means.append(float(np.mean(seg)))
        tail = x[(n - block):] if n > block else None
        if tail is not None and tail.size > 0:
            means.append(float(np.mean(tail)))
    if not means:
        return None
    m_arr = np.array(means, dtype=np.float64)
    med = float(np.median(m_arr))
    mad = float(np.median(np.abs(m_arr - med))) + 1e-18
    mask = np.abs(m_arr - med) <= 6.0 * mad
    core = m_arr[mask] if np.any(mask) else m_arr
    med2 = float(np.median(core))
    mad2 = float(np.median(np.abs(core - med2))) + 1e-18
    r = core - med2
    w = 1.0 / (1.0 + (np.abs(r) / (3.0 * mad2))**2)
    w_sum = float(np.sum(w))
    if w_sum > 0:
        dc_est = float(np.sum(core * w) / w_sum)
    else:
        dc_est = med2
    dc = abs(dc_est)
    if dc < 1e-15:
        return -np.inf
    return dbfs(dc)

def _stereo_width_iqr_db(L, R, sr):
    import os
    if L is None or R is None or sr is None or sr <= 0:
        return None
    n = min(len(L), len(R))
    if n <= 0:
        return None
    L = np.asarray(L[:n], dtype=np.float64)
    R = np.asarray(R[:n], dtype=np.float64)
    try:
        native = os.environ.get("AQC_NATIVE_SR", "").strip().lower() in ("1", "true", "on", "yes", "y")
    except Exception:
        native = False
    sr_eff = int(sr)
    if not native:
        try:
            if sr > 48000:
                from scipy.signal import resample_poly
                g = math.gcd(int(sr), 48000)
                up = 48000 // g
                down = int(sr) // g
                if up > 0 and down > 0:
                    L = resample_poly(L, up, down).astype(np.float64, copy=False)
                    R = resample_poly(R, up, down).astype(np.float64, copy=False)
                    sr_eff = int(round(sr * (up / down)))
                else:
                    sr_eff = int(sr)
            else:
                sr_eff = int(sr)
        except Exception:
            sr_eff = int(sr)
    M = (L + R) * 0.5
    S = (L - R) * 0.5
    try:
        from scipy.signal import butter, sosfilt, sosfiltfilt
        if not hasattr(_stereo_width_iqr_db, "_hp_cache"):
            _stereo_width_iqr_db._hp_cache = {}
        nyq = sr_eff * 0.5
        fc = 120.0
        if nyq > fc:
            Wn = min(0.999, fc / nyq)
            key = (sr_eff, round(Wn, 9))
            sos = _stereo_width_iqr_db._hp_cache.get(key)
            if sos is None:
                sos = butter(2, Wn, btype='highpass', output='sos')
                _stereo_width_iqr_db._hp_cache[key] = sos
            try:
                M = sosfiltfilt(sos, M)
                S = sosfiltfilt(sos, S)
            except Exception:
                M = sosfilt(sos, M)
                S = sosfilt(sos, S)
    except Exception:
        pass
    win = int(round(0.20 * sr_eff))
    hop = max(1, int(round(0.05 * sr_eff)))
    if win < 2 or hop < 1 or M.size < win:
        return None
    def _frame_rows(x, frame_len, hop_len):
        n = x.size
        n_frames = 1 + (n - frame_len) // hop_len
        if n_frames <= 0:
            return np.empty((0, frame_len), dtype=np.float64)
        stride = x.strides[0]
        return np.lib.stride_tricks.as_strided(x, shape=(n_frames, frame_len), strides=(hop_len * stride, stride))
    MF = _frame_rows(M, win, hop)
    SF = _frame_rows(S, win, hop)
    if MF.shape[0] == 0 or SF.shape[0] == 0:
        return None
    rms_M = np.sqrt(np.mean(MF * MF, axis=1))
    rms_S = np.sqrt(np.mean(SF * SF, axis=1))
    with np.errstate(divide='ignore'):
        m_db = 20.0 * np.log10(np.maximum(rms_M, 1e-30))
        s_db = 20.0 * np.log10(np.maximum(rms_S, 1e-30))
    valid = np.isfinite(m_db) & np.isfinite(s_db) & (m_db >= -60.0)
    if not np.any(valid):
        return None
    w = s_db[valid] - m_db[valid]
    w = w[np.isfinite(w)]
    if w.size < 5:
        return None
    p95 = float(np.percentile(w, 95))
    p5 = float(np.percentile(w, 5))
    return float(p95 - p5)

def _policy_select_nf_profile(value, r, policy):
    pol = _policy_resolve(policy)

    def _sf(v):
        try:
            f = float(v)
            return f if np.isfinite(f) else None
        except Exception:
            return None

    try:
        if r and isinstance(r.get('hygiene_eval_bits'), (int, float)):
            pass
    except Exception:
        pass

    candidates = []
    v0 = _sf(value)
    if v0 is not None:
        candidates.append(v0)
    if r is not None:
        for k in ("noise_floor_dbfs", "nf_broadband_dbfs", "noise_floor_raw_dbfs"):
            vv = _sf(r.get(k))
            if vv is not None:
                candidates.append(vv)
    nf_ref = None
    if candidates:
        nf_ref = float(min(candidates))

    nf_p90 = _sf(r.get('noise_floor_cross_p90_dbfs') if r else None)

    conf_raw = (r.get("noise_confidence") or "").strip().lower() if r else ""
    if conf_raw.startswith("alt"):   conf_label = "Alta"
    elif conf_raw.startswith("med"): conf_label = "Media"
    elif conf_raw.startswith("bas"): conf_label = "Bassa"
    else:                            conf_label = "Alta"

    try:
        bd = int(r.get('bit_depth')) if r and r.get('bit_depth') is not None else None
    except Exception:
        bd = None
    try:
        eff = float(r.get('effective_bit_depth')) if r and r.get('effective_bit_depth') is not None else None
    except Exception:
        eff = None
    try:
        sr = int(r.get('samplerate')) if r and r.get('samplerate') is not None else None
    except Exception:
        sr = None

    try:
        nwin = int(r.get('noise_windows_count') or 0)
    except Exception:
        nwin = 0
    try:
        ndur = float(r.get('noise_total_duration_sec') or 0.0)
    except Exception:
        ndur = 0.0
    try:
        diff = float(r.get('noise_consistency_diff_db') or 0.0)
    except Exception:
        diff = 0.0
    music_limited = bool(r.get("music_limited")) if r and r.get("music_limited") is not None else False

    extreme_uncertain = (diff > 30.0) or (nwin < 2 and ndur < 3.0)
    moderate_uncertain = (diff > 20.0) or (nwin < 2 or ndur < 3.0)

    if nf_ref is None or not np.isfinite(nf_ref):
        return "noise_floor_dbfs_16"

    nf_ref_adj = float(nf_ref)
    if nf_p90 is not None and np.isfinite(nf_p90):
        nf_ref_adj = min(nf_ref_adj, float(nf_p90) - 2.0)

    if music_limited or extreme_uncertain or conf_label == "Bassa":
        return "noise_floor_dbfs_16"
    if moderate_uncertain:
        return "noise_floor_dbfs_16"

    hard_24 = -106.0
    soft_24 = -100.0
    hi_sr = (sr is not None and sr >= 96000)

    if eff is not None and eff >= 20 and nf_ref_adj <= hard_24:
        return "noise_floor_dbfs_24"
    if (bd is not None and bd >= 24) or hi_sr:
        if (eff is None or eff >= 18) and nf_ref_adj <= soft_24:
            return "noise_floor_dbfs_24"
    if nf_ref_adj <= (soft_24 - 2.0) and conf_label == "Alta":
        return "noise_floor_dbfs_24"

    return "noise_floor_dbfs_16"

def _metric_contribution(r, policy=None):
    import os
    pol = _policy_resolve(policy)

    def _band_from_symbol(sym):
        s = (sym or "").strip()
        if s == "✅": return "GOOD"
        if s == "⚠️": return "WARN"
        if s == "❌": return "BAD"
        return "N/A"

    def _status(key, val):
        return get_metric_status(key, val, policy=pol, r=r)

    def _safe(v):
        try:
            f = float(v)
            return f if np.isfinite(f) else None
        except Exception:
            return None

    wH_base = pol["weights"]["H"]
    wQ_base = pol["weights"]["Q"]
    mix_base = pol["score_mix"]

    nf_val = _safe(r.get("noise_floor_dbfs"))
    nf_bb = _safe(r.get("nf_broadband_dbfs", nf_val))
    spur_label = str(r.get("noise_spur_label") or "")
    spur_hc = int(r.get("noise_spur_harmonics_count") or 0)
    hum_dense = spur_label.startswith("HUM") and spur_hc >= 10
    nf_use_for_score = nf_bb if (hum_dense and nf_bb is not None and np.isfinite(nf_bb)) else nf_val
    nf_gate_hi = nf_use_for_score is not None and nf_use_for_score > -35.0
    music_limited = bool(r.get("music_limited")) if r.get("music_limited") is not None else False

    conf_raw = (r.get("noise_confidence") or "").strip().lower()
    if conf_raw.startswith("alt"):   conf_label = "Alta"
    elif conf_raw.startswith("med"): conf_label = "Media"
    elif conf_raw.startswith("bas"): conf_label = "Bassa"
    else:                            conf_label = "Alta"
    conf_scale = pol["confidence_weights"].get(conf_label.split()[0].capitalize() if conf_label else "Alta", 1.0)
    nf_delta = _safe(r.get("noise_consistency_diff_db")) or 0.0
    nwin = int(r.get("noise_windows_count") or 0)
    ndur = float(r.get("noise_total_duration_sec") or 0.0)

    mixH_eff = float(mix_base["H"])
    mixQ_eff = float(mix_base["Q"])
    if nf_gate_hi or hum_dense:
        mixH_eff = 0.72
        mixQ_eff = 0.28
    if music_limited:
        mixH_eff = 0.65
        mixQ_eff = 0.35
    if conf_label.startswith("Bassa"):
        mixH_eff = 0.60
        mixQ_eff = 0.40

    w_nf0 = wH_base["noise_floor_dbfs"] * conf_scale
    w_sp0 = wH_base["noise_spur_db"]    * conf_scale
    w_sum_ns = w_nf0 + w_sp0

    w_nf_eff = w_nf0
    w_sp_eff = w_sp0

    if hum_dense:
        w_nf_eff = w_nf0 * 0.6
        w_sp_eff = max(0.0, w_sum_ns - w_nf_eff)

    if music_limited:
        w_nf_eff *= 0.7
        w_sp_eff *= 0.85

    if conf_label.startswith("Bassa"):
        if nf_delta > 10.0:
            w_nf_eff *= 0.6
            w_sp_eff *= 0.6
        elif nf_delta > 8.0:
            w_nf_eff *= 0.65
            w_sp_eff *= 0.65
        elif nf_delta > 6.0:
            w_nf_eff *= 0.75
            w_sp_eff *= 0.75

    moderate_uncertain = (nwin < 2 or ndur < 3.0) or (nf_delta > 20.0)
    extreme_uncertain  = (nf_delta > 30.0) or ((nwin < 2) and (ndur < 3.0))
    if extreme_uncertain:
        w_sp_eff *= 0.70
        w_nf_eff = 0.0
    elif moderate_uncertain:
        w_sp_eff *= 0.75
        w_nf_eff *= 0.50

    contrib = {}

    safe_env = os.environ.get("AQC_NF_SAFE", "").strip().lower()
    nf_safe = (safe_env not in ("0", "false", "off", "no", "n"))

    nf_profile_key = _policy_select_nf_profile(nf_use_for_score, r, pol)
    thr_nf_sel = pol["thresholds"].get(nf_profile_key) if nf_profile_key else None
    thr16 = pol["thresholds"].get("noise_floor_dbfs_16")
    thr24 = pol["thresholds"].get("noise_floor_dbfs_24")
    if nf_safe:
        s16 = policy_map_to_unit_score(nf_use_for_score, thr16) if thr16 is not None else None
        s24 = policy_map_to_unit_score(nf_use_for_score, thr24) if thr24 is not None else None
        cand = [x for x in (s16, s24) if x is not None]
        s_nf = max(cand) if cand else policy_map_to_unit_score(nf_use_for_score, thr_nf_sel)
        nf_mode = "SAFE"
    else:
        s_nf = policy_map_to_unit_score(nf_use_for_score, thr_nf_sel)
        nf_mode = "PROFILE_24" if (nf_profile_key and nf_profile_key.endswith("_24")) else "PROFILE_16"

    H_den = w_nf_eff + w_sp_eff + wH_base["isp_margin_db"] + wH_base["dc_offset_dbfs"]

    if w_nf_eff > 0:
        pts = 0.0
        if s_nf is not None and H_den > 0:
            pts = 100.0 * mixH_eff * (s_nf * w_nf_eff / H_den)
        contrib["noise_floor_dbfs"] = {
            "value": nf_use_for_score,
            "band": _band_from_symbol(_status("noise_floor_dbfs", nf_use_for_score)),
            "punti": float(pts),
            "group": "H",
            "mode": nf_mode
        }
    else:
        contrib["noise_floor_dbfs"] = {
            "value": nf_use_for_score,
            "band": _band_from_symbol(_status("noise_floor_dbfs", nf_use_for_score)),
            "punti": 0.0,
            "group": "H",
            "mode": "EXCLUDED"
        }

    spur_val = _safe(r.get("noise_spur_db"))
    s_spur = policy_map_to_unit_score(spur_val, _policy_metric_thresholds("noise_spur_db", spur_val, pol, r=r))
    pts_sp = 0.0
    if s_spur is not None and H_den > 0 and w_sp_eff > 0:
        pts_sp = 100.0 * mixH_eff * (s_spur * w_sp_eff / H_den)
    contrib["noise_spur_db"] = {
        "value": spur_val,
        "band": _band_from_symbol(_status("noise_spur_db", spur_val)),
        "punti": float(pts_sp),
        "group": "H"
    }

    isp_val = _safe(r.get("isp_margin_db"))
    s_isp = policy_map_to_unit_score(isp_val, _policy_metric_thresholds("isp_margin_db", isp_val, pol, r=r))
    pts_isp = 0.0
    if s_isp is not None and H_den > 0 and wH_base["isp_margin_db"] > 0:
        pts_isp = 100.0 * mixH_eff * (s_isp * wH_base["isp_margin_db"] / H_den)
    contrib["isp_margin_db"] = {
        "value": isp_val,
        "band": _band_from_symbol(_status("isp_margin_db", isp_val)),
        "punti": float(pts_isp),
        "group": "H"
    }

    dc_val = _safe(r.get("dc_offset_dbfs"))
    s_dc = policy_map_to_unit_score(dc_val, _policy_metric_thresholds("dc_offset_dbfs", dc_val, pol, r=r))
    pts_dc = 0.0
    if s_dc is not None and H_den > 0 and wH_base["dc_offset_dbfs"] > 0:
        pts_dc = 100.0 * mixH_eff * (s_dc * wH_base["dc_offset_dbfs"] / H_den)
    contrib["dc_offset_dbfs"] = {
        "value": dc_val,
        "band": _band_from_symbol(_status("dc_offset_dbfs", dc_val)),
        "punti": float(pts_dc),
        "group": "H"
    }

    plr_val = _safe(r.get("plr_effective_db", r.get("plr_est")))
    wQ_eff_map = dict(wQ_base)
    H_score_val = _safe(r.get("hygiene_score"))
    if conf_label.startswith("Bassa") or music_limited or extreme_uncertain or (H_score_val is not None and H_score_val < 0.45):
        total = sum(wQ_eff_map.values())
        wQ_eff_map["spectral_balance_dev_db"] = 0.20
        wQ_eff_map["stereo_width_iqr_db"] = 0.01
        wQ_eff_map["lra_est"] = 0.14
        wQ_eff_map["dr_tt_avg"] = 0.30
        wQ_eff_map["plr_est"] = 0.20
        wQ_eff_map["st_lufs_iqr_db"] = 0.15
        ssum = sum(wQ_eff_map.values())
        if ssum > 0 and abs(ssum - total) > 1e-9:
            scale = total / ssum
            for k in wQ_eff_map:
                wQ_eff_map[k] *= scale

    Q_specs = [
        ("dr_tt_avg",               wQ_eff_map["dr_tt_avg"],               _policy_metric_thresholds("dr_tt_avg", _safe(r.get("dr_tt_avg")), pol, r=r)),
        ("plr_effective_db",        wQ_eff_map["plr_est"],                 _policy_metric_thresholds("plr_est", plr_val, pol, r=r)),
        ("lra_est",                 wQ_eff_map["lra_est"],                 _policy_metric_thresholds("lra_est", _safe(r.get("lra_est")), pol, r=r)),
        ("st_lufs_iqr_db",          wQ_eff_map["st_lufs_iqr_db"],          _policy_metric_thresholds("st_lufs_iqr_db", _safe(r.get("st_lufs_iqr_db")), pol, r=r)),
        ("spectral_balance_dev_db", wQ_eff_map["spectral_balance_dev_db"], _policy_metric_thresholds("spectral_balance_dev_db", _safe(r.get("spectral_balance_dev_db")), pol, r=r)),
        ("stereo_width_iqr_db",     wQ_eff_map["stereo_width_iqr_db"],     _policy_metric_thresholds("stereo_width_iqr_db", _safe(r.get("stereo_width_iqr_db")), pol, r=r)),
    ]
    Q_den = sum(w for _, w, _ in Q_specs if w > 0)

    for key, w_eff, thr in Q_specs:
        if key == "plr_effective_db":
            val = plr_val
            band_key = "plr_est"
        else:
            val = _safe(r.get(key))
            band_key = key
        s = policy_map_to_unit_score(val, thr) if thr is not None else None
        if key == "stereo_width_iqr_db" and s is not None and val is not None and np.isfinite(val):
            if val > 10.0:
                factor = max(0.4, min(1.0, 10.0 / float(val)))
                s = s * factor
        pts = 0.0
        if s is not None and Q_den > 0 and w_eff > 0:
            pts = 100.0 * mixQ_eff * (s * w_eff / Q_den)
        contrib[key] = {
            "value": val,
            "band": _band_from_symbol(_status(band_key, val)),
            "punti": float(pts),
            "group": "Q"
        }

    extras = [
        "clipping_detected",
        "true_peak_est_dbtp",
        "peak_dbfs_overall",
        "loudness_lufs",
        "thdn_db",
        "jitter_ppm",
        "hf_rms_var_db",
        "hf_var_norm_pct",
        "stereo_correlation",
        "reverb_tail_ratio_db",
        "transient_crest_med",
        "transient_rise_med_ms",
        "effective_bit_depth",
        "noise_index_db",
        "noise_floor_raw_dbfs",
        "nf_broadband_dbfs",
        "noise_windows_count",
        "noise_total_duration_sec",
        "noise_total_duration_uncapped_sec",
        "nf_notch_reduction_db",
        "notch_f0",
        "notch_harmonics",
        "music_limited",
        "noise_rms_p90_dbfs",
        "hum_estimated_hcount",
        "noise_floor_cap_reason",
    ]
    for key in extras:
        if key in contrib:
            continue
        val = r.get(key)
        try:
            if isinstance(val, (bool, np.bool_)):
                v = bool(val)
            elif val is None:
                v = None
            else:
                v = float(val)
        except Exception:
            v = val
        contrib[key] = {
            "value": v,
            "band": _band_from_symbol(_status(key, v)),
            "punti": 0.0,
            "group": "support"
        }

    contrib["_meta"] = {
        "score_mix_eff": {"H": float(mixH_eff), "Q": float(mixQ_eff)},
        "weights_H_eff": {
            "noise_floor_dbfs": float(w_nf_eff),
            "noise_spur_db": float(w_sp_eff),
            "isp_margin_db": float(wH_base["isp_margin_db"]),
            "dc_offset_dbfs": float(wH_base["dc_offset_dbfs"])
        },
        "weights_Q_eff": {k: float(v) for k, v in wQ_eff_map.items()},
        "nf_scoring_mode": "SAFE" if nf_safe else ("PROFILE_24" if (nf_profile_key and nf_profile_key.endswith("_24")) else "PROFILE_16")
    }
    return contrib


def write_human_log(results_list, path_dir=".", excluded_clipping=None, md5_groups=None, md5_representatives=None, policy=None):
    pol = _policy_resolve(policy)
    if not results_list:
        print("Nessun risultato da loggare.")
        return
    import sys, platform, hashlib, subprocess, time, os

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    fname = f"log_{ts}.txt"
    fpath = os.path.join(path_dir, fname)

    def _fmt(v, dec=2, na="N/A"):
        try:
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                return na
            return f"{float(v):.{dec}f}"
        except Exception:
            return na

    def _fmt_signed(v, dec=2, na="N/A"):
        try:
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                return na
            f = float(v)
            sign = "+" if f >= 0 else ""
            return f"{sign}{f:.{dec}f}"
        except Exception:
            return na

    def _fmt_dc(v):
        try:
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                return "N/A"
            f = float(v)
            if f <= -180.0:
                return "≤-180.0"
            return f"{f:.1f}"
        except Exception:
            return "N/A"

    def _conf_label(r):
        raw = (r.get("noise_confidence") or "").strip()
        return raw if raw else "Alta"

    def _status(metric_key, value, r):
        return get_metric_status(metric_key, value, policy=pol, r=r) or ""

    def _contrib_map(r):
        return _metric_contribution(r, policy=pol)

    def _md5_for_result(r):
        if not isinstance(md5_groups, dict):
            return None
        original_path = r.get('_original_ref_path', r.get('filepath'))
        for h, paths in md5_groups.items():
            if original_path in paths:
                return h
        return None

    def _sig_tuple(r):
        try:
            return make_signature(r)
        except Exception:
            return None

    def _sig_id(sig):
        try:
            raw = repr(sig).encode('utf-8', 'ignore')
            return hashlib.sha1(raw).hexdigest()
        except Exception:
            return None

    def _fmt_time_range(pair, sr):
        try:
            s, e = pair
            s = int(s); e = int(e); sr = float(sr)
            if sr <= 0 or e <= s:
                return None
            def mmss(t):
                m = int(t // 60)
                s = int(t % 60)
                return f"{m:02d}:{s:02d}"
            return f"{mmss(s/sr)}–{mmss(e/sr)}"
        except Exception:
            return None

    def _thr_pack_for(metric_key, value, r):
        thr = _policy_metric_thresholds(metric_key, value, pol, r=r)
        if thr is None:
            return None, None
        good = thr.get("good"); warn = thr.get("warn"); bad = thr.get("bad")
        hib = thr.get("higher_is_better", True)
        return thr, f"good={_fmt(good,2)} warn={_fmt(warn,2)} bad={_fmt(bad,2)} dir={'↑' if hib else '↓'}"

    def _hum_dense_flag(r):
        try:
            lbl = str(r.get("noise_spur_label") or "")
            hc = int(r.get("noise_spur_harmonics_count") or 0)
            return lbl.startswith("HUM") and hc >= 10
        except Exception:
            return False

    def _nf_interval_from_r(r):
        def _sf(x):
            try:
                f = float(x)
                return f if np.isfinite(f) else None
            except Exception:
                return None
        nf_bb = _sf(r.get('nf_broadband_dbfs'))
        nf_raw = _sf(r.get('noise_floor_raw_dbfs'))
        nf_cap = _sf(r.get('noise_floor_dbfs'))
        center = nf_bb if nf_bb is not None else (nf_raw if nf_raw is not None else nf_cap)
        if center is None:
            return None
        nwin = int(r.get("noise_windows_count") or 0)
        ndur = _sf(r.get("noise_total_duration_sec")) or 0.0
        diff = _sf(r.get("noise_consistency_diff_db")) or 0.0
        nf_cross = _sf(r.get("noise_floor_cross_rms_dbfs"))
        nf_cap_appl = bool(r.get("noise_floor_sanity_applied"))
        if (nwin >= 3 and ndur >= 3.0 and diff <= 6.0):
            unc = 2.0; conf = "Alta"
        elif (nwin >= 2 and ndur >= 2.0 and diff <= 10.0):
            unc = 3.0; conf = "Media"
        elif diff <= 20.0:
            unc = 4.0; conf = "Media"
        elif diff <= 30.0:
            unc = 5.0; conf = "Bassa"
        else:
            unc = 6.0; conf = "Bassa"
        if nf_cross is not None:
            delta = abs(center - nf_cross)
            if delta > unc:
                unc = min(8.0, delta + 1.0)
        if nf_cap_appl:
            unc = max(unc, 4.0)
        low = center - unc
        high = center + unc
        return (low, high, center, unc, conf)

    def _sf(x, d=0.0):
        try:
            f = float(x)
            return f if np.isfinite(f) else d
        except Exception:
            return d

    def _conf_from_metrics(r0, r1):
        def _f(x):
            try:
                v = float(x)
                return v if np.isfinite(v) else None
            except Exception:
                return None
        spur0, spur1 = _f(r0.get('noise_spur_db')), _f(r1.get('noise_spur_db'))
        isp0,  isp1  = _f(r0.get('isp_margin_db')), _f(r1.get('isp_margin_db'))
        dc0,   dc1   = _f(r0.get('dc_offset_dbfs')), _f(r1.get('dc_offset_dbfs'))

        def adiff(a, b):
            if a is None or b is None:
                return None
            return abs(a - b)

        d_spur = adiff(spur0, spur1)
        d_isp  = adiff(isp0,  isp1)
        d_dc   = adiff(dc0,   dc1)

        strong = ((d_spur is not None and d_spur >= 3.0) or
                  (d_isp  is not None and d_isp  >= 0.15) or
                  (d_dc   is not None and d_dc   >= 3.0))
        weak   = ((d_spur is not None and d_spur >= 1.0) or
                  (d_isp  is not None and d_isp  >= 0.05) or
                  (d_dc   is not None and d_dc   >= 1.0))

        if strong:
            return "Alta (tie-metriche)"
        if weak:
            return "Media (tie-metriche)"
        return "Bassa (tie-metriche)"

    def _conf_debug_top(ordered_list, tie_mode_local):
        if not ordered_list or len(ordered_list) < 2:
            return "N/A"
        a, b = ordered_list[0], ordered_list[1]
        s0 = _sf(a.get('score_float'), _sf(a.get('score')))
        s1 = _sf(b.get('score_float'), _sf(b.get('score')))

        if tie_mode_local in ("none", "off", "0"):
            if s0 > s1:
                return "Alta (score)"
            return _conf_from_metrics(a, b)

        resid = a.get('pair_residual_to_next_db')
        try:
            if isinstance(resid, (int, float)) and np.isfinite(resid):
                if resid <= -3.0:
                    return "Alta (residuo)"
                if resid <= -1.5:
                    return "Media (residuo)"
                return "Bassa (residuo)"
        except Exception:
            pass
        if s0 > s1:
            return "Alta (score)"
        return _conf_from_metrics(a, b)

    try:
        np_ver = getattr(np, "__version__", "N/A")
    except Exception:
        np_ver = "N/A"
    try:
        import scipy
        sp_ver = getattr(scipy, "__version__", "N/A")
    except Exception:
        sp_ver = "N/A"
    try:
        import pyloudnorm as pyln
        pl_ver = getattr(pyln, "__version__", "N/A")
    except Exception:
        pl_ver = "N/A"
    try:
        out_ff = subprocess.run(['ffmpeg', '-version'], check=False, capture_output=True, text=True, encoding="utf-8", errors="replace")
        ffm_head = (out_ff.stdout or out_ff.stderr or "").splitlines()[0] if (out_ff.stdout or out_ff.stderr) else ""
    except Exception:
        ffm_head = ""
    try:
        out_fp = subprocess.run(['ffprobe', '-version'], check=False, capture_output=True, text=True, encoding="utf-8", errors="replace")
        ffp_head = (out_fp.stdout or out_fp.stderr) or ""
        ffp_head = ffp_head.splitlines()[0] if ffp_head else ""
    except Exception:
        ffp_head = ""
    env_keys = ["MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","VECLIB_MAXIMUM_THREADS","ACCELERATE_NTHREADS","NUMBA_NUM_THREADS","MKL_DYNAMIC","OMP_DYNAMIC","FFMPEG_THREADS"]
    env_dump = {k: os.environ.get(k) for k in env_keys}
    cpu_cnt = os.cpu_count() or 1
    eng_ver = None
    for r in results_list:
        ev = r.get("engine_version")
        if isinstance(ev, str) and ev.strip():
            eng_ver = ev.strip()
            break
    try:
        any_pyl = any(str(r.get("loudness_backend") or "").strip().lower() == "pyloudnorm" for r in results_list)
        if pl_ver == "N/A" and any_pyl:
            pl_ver = "active"
    except Exception:
        pass

    tie_mode = (os.environ.get("AQC_TIEBREAK", "none").strip().lower())
    dq_last  = (os.environ.get("AQC_RANK_DQ_LAST", "1").strip().lower() in ("1", "true", "yes", "on"))
    if tie_mode in ("none", "off", "0"):
        crit = "score (solo score)"
    elif tie_mode in ("residual", "residuo", "default"):
        crit = "score + tie-break (residuo)"
    else:
        crit = f"score + tie-break ({tie_mode})"
    crit += " | DQ in coda" if dq_last else " | DQ inclusi"

    lines = []
    lines.append("=== LOG DETTAGLIATO ANALISI AUDIO ===\n")
    lines.append(f"Data/Ora: {ts}\n")
    lines.append(f"Policy: {pol.get('version','N/A')}\n")
    lines.append(f"Engine: {eng_ver or 'N/A'}\n")
    lines.append(f"Python: {sys.version.split()[0]} | NumPy: {np_ver} | SciPy: {sp_ver} | pyloudnorm: {pl_ver}\n")
    lines.append(f"FFmpeg: {ffm_head or 'N/A'}\n")
    lines.append(f"FFprobe: {ffp_head or 'N/A'}\n")
    lines.append(f"Platform: {platform.platform()} | CPU: {cpu_cnt}\n")
    lines.append(f"Criterio classifica: {crit}\n")
    lines.append("Thread/Env:\n")
    for k in env_keys:
        lines.append(f" • {k}={env_dump.get(k) or 'N/A'}\n")
    lines.append(f"\nTotale file analizzati: {len(results_list)}\n")

    lines.append("\nEsclusi per clipping:\n")
    excl_list = excluded_clipping if excluded_clipping else []
    if excl_list:
        for e in excl_list:
            if isinstance(e, dict):
                nm  = e.get('filename') or os.path.basename(e.get('filepath', ''))
                tp  = e.get('true_peak_est_dbtp', e.get('peak_dbfs_overall'))
                isp = e.get('isp_margin_db')
                lines.append(f" • {nm} | TP={_fmt(tp,2)} dBTP | ISP={_fmt(isp,2)} dB\n")
            else:
                lines.append(f" • {str(e)}\n")
    else:
        lines.append(" • Nessuno\n")

    lines.append("\nDuplicati raggruppati (MD5):\n")
    if isinstance(md5_groups, dict) and md5_groups:
        any_group = False
        for md5_hash, paths in md5_groups.items():
            if not isinstance(paths, (list, tuple)) or len(paths) <= 1:
                continue
            any_group = True
            rep = None
            if isinstance(md5_representatives, dict):
                rep = md5_representatives.get(md5_hash)
            rep = rep or (paths[0] if paths else None)
            rep_name = os.path.basename(rep) if rep else "(sconosciuto)"
            others = [os.path.basename(p) for p in paths if p != rep]
            short_md5 = (md5_hash[:8] + "...") if isinstance(md5_hash, str) and len(md5_hash) > 8 else str(md5_hash)
            if others:
                preview = ", ".join(others[:2]) + ("..." if len(others) > 2 else "")
                lines.append(f" • {short_md5}: {rep_name} (rappresentante) + {len(others)} duplicati: {preview}\n")
            else:
                lines.append(f" • {short_md5}: {rep_name} (rappresentante)\n")
        if not any_group:
            lines.append(" • Nessun duplicato\n")
    else:
        lines.append(" • Nessun duplicato\n")

    try:
        sig_groups = {}
        for r in results_list:
            sig = _sig_tuple(r)
            if sig is None:
                continue
            sid = _sig_id(sig) or "NA"
            sig_groups.setdefault(sid, []).append(r)
        uniq_sig = len(sig_groups)
        dup_sig = sum(1 for s, lst in sig_groups.items() if len(lst) > 1)
        lines.append(f"\nSignature tecniche uniche: {uniq_sig} (gruppi multi-file: {dup_sig})\n")
        if dup_sig > 0:
            for sid, lst in sig_groups.items():
                if len(lst) <= 1:
                    continue
                names = ", ".join([os.path.basename(x.get('_original_ref_path', x.get('filepath',''))) for x in lst])
                lines.append(f" • SigID {sid[:10]}… → {len(lst)} file: {names}\n")
    except Exception:
        lines.append("\nSignature tecniche uniche: N/A\n")

    lines.append("\nIndice:\n")
    index_list = list(results_list)
    for i, r in enumerate(index_list, 1):
        lines.append(f" {i}) {r['filename']:<50} {r.get('score_float', 0):>6.2f}/100\n")
    lines.append("\n")

    for i, r in enumerate(index_list, 1):
        lines.append("-" * 60 + "\n")
        lines.append(f"File {i}: {r['filename']}\n")
        full_path = r.get('filepath') or ""
        lines.append(f"Percorso: {full_path}\n")
        try:
            if full_path and os.path.exists(full_path):
                sz = os.path.getsize(full_path)
                mt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(full_path)))
                lines.append(f"File size: {sz} B | MTime: {mt}\n")
        except Exception:
            pass
        md5h = _md5_for_result(r)
        if md5h:
            lines.append(f"MD5 audio: {md5h}\n")
        sig = _sig_tuple(r)
        sid = _sig_id(sig) if sig is not None else None
        if sid:
            lines.append(f"SigID: {sid}\n")

        sc   = r.get('score_float', 0.0)
        pre  = r.get('score_pre_cap', sc)
        cap  = r.get('hygiene_cap', 100.0)
        clip = r.get('clipping_class', 'clean')
        H    = r.get('hygiene_score', None)
        Q    = r.get('quality_score', None)
        eval_bits = r.get('hygiene_eval_bits', 16)
        conf = _conf_label(r)
        mix_eff = r.get("score_mix_eff")
        wH_eff = r.get("weights_H_eff")
        wQ_eff = r.get("weights_Q_eff")
        fsc  = r.get("fine_score", None)
        tieb = r.get("tie_bonus", None)

        extras = []
        extras.append(f"pre {pre:.2f}")
        if isinstance(fsc, (int, float)) and np.isfinite(fsc):
            extras.append(f"fine {_fmt(fsc,2)}")
        if isinstance(tieb, (int, float)) and np.isfinite(tieb) and abs(tieb) > 0:
            extras.append(f"tie {_fmt_signed(tieb,2)}")
        extras_str = ", ".join(extras)
        lines.append(f"Score globale: {sc:.2f} / 100 ({extras_str})\n")
        lines.append(f"Clipping: {clip} | Conf rumore: {conf} | Bit valutazione igiene: {eval_bits}\n")
        if isinstance(mix_eff, dict):
            lines.append(f"Mix score effettivo: H={_fmt(mix_eff.get('H'),2)} Q={_fmt(mix_eff.get('Q'),2)}\n")
        if isinstance(wH_eff, dict):
            lines.append(f"Pesi H eff.: NF={_fmt(wH_eff.get('noise_floor_dbfs'),2)} Spur={_fmt(wH_eff.get('noise_spur_db'),2)} ISP={_fmt(wH_eff.get('isp_margin_db'),2)} DC={_fmt(wH_eff.get('dc_offset_dbfs'),2)}\n")
        if isinstance(wQ_eff, dict):
            lines.append(f"Pesi Q eff.: DR={_fmt(wQ_eff.get('dr_tt_avg'),2)} PLR={_fmt(wQ_eff.get('plr_est'),2)} LRA={_fmt(wQ_eff.get('lra_est'),2)} ST-IQR={_fmt(wQ_eff.get('st_lufs_iqr_db'),2)} Tone={_fmt(wQ_eff.get('spectral_balance_dev_db'),2)} Width={_fmt(wQ_eff.get('stereo_width_iqr_db'),2)}\n")

        conv_cont = r.get("conversion_container")
        if conv_cont:
            conv_prof = r.get("conversion_profile")
            conv_sr = r.get("conversion_target_sr")
            conv_ok = r.get("conversion_verified")
            lines.append(f"Conversione PCM: {str(conv_cont).upper()} | profilo={conv_prof or 'N/A'} | target SR={_fmt(conv_sr,0)} Hz | verificato={'SÌ' if conv_ok else 'NO'}\n")

        if i == 1:
            conf_dbg = _conf_debug_top(index_list, tie_mode)
            lines.append(f"Conf. debug: {conf_dbg}\n")

        ns_mode = r.get("noise_selection_mode")
        notch_h = int(r.get("notch_harmonics") or 0)
        ns_mode_label = None
        if isinstance(ns_mode, str):
            ns_mode_label = ns_mode
            if ns_mode.strip().lower() == "strict+notch" and notch_h > 0:
                ns_mode_label = f"{ns_mode} (applied)"
            lines.append(f"Selettore rumore: {ns_mode_label} | finestre={int(r.get('noise_windows_count') or 0)} | durata={_fmt(r.get('noise_total_duration_sec'),2)} s\n")
            wins_preview = r.get("noise_windows_preview")
            if isinstance(wins_preview, (list, tuple)) and wins_preview:
                stamps = []
                for pair in wins_preview[:3]:
                    s = _fmt_time_range(pair, r.get('samplerate'))
                    if s:
                        stamps.append(s)
                if stamps:
                    lines.append(f" • Finestre (anteprima): {', '.join(stamps)}\n")

        unk = r.get("hygiene_unknown_count")
        if isinstance(unk,(int,float)):
            lines.append(f"Metriche igiene sconosciute: {int(unk)}\n")
        lines.append(f"Componenti: H={_fmt(H,3)} Q={_fmt(Q,3)}\n")

        g_next = r.get('pair_gain_to_next_db')
        r_next = r.get('pair_residual_to_next_db')
        d_next = r.get('pair_drift_to_next_ppm')
        sm_next = r.get('pair_same_master_next')
        sim_next = r.get('pair_similarity')
        rL = r.get('pair_residual_low_db')
        rM = r.get('pair_residual_mid_db')
        rH = r.get('pair_residual_high_db')
        if isinstance(r_next, (int, float)) and np.isfinite(r_next) and sm_next is None:
            sm_next = bool(r_next <= -35.0)
        if d_next is None or not isinstance(d_next, (int, float)) or not np.isfinite(d_next):
            d_next = 0.0
        if (g_next is not None) or (r_next is not None) or (sm_next is not None) or (d_next is not None):
            lines.append("\nConfronto con successivo in classifica:\n")
            lines.append(f" • Offset gain ottimale: {_fmt(g_next,2)} dB\n")
            lines.append(f" • Residuo relativo: {_fmt(r_next,1)} dBFS\n")
            lines.append(f" • Drift: {_fmt(d_next,1)} ppm\n")
            lines.append(f" • Stessa master (solo volume): {'SÌ' if bool(sm_next) else 'NO'}\n")
            if isinstance(sim_next, (int, float)) and np.isfinite(sim_next):
                lines.append(f" • Similarità: {_fmt(sim_next,3)}\n")
            if any(isinstance(v, (int, float)) for v in (rL, rM, rH)):
                lines.append(f" • Residuo bande (dB): Low={_fmt(rL,1)} | Mid={_fmt(rM,1)} | High={_fmt(rH,1)}\n")
        g_prev = r.get('pair_gain_to_prev_db')
        r_prev = r.get('pair_residual_to_prev_db')
        d_prev = r.get('pair_drift_to_prev_ppm')
        sm_prev = r.get('pair_same_master_prev')
        if isinstance(r_prev, (int, float)) and np.isfinite(r_prev) and sm_prev is None:
            sm_prev = bool(r_prev <= -35.0)
        if d_prev is None or not isinstance(d_prev, (int, float)) or not np.isfinite(d_prev):
            d_prev = 0.0
        if (g_prev is not None) or (r_prev is not None) or (sm_prev is not None) or (d_prev is not None):
            lines.append("\nConfronto con precedente in classifica:\n")
            lines.append(f" • Offset gain ottimale: {_fmt(g_prev,2)} dB\n")
            lines.append(f" • Residuo relativo: {_fmt(r_prev,1)} dBFS\n")
            lines.append(f" • Drift: {_fmt(d_prev,1)} ppm\n")
            lines.append(f" • Stessa master (solo volume): {'SÌ' if bool(sm_prev) else 'NO'}\n")

        lines.append("-" * 60 + "\n\n")
        lines.append("Sezione Pulizia (contributi a H):\n")
        contrib = _contrib_map(r)
        nf = r.get('noise_floor_dbfs')
        nf_raw = r.get('noise_floor_raw_dbfs')
        nf_bb = r.get('nf_broadband_dbfs') if 'nf_broadband_dbfs' in r else r.get('noise_floor_dbfs')
        nwin = int(r.get("noise_windows_count") or 0)
        ndur = float(r.get("noise_total_duration_sec") or 0.0)
        diff = float(r.get("noise_consistency_diff_db") or 0.0)
        extreme_uncertain = (diff > 30.0) or (nwin < 2 and ndur < 3.0)
        moderate_uncertain = (diff > 20.0) or (nwin < 2 or ndur < 3.0)
        nf_cap_applied = bool(r.get("noise_floor_sanity_applied"))
        hum_dense = _hum_dense_flag(r)
        nf_label = "NF cap (guardrail)" if (nf_cap_applied or extreme_uncertain) else "Noise floor"
        lines.append(f" • {nf_label}: {_fmt(nf,1)} dBFS{_status('noise_floor_dbfs', nf, r)} -> {_fmt(contrib.get('noise_floor_dbfs',{}).get('punti'),2)} pt\n")
        lines.append(f"   NF raw/BB:   {_fmt(nf_raw,1)} / {_fmt(nf_bb,1)} dBFS\n")
        nf_mode = contrib.get('noise_floor_dbfs', {}).get('mode') or r.get('nf_scoring_mode')
        if nf_mode:
            if nf_mode == "SAFE":
                mode_label = "SAFE (max(16,24))"
            elif nf_mode == "PROFILE_24":
                mode_label = "PROFILE_24"
            elif nf_mode == "PROFILE_16":
                mode_label = "PROFILE_16"
            elif nf_mode == "EXCLUDED":
                mode_label = "EXCLUDED"
            else:
                mode_label = str(nf_mode)
            lines.append(f"   NF scoring mode: {mode_label}\n")
        if extreme_uncertain:
            lines.append(f"   NF non usato per score (escluso)\n")
        else:
            if hum_dense:
                lines.append(f"   NF usato per score: {_fmt(nf_bb,1)} dBFS (BB)\n")
            else:
                lines.append(f"   NF usato per score: {_fmt(nf,1)} dBFS\n")
        if extreme_uncertain:
            lines.append(f"   NF scoring: ESCLUSO (Δ={_fmt(diff,2)} dB, finestre={nwin}, durata={_fmt(ndur,2)} s)\n")
        elif moderate_uncertain:
            lines.append(f"   NF scoring: PESO RIDOTTO (Δ={_fmt(diff,2)} dB, finestre={nwin}, durata={_fmt(ndur,2)} s)\n")
        intrv = _nf_interval_from_r(r)
        if intrv is not None:
            low, high, center, unc, confI = intrv
            lines.append(f"   NF intervallo credibile: [{_fmt(low,1)}, {_fmt(high,1)}] dBFS (±{_fmt(unc,1)}, conf={confI})\n")
        spur = r.get('noise_spur_db')
        lines.append(f" • Noise spur:  {_fmt(spur,1)} dB{_status('noise_spur_db', spur, r)} -> {_fmt(contrib.get('noise_spur_db',{}).get('punti'),2)} pt\n")
        isp = r.get('isp_margin_db')
        lines.append(f" • ISP margin:  {_fmt(isp,2)} dB{_status('isp_margin_db', isp, r)} -> {_fmt(contrib.get('isp_margin_db',{}).get('punti'),2)} pt\n")
        dc = r.get('dc_offset_dbfs')
        lines.append(f" • DC offset:   {_fmt_dc(dc)} dBFS{_status('dc_offset_dbfs', dc, r)} -> {_fmt(contrib.get('dc_offset_dbfs',{}).get('punti'),2)} pt\n")
        nf_cr = r.get("noise_floor_cross_rms_dbfs")
        nf_df = r.get("noise_consistency_diff_db")
        nf_p90 = r.get("noise_floor_cross_p90_dbfs")
        nf_cap = r.get("noise_floor_sanity_cap_dbfs")
        nf_ap = r.get("noise_floor_sanity_applied")
        if nf_cr is not None or nf_df is not None:
            lines.append(f" • NF cross:    {_fmt(nf_cr,1)} dBFS | Δ={_fmt(nf_df,2)} dB\n")
        if nf_p90 is not None or nf_cap is not None or nf_ap is not None:
            reason = r.get("noise_floor_cap_reason")
            lines.append(f" • NF sanity:   p90={_fmt(nf_p90,1)} dBFS | cap={_fmt(nf_cap,1)} dBFS | applied={'SÌ' if bool(nf_ap) else 'NO'}{(' | reason='+str(reason)) if reason else ''}\n")
        notch_f0 = r.get("notch_f0")
        nf_red = r.get("nf_notch_reduction_db")
        if notch_f0 or nf_red:
            lines.append(f" • HUM/notch:   f0={_fmt(notch_f0,1)} Hz | ΔBB={_fmt(nf_red,2)} dB\n")
        lbl = r.get("noise_spur_label")
        f0  = r.get("noise_spur_fundamental_hz")
        hc  = r.get("noise_spur_harmonics_count")
        if lbl or f0 or hc:
            f0s = f"{_fmt(f0,1)} Hz" if isinstance(f0, (int,float)) else "N/A"
            hcs = str(hc) if isinstance(hc, (int, float)) else "0"
            lines.append(f" • Spur meta:   {lbl or 'N/A'} | f0={f0s} | H={hcs}\n")
        lines.append("\nSezione Qualità (contributi a Q):\n")
        lines.append(f" • DR (TT):     {_fmt(r.get('dr_tt_avg'),1)} dB{_status('dr_tt_avg', r.get('dr_tt_avg'), r)} -> {_fmt(contrib.get('dr_tt_avg',{}).get('punti'),2)} pt\n")
        plr_int = r.get('plr_est')
        plr_act = r.get("plr_active_db")
        plr_eff = r.get("plr_effective_db", plr_int if plr_int is not None else plr_act)
        lines.append(f" • PLR (eff):   {_fmt(plr_eff,1)} dB{_status('plr_est', plr_eff, r)} -> {_fmt(contrib.get('plr_effective_db',{}).get('punti'),2)} pt\n")
        if plr_int is not None or plr_act is not None:
            lines.append(f"   PLR integr./attivo: {_fmt(plr_int,1)} / {_fmt(plr_act,1)} dB\n")
        lines.append(f" • LRA:         {_fmt(r.get('lra_est'),1)} LU{_status('lra_est', r.get('lra_est'), r)} -> {_fmt(contrib.get('lra_est',{}).get('punti'),2)} pt\n")
        lines.append(f" • ST IQR:      {_fmt(r.get('st_lufs_iqr_db'),1)} dB{_status('st_lufs_iqr_db', r.get('st_lufs_iqr_db'), r)} -> {_fmt(contrib.get('st_lufs_iqr_db',{}).get('punti'),2)} pt\n")
        lines.append(f" • ToneBal:     {_fmt(r.get('spectral_balance_dev_db'),1)} dB{_status('spectral_balance_dev_db', r.get('spectral_balance_dev_db'), r)} -> {_fmt(contrib.get('spectral_balance_dev_db',{}).get('punti'),2)} pt\n")
        lines.append(f" • Width IQR:   {_fmt(r.get('stereo_width_iqr_db'),1)} dB{_status('stereo_width_iqr_db', r.get('stereo_width_iqr_db'), r)} -> {_fmt(contrib.get('stereo_width_iqr_db',{}).get('punti'),2)} pt\n")
        lines.append("\nAltre metriche e diagnostica:\n")
        tpv = r.get('true_peak_est_dbtp', r.get('peak_dbfs_overall'))
        lines.append(f" • True Peak:   {_fmt(tpv,2)} dBTP\n")
        lines.append(f" • Clipping:    {('SÌ' if r.get('clipping_detected') else 'NO') if r.get('clipping_detected') is not None else 'N/A'}\n")
        lines.append(f" • LUFS:        {_fmt(r.get('loudness_lufs'),1)} LUFS (backend: {r.get('loudness_backend','N/A')})\n")
        lines.append(f" • LRA:         {_fmt(r.get('lra_est'),1)} LU\n")
        lines.append("\nSoglie applicate:\n")
        nf_key = _policy_select_nf_profile(r.get('noise_floor_dbfs'), r, pol)
        thr_nf = pol["thresholds"].get(nf_key)
        if thr_nf:
            lines.append(f" • Noise floor [{nf_key[-2:]}]: good={_fmt(thr_nf['good'],2)} warn={_fmt(thr_nf['warn'],2)} bad={_fmt(thr_nf['bad'],2)} dir=↓\n")
        for mkey, lbl in [
            ("noise_spur_db","Noise spur"),
            ("isp_margin_db","ISP margin"),
            ("dc_offset_dbfs","DC offset"),
            ("dr_tt_avg","DR (TT)"),
            ("plr_est","PLR"),
            ("lra_est","LRA"),
            ("st_lufs_iqr_db","ST IQR"),
            ("spectral_balance_dev_db","ToneBal"),
            ("stereo_width_iqr_db","Width IQR"),
        ]:
            val = r.get(mkey if mkey!="plr_est" else "plr_effective_db", r.get(mkey))
            thr, pack = _thr_pack_for("plr_est" if mkey=="plr_est" else mkey, val, r)
            if pack:
                lines.append(f" • {lbl}: {pack}\n")
        lines.append(f"\nAssessment: {r.get('assessment','')}\n\n")

    try:
        with open(fpath, "w", encoding="utf-8") as fh:
            fh.writelines(lines)
        print(f"Log dettagliato salvato in '{fname}'.")
    except OSError as e:
        print(f"Impossibile scrivere il log: {e}")

def run_cli():
    import os
    try:
        auto = os.environ.get("AQC_AUTO_WORKERS", "").strip().lower()
        if auto == "":
            os.environ["AQC_AUTO_WORKERS"] = "auto"
    except Exception:
        pass
    try:
        sorted_representative_results, winner_rep_result = main_entry()
        return sorted_representative_results, winner_rep_result
    finally:
        wait_for_user_exit()

def run_analysis_jobs(files_to_analyze_fully_info):
    import os
    representative_analysis_results = []
    ordered_jobs = order_jobs_by_weight(list(files_to_analyze_fully_info or []))
    num_workers_requested = prompt_num_workers(files_meta=ordered_jobs)
    workers, threads_per_worker = configure_threading(num_workers_requested)
    max_cpu = os.cpu_count() or 1
    if workers > 1 and len(ordered_jobs) > 1:
        try:
            import concurrent.futures
            print(f"Avvio analisi parallela con {workers} worker, {threads_per_worker} thread/worker (CPU: {max_cpu})...")
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
                future_to_job = {}
                for args in ordered_jobs:
                    path_for_analysis, original_ref_path, is_temp_for_call = args
                    print(f"Avvio analisi: '{os.path.basename(path_for_analysis)}' (da originale: '{os.path.basename(original_ref_path)}')")
                    fut = ex.submit(_analyze_wrapper, args)
                    future_to_job[fut] = args
                for fut in concurrent.futures.as_completed(future_to_job):
                    args = future_to_job[fut]
                    path_for_analysis, original_ref_path, is_temp_for_call = args
                    try:
                        res = fut.result()
                        if res:
                            representative_analysis_results.append(res)
                            print(f"Completata: '{os.path.basename(path_for_analysis)}' (Score: {res.get('score_float','N/A')}/100)")
                        else:
                            print(f"Completata: '{os.path.basename(path_for_analysis)}' (nessun risultato)")
                    except Exception as e:
                        print(f"Errore analisi '{os.path.basename(path_for_analysis)}': {type(e).__name__}: {e}")
        except Exception as e:
            print(f"Analisi parallela non disponibile/errore ({type(e).__name__}: {e}). Esecuzione in modalità sequenziale.")
            for path_for_analysis, original_ref_path, is_temp_for_call in ordered_jobs:
                print(f"Avvio analisi: '{os.path.basename(path_for_analysis)}' (da originale: '{os.path.basename(original_ref_path)}')")
                analysis_result_dict = analyze_audio(path_for_analysis, is_temporary=is_temp_for_call)
                if analysis_result_dict:
                    analysis_result_dict['_original_ref_path'] = original_ref_path
                    representative_analysis_results.append(analysis_result_dict)
                    print(f"Completata: '{os.path.basename(path_for_analysis)}' (Score: {analysis_result_dict.get('score_float','N/A')}/100)")
                else:
                    print(f"Completata: '{os.path.basename(path_for_analysis)}' (nessun risultato)")
    else:
        print(f"Avvio analisi sequenziale con 1 worker (CPU: {max_cpu})...")
        for path_for_analysis, original_ref_path, is_temp_for_call in ordered_jobs:
            print(f"Avvio analisi: '{os.path.basename(path_for_analysis)}' (da originale: '{os.path.basename(original_ref_path)}')")
            analysis_result_dict = analyze_audio(path_for_analysis, is_temporary=is_temp_for_call)
            if analysis_result_dict:
                analysis_result_dict['_original_ref_path'] = original_ref_path
                representative_analysis_results.append(analysis_result_dict)
                print(f"Completata: '{os.path.basename(path_for_analysis)}' (Score: {analysis_result_dict.get('score_float','N/A')}/100)")
            else:
                print(f"Completata: '{os.path.basename(path_for_analysis)}' (nessun risultato)")
    return representative_analysis_results, ordered_jobs, workers, threads_per_worker

def perform_full_analysis_and_reporting(files_to_analyze_fully_info, md5_audio_groups, md5_representatives, temp_files_registry, all_temp_files_created, temp_files_to_keep, pol):
    import os, sys
    print(f"\n--- Fase 2: Analisi Dettagliata ({len(files_to_analyze_fully_info)} file/gruppi unici) ---")
    representative_analysis_results, ordered_jobs, workers, threads_per_worker = run_analysis_jobs(files_to_analyze_fully_info)
    if not representative_analysis_results:
        print("\nNessun file valido analizzato con successo.")
        for temp_f_path in set(all_temp_files_created):
            if temp_f_path not in temp_files_to_keep and os.path.exists(temp_f_path):
                try:
                    os.remove(temp_f_path)
                    print(f"Pulito temp: {os.path.basename(temp_f_path)}")
                except OSError as e_clean:
                    print(f"Impossibile eliminare temp '{os.path.basename(temp_f_path)}': {e_clean}")
        sys.exit(0)
    technical_signature_groups = {}
    for res_dict in representative_analysis_results:
        try:
            tech_sig = make_signature(res_dict)
            if tech_sig not in technical_signature_groups:
                technical_signature_groups[tech_sig] = []
            technical_signature_groups[tech_sig].append(res_dict)
        except Exception:
            continue
    for tech_group_list in technical_signature_groups.values():
        if not tech_group_list:
            continue
        try:
            shared_score_float = float(np.mean([g_res['score_float'] for g_res in tech_group_list]))
        except Exception:
            shared_score_float = None
        if shared_score_float is not None and np.isfinite(shared_score_float):
            for g_res in tech_group_list:
                g_res['score_float'] = shared_score_float
                g_res['score'] = int(round(shared_score_float))
    sorted_representative_results, winner_rep_result, ranking_mode = rank_results_hygiene_first(representative_analysis_results)
    print("\n--- Report Dettagliato ---")
    print("(✅ Ottimo, ⚠️ Attenzione, ❌ Critico)")
    print_console_report(sorted_representative_results, policy=pol)
    announce_winner(winner_rep_result, md5_audio_groups, temp_files_registry)
    move_winner_files(winner_rep_result, md5_audio_groups, temp_files_registry, temp_files_to_keep)
    try:
        log_choice = input("\nVuoi salvare un log tecnico dettagliato? [y/N]: ").strip().lower()
    except Exception:
        log_choice = "n"
    if log_choice in ("y", "yes", "s", "si"):
        log_dir = os.path.dirname(__file__) if '__file__' in globals() else os.getcwd()
        excluded_clipping = [r for r in representative_analysis_results if r.get('clipping_class') in ('borderline', 'hard')]
        write_human_log(
            sorted_representative_results,
            path_dir=log_dir,
            excluded_clipping=excluded_clipping,
            md5_groups=md5_audio_groups,
            md5_representatives=md5_representatives,
            policy=pol
        )
    print("\n--- Pulizia file temporanei ---")
    deleted_final_count = 0
    kept_final_count = 0
    unique_temps_for_cleanup = set(all_temp_files_created)
    for temp_path_to_clean in unique_temps_for_cleanup:
        if temp_path_to_clean in temp_files_to_keep:
            print(f"Conservato (vincitore/duplicato spostato): {os.path.basename(temp_path_to_clean)}")
            kept_final_count += 1
        else:
            try:
                if os.path.exists(temp_path_to_clean):
                    os.remove(temp_path_to_clean)
                    print(f"Eliminato: {os.path.basename(temp_path_to_clean)}")
                    deleted_final_count += 1
            except OSError as e_final_clean:
                print(f"Impossibile eliminare temporaneo '{os.path.basename(temp_path_to_clean)}': {e_final_clean}")
    if not unique_temps_for_cleanup:
        print("Nessun file temporaneo creato o da pulire.")
    else:
        print(f"Pulizia completata: {deleted_final_count} file eliminati, {kept_final_count} conservati.")
    print("\nAnalisi completata.")
    return sorted_representative_results, winner_rep_result

def effective_band_edge_db(spectrum_db, freqs, noise_margin=20):
    if spectrum_db is None or freqs is None:
        return None
    if not isinstance(spectrum_db, np.ndarray) or not isinstance(freqs, np.ndarray):
        return None
    if spectrum_db.size != freqs.size or freqs.size < 16:
        return None
    spec = np.asarray(spectrum_db, dtype=np.float64)
    frq = np.asarray(freqs, dtype=np.float64)
    finite = np.isfinite(spec) & np.isfinite(frq)
    if not np.any(finite):
        return None
    spec = spec[finite]
    frq = frq[finite]
    if spec.size < 16:
        return None
    n = spec.size
    from scipy.signal import fftconvolve
    k = max(11, int(round(n * 0.005)))
    if k % 2 == 0:
        k += 1
    if k >= n:
        k = max(11, n // 3 if (n // 3) % 2 == 1 else max(11, n // 3 - 1))
    ker = np.ones(k, dtype=np.float64) / k
    smoothed = fftconvolve(spec, ker, mode='same')
    nyq = float(frq[-1])
    if not np.isfinite(nyq) or nyq <= 0:
        return None
    fb_lo = 20.0
    fb_hi = min(20000.0, 0.98 * nyq)
    band_mask = (frq >= fb_lo) & (frq <= fb_hi)
    if not np.any(band_mask):
        return 0.0
    frq_b = frq[band_mask]
    sm_b = smoothed[band_mask]
    hi_band_mask = frq_b >= max(fb_lo, 0.85 * nyq)
    if np.sum(hi_band_mask) < 8:
        hi_band_mask = np.zeros_like(frq_b, dtype=bool)
        hi_band_mask[int(frq_b.size * 0.9):] = True
    hi_vals = sm_b[hi_band_mask]
    if hi_vals.size < 4:
        return 0.0
    nf_db = float(np.percentile(hi_vals, 30))
    if not np.isfinite(nf_db) or nf_db > -5:
        return 0.0
    margin = float(np.clip(noise_margin, 8.0, 30.0))
    thr_db = nf_db + margin
    above = sm_b >= thr_db
    if not np.any(above):
        return 0.0
    bin_hz = float(frq_b[1] - frq_b[0]) if frq_b.size > 1 else 0.0
    min_width_hz = 500.0
    min_bins = 3 if bin_hz <= 0 else max(3, int(round(min_width_hz / bin_hz)))
    last_run_end = None
    i = 0
    m = above.size
    while i < m:
        if above[i]:
            j = i
            while j + 1 < m and above[j + 1]:
                j += 1
            if (j - i + 1) >= min_bins:
                last_run_end = j
            i = j + 1
        else:
            i += 1
    if last_run_end is None:
        idx_true = np.where(above)[0]
        if idx_true.size == 0:
            return 0.0
        last_run_end = int(idx_true[-1])
    if last_run_end >= m - 1:
        edge = float(frq_b[-1])
        return min(edge, nyq)
    i0 = last_run_end
    i1 = last_run_end + 1
    y0 = float(sm_b[i0] - thr_db)
    y1 = float(sm_b[i1] - thr_db)
    if not np.isfinite(y0) or not np.isfinite(y1) or (y0 - y1) == 0:
        edge = float(frq_b[i0])
        return min(edge, nyq)
    t = float(np.clip(y0 / (y0 - y1), 0.0, 1.0))
    f0 = float(frq_b[i0])
    f1 = float(frq_b[i1])
    edge = f0 + t * (f1 - f0)
    if not np.isfinite(edge):
        return None
    return min(max(edge, 0.0), nyq)

def print_console_report(results_list, policy=None):
    pol = _policy_resolve(policy)
    if not results_list:
        print("\nNessun risultato da mostrare.")
        return

    import os

    def _fmt(v, dec=1, na="N/A"):
        try:
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                return na
            return f"{float(v):.{dec}f}"
        except Exception:
            return na

    def _fmt_signed(v, dec=2, na="N/A"):
        try:
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                return na
            f = float(v)
            sign = "+" if f >= 0 else ""
            return f"{sign}{f:.{dec}f}"
        except Exception:
            return na

    def _status(metric_key, value, r):
        return get_metric_status(metric_key, value, policy=pol, r=r) or ""

    def _hum_dense_flag(r):
        try:
            lbl = str(r.get("noise_spur_label") or "")
            hc = int(r.get("noise_spur_harmonics_count") or 0)
            return lbl.startswith("HUM") and hc >= 10
        except Exception:
            return False

    def _sf(x, d=0.0):
        try:
            f = float(x)
            return f if np.isfinite(f) else d
        except Exception:
            return d

    tie_mode = (os.environ.get("AQC_TIEBREAK", "none").strip().lower())
    dq_last  = (os.environ.get("AQC_RANK_DQ_LAST", "1").strip().lower() in ("1", "true", "yes", "on"))
    if tie_mode in ("none", "off", "0"):
        crit = "score (solo score)"
    elif tie_mode in ("residual", "residuo", "default"):
        crit = "score + tie-break (residuo)"
    else:
        crit = f"score + tie-break ({tie_mode})"
    crit += " | DQ in coda" if dq_last else " | DQ inclusi"

    ordered = list(results_list)

    def _conf_from_metrics(r0, r1):
        def _f(x):
            try:
                v = float(x)
                return v if np.isfinite(v) else None
            except Exception:
                return None
        spur0, spur1 = _f(r0.get('noise_spur_db')), _f(r1.get('noise_spur_db'))
        isp0,  isp1  = _f(r0.get('isp_margin_db')), _f(r1.get('isp_margin_db'))
        dc0,   dc1   = _f(r0.get('dc_offset_dbfs')), _f(r1.get('dc_offset_dbfs'))

        def adiff(a, b):
            if a is None or b is None:
                return None
            return abs(a - b)

        d_spur = adiff(spur0, spur1)
        d_isp  = adiff(isp0,  isp1)
        d_dc   = adiff(dc0,   dc1)

        strong = ((d_spur is not None and d_spur >= 3.0) or
                  (d_isp  is not None and d_isp  >= 0.15) or
                  (d_dc   is not None and d_dc   >= 3.0))
        weak   = ((d_spur is not None and d_spur >= 1.0) or
                  (d_isp  is not None and d_isp  >= 0.05) or
                  (d_dc   is not None and d_dc   >= 1.0))

        if strong:
            return "Alta (tie-metriche)"
        if weak:
            return "Media (tie-metriche)"
        return "Bassa (tie-metriche)"

    def _conf_debug_top(ordered_list):
        if not ordered_list or len(ordered_list) < 2:
            return "N/A"
        a, b = ordered_list[0], ordered_list[1]
        s0 = _sf(a.get('score_float'), _sf(a.get('score')))
        s1 = _sf(b.get('score_float'), _sf(b.get('score')))

        if tie_mode in ("none", "off", "0"):
            if s0 > s1:
                return "Alta (score)"
            return _conf_from_metrics(a, b)

        resid = a.get('pair_residual_to_next_db')
        sim   = a.get('pair_similarity')
        try:
            if isinstance(resid, (int, float)) and np.isfinite(resid):
                if resid <= -3.0:
                    return "Alta (residuo)"
                if resid <= -1.5:
                    return "Media (residuo)"
                return "Bassa (residuo)"
        except Exception:
            pass
        if s0 > s1:
            return "Alta (score)"
        return _conf_from_metrics(a, b)

    print(f"\nCriterio classifica: {crit}")
    print("\nIndice (ordine finale):")
    for i, r in enumerate(ordered, 1):
        name = r.get('filename') or os.path.basename(r.get('filepath','')) or "(sconosciuto)"
        score = _sf(r.get('score_float'), _sf(r.get('score')))
        print(f" {i}) {name:<50} {score:>6.2f}/100")
    print()

    for i, r in enumerate(ordered, 1):
        print("-" * 60)
        print(f"File {i}: {r.get('filename')}")
        full_path = r.get('filepath') or ""
        print(f"Percorso: {full_path}")

        sc   = r.get('score_float', 0.0)
        pre  = r.get('score_pre_cap', sc)
        clip = r.get('clipping_class', 'clean')
        H    = r.get('hygiene_score')
        Q    = r.get('quality_score')
        fsc  = r.get('fine_score')
        tieb = r.get('tie_bonus')
        eval_bits = r.get('hygiene_eval_bits', 16)
        conf = (r.get("noise_confidence") or "Alta").strip()

        extras = [f"pre {pre:.2f}"]
        if isinstance(fsc, (int, float)) and np.isfinite(fsc):
            extras.append(f"fine {_fmt(fsc,2)}")
        if isinstance(tieb, (int, float)) and np.isfinite(tieb) and abs(tieb) > 0:
            extras.append(f"tie {_fmt_signed(tieb,2)}")
        extras_str = ", ".join(extras)
        print(f"Score: {sc:.2f}/100 ({extras_str}) | Clipping: {clip} | Conf rumore: {conf} | Bit igiene: {eval_bits}")

        if i == 1:
            print(f"Conf. debug: {_conf_debug_top(ordered)}")

        conv_cont = r.get("conversion_container")
        if conv_cont:
            conv_prof = r.get("conversion_profile")
            conv_sr = r.get("conversion_target_sr")
            conv_ok = r.get("conversion_verified")
            print(f"Conversione PCM: {str(conv_cont).upper()} | profilo={conv_prof or 'N/A'} | target SR={_fmt(conv_sr,0)} Hz | verificato={'SÌ' if conv_ok else 'NO'}")

        nf = r.get('noise_floor_dbfs')
        nf_raw = r.get('noise_floor_raw_dbfs')
        nf_bb = r.get('nf_broadband_dbfs', r.get('noise_floor_dbfs'))
        nwin = int(r.get("noise_windows_count") or 0)
        ndur = float(r.get("noise_total_duration_sec") or 0.0)
        diff = float(r.get("noise_consistency_diff_db") or 0.0)
        extreme_uncertain = (diff > 30.0) or (nwin < 2 and ndur < 3.0)
        moderate_uncertain = (diff > 20.0) or (nwin < 2 or ndur < 3.0)
        nf_cap_applied = bool(r.get("noise_floor_sanity_applied"))
        hum_dense = _hum_dense_flag(r)

        nf_label = "NF cap (guardrail)" if (nf_cap_applied or extreme_uncertain) else "Noise floor"
        print(f" • {nf_label}: {_fmt(nf,1)} dBFS{_status('noise_floor_dbfs', nf, r)}")
        print(f"   NF raw/BB:   {_fmt(nf_raw,1)} / {_fmt(nf_bb,1)} dBFS")

        nf_mode = r.get("nf_scoring_mode")
        if nf_mode is None:
            safe_env = os.environ.get("AQC_NF_SAFE", "").strip().lower()
            nf_safe = (safe_env not in ("0", "false", "off", "no", "n"))
            if extreme_uncertain:
                nf_mode = "EXCLUDED"
            else:
                if nf_safe:
                    nf_mode = "SAFE"
                else:
                    key = _policy_select_nf_profile(nf, r, pol)
                    nf_mode = "PROFILE_24" if (key and str(key).endswith("_24")) else "PROFILE_16"
        if nf_mode == "SAFE":
            print("   NF scoring mode: SAFE (max(16,24))")
        elif nf_mode == "PROFILE_24":
            print("   NF scoring mode: PROFILE_24")
        elif nf_mode == "PROFILE_16":
            print("   NF scoring mode: PROFILE_16")
        elif nf_mode == "EXCLUDED":
            print("   NF scoring mode: EXCLUDED")
        else:
            print(f"   NF scoring mode: {nf_mode}")

        if extreme_uncertain:
            print("   NF non usato per score (escluso)")
        else:
            if hum_dense:
                print(f"   NF usato per score: {_fmt(nf_bb,1)} dBFS (BB)")
            else:
                print(f"   NF usato per score: {_fmt(nf,1)} dBFS")
        if extreme_uncertain:
            print(f"   NF scoring: ESCLUSO (Δ={_fmt(diff,2)} dB, finestre={nwin}, durata={_fmt(ndur,2)} s)")
        elif moderate_uncertain:
            print(f"   NF scoring: PESO RIDOTTO (Δ={_fmt(diff,2)} dB, finestre={nwin}, durata={_fmt(ndur,2)} s)")

        if 'nf_interval_low_dbfs' not in r or r.get('nf_interval_low_dbfs') is None:
            try:
                compute_nf_interval_fields(r)
            except Exception:
                pass
        low_i = r.get('nf_interval_low_dbfs')
        high_i = r.get('nf_interval_high_dbfs')
        unc_i = r.get('nf_interval_unc_db')
        conf_i = r.get('nf_interval_conf_label')
        if low_i is not None and high_i is not None and unc_i is not None:
            print(f"   NF intervallo credibile: [{_fmt(low_i,1)}, {_fmt(high_i,1)}] dBFS (±{_fmt(unc_i,1)}, conf={conf_i or 'N/A'})")

        spur = r.get('noise_spur_db')
        isp = r.get('isp_margin_db')
        dc = r.get('dc_offset_dbfs')
        print(f" • Noise spur:  {_fmt(spur,1)} dB{_status('noise_spur_db', spur, r)}")
        print(f" • ISP margin:  {_fmt(isp,2)} dB{_status('isp_margin_db', isp, r)}")
        print(f" • DC offset:   {_fmt(dc,1)} dBFS{_status('dc_offset_dbfs', dc, r)}")

        nf_cr = r.get("noise_floor_cross_rms_dbfs")
        nf_df = r.get("noise_consistency_diff_db")
        nf_p90 = r.get("noise_floor_cross_p90_dbfs")
        nf_cap = r.get("noise_floor_sanity_cap_dbfs")
        nf_ap = r.get("noise_floor_sanity_applied")
        if nf_cr is not None or nf_df is not None:
            print(f" • NF cross:    {_fmt(nf_cr,1)} dBFS | Δ={_fmt(nf_df,2)} dB")
        if (nf_p90 is not None) or (nf_cap is not None) or (nf_ap is not None):
            reason = r.get("noise_floor_cap_reason")
            reason_s = f" | reason={reason}" if reason else ""
            print(f" • NF sanity:   p90={_fmt(nf_p90,1)} dBFS | cap={_fmt(nf_cap,1)} dBFS | applied={'SÌ' if bool(nf_ap) else 'NO'}{reason_s}")

        print("\nSezione Qualità:")
        print(f" • DR (TT):     {_fmt(r.get('dr_tt_avg'),1)} dB{_status('dr_tt_avg', r.get('dr_tt_avg'), r)}")
        plr_int = r.get('plr_est')
        plr_act = r.get("plr_active_db")
        plr_eff = r.get("plr_effective_db", plr_int if plr_int is not None else plr_act)
        print(f" • PLR (eff):   {_fmt(plr_eff,1)} dB{_status('plr_est', plr_eff, r)}")
        if plr_int is not None or plr_act is not None:
            print(f"   PLR integ./attivo: {_fmt(plr_int,1)} / {_fmt(plr_act,1)} dB")
        print(f" • LRA:         {_fmt(r.get('lra_est'),1)} LU{_status('lra_est', r.get('lra_est'), r)}")
        print(f" • ST IQR:      {_fmt(r.get('st_lufs_iqr_db'),1)} dB{_status('st_lufs_iqr_db', r.get('st_lufs_iqr_db'), r)}")
        print(f" • ToneBal:     {_fmt(r.get('spectral_balance_dev_db'),1)} dB{_status('spectral_balance_dev_db', r.get('spectral_balance_dev_db'), r)}")
        print(f" • Width IQR:   {_fmt(r.get('stereo_width_iqr_db'),1)} dB{_status('stereo_width_iqr_db', r.get('stereo_width_iqr_db'), r)}")

        tpv = r.get('true_peak_est_dbtp', r.get('peak_dbfs_overall'))
        print("\nAltre metriche:")
        print(f" • True Peak:   {_fmt(tpv,2)} dBTP")
        print(f" • Clipping:    {('SÌ' if r.get('clipping_detected') else 'NO') if r.get('clipping_detected') is not None else 'N/A'}")
        print(f" • LUFS:        {_fmt(r.get('loudness_lufs'),1)} LUFS (backend: {r.get('loudness_backend','N/A')})")
        print(f" • LRA:         {_fmt(r.get('lra_est'),1)} LU")
        print()

def print_sorted_index_console(results_list):
    if not results_list:
        print("\nIndice: nessun file.")
        return

    import os
    import numpy as np

    def _sf(x, d=0.0):
        try:
            f = float(x)
            return f if np.isfinite(f) else d
        except Exception:
            return d

    def _fmt(v, dec=1, na="N/A"):
        try:
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                return na
            return f"{float(v):.{dec}f}"
        except Exception:
            return na

    def _conf_from_metrics(r0, r1):
        def _f(x):
            try:
                v = float(x)
                return v if np.isfinite(v) else None
            except Exception:
                return None
        spur0, spur1 = _f(r0.get('noise_spur_db')), _f(r1.get('noise_spur_db'))
        isp0,  isp1  = _f(r0.get('isp_margin_db')), _f(r1.get('isp_margin_db'))
        dc0,   dc1   = _f(r0.get('dc_offset_dbfs')), _f(r1.get('dc_offset_dbfs'))

        def adiff(a, b):
            if a is None or b is None:
                return None
            return abs(a - b)

        d_spur = adiff(spur0, spur1)
        d_isp  = adiff(isp0,  isp1)
        d_dc   = adiff(dc0,   dc1)

        strong = ((d_spur is not None and d_spur >= 3.0) or
                  (d_isp  is not None and d_isp  >= 0.15) or
                  (d_dc   is not None and d_dc   >= 3.0))
        weak   = ((d_spur is not None and d_spur >= 1.0) or
                  (d_isp  is not None and d_isp  >= 0.05) or
                  (d_dc   is not None and d_dc   >= 1.0))

        if strong:
            return "Alta (tie-metriche)"
        if weak:
            return "Media (tie-metriche)"
        return "Bassa (tie-metriche)"

    def _conf_debug_top(ordered_list, tie_mode_local):
        if not ordered_list or len(ordered_list) < 2:
            return "N/A"
        a, b = ordered_list[0], ordered_list[1]
        s0 = _sf(a.get('score_float'), _sf(a.get('score')))
        s1 = _sf(b.get('score_float'), _sf(b.get('score')))

        if tie_mode_local in ("none", "off", "0"):
            if s0 > s1:
                return "Alta (score)"
            return _conf_from_metrics(a, b)

        resid = a.get('pair_residual_to_next_db')
        try:
            if isinstance(resid, (int, float)) and np.isfinite(resid):
                if resid <= -3.0:
                    return "Alta (residuo)"
                if resid <= -1.5:
                    return "Media (residuo)"
                return "Bassa (residuo)"
        except Exception:
            pass
        if s0 > s1:
            return "Alta (score)"
        return _conf_from_metrics(a, b)

    tie_mode = (os.environ.get("AQC_TIEBREAK", "none").strip().lower())
    dq_last  = (os.environ.get("AQC_RANK_DQ_LAST", "1").strip().lower() in ("1", "true", "yes", "on"))
    if tie_mode in ("none", "off", "0"):
        crit = "score (solo score)"
    elif tie_mode in ("residual", "residuo", "default"):
        crit = "score + tie-break (residuo)"
    else:
        crit = f"score + tie-break ({tie_mode})"
    crit += " | DQ in coda" if dq_last else " | DQ inclusi"

    ordered = list(results_list)

    print(f"\nCriterio classifica: {crit}")
    print("\nIndice (ordine finale):")
    for i, r in enumerate(ordered, 1):
        name = r.get('filename') or os.path.basename(r.get('filepath', '')) or "(sconosciuto)"
        score = _sf(r.get('score_float'), _sf(r.get('score')))
        print(f" {i}) {name:<50} {score:>6.2f}/100")

    if ordered:
        conf_dbg = _conf_debug_top(ordered, tie_mode)
        print(f"\nConf. debug sul 1°: {conf_dbg}")

    for i, r in enumerate(ordered, 1):
        nf = r.get('noise_floor_dbfs')
        nf_raw = r.get('noise_floor_raw_dbfs')
        nf_bb = r.get('nf_broadband_dbfs', r.get('noise_floor_dbfs'))
        nwin = int(r.get("noise_windows_count") or 0)
        ndur = float(r.get("noise_total_duration_sec") or 0.0)
        diff = float(r.get("noise_consistency_diff_db") or 0.0)
        extreme_uncertain = (diff > 30.0) or (nwin < 2 and ndur < 3.0)
        nf_mode = r.get("nf_scoring_mode")
        if nf_mode is None:
            safe_env = os.environ.get("AQC_NF_SAFE", "").strip().lower()
            nf_safe = (safe_env not in ("0", "false", "off", "no", "n"))
            if extreme_uncertain:
                nf_mode = "EXCLUDED"
            else:
                if nf_safe:
                    nf_mode = "SAFE"
                else:
                    key = _policy_select_nf_profile(nf, r, pol)
                    nf_mode = "PROFILE_24" if (key and str(key).endswith("_24")) else "PROFILE_16"
        if i == 1:
            print("\nDettaglio NF (1° in classifica):")
            print(f" • NF raw/BB/cap: {_fmt(nf_raw,1)} / {_fmt(nf_bb,1)} / {_fmt(nf,1)} dBFS")
            if nf_mode == "SAFE":
                print(" • NF scoring mode: SAFE (max(16,24))")
            elif nf_mode == "PROFILE_24":
                print(" • NF scoring mode: PROFILE_24")
            elif nf_mode == "PROFILE_16":
                print(" • NF scoring mode: PROFILE_16")
            elif nf_mode == "EXCLUDED":
                print(" • NF scoring mode: EXCLUDED")

def measure_loudness(filepath, audio_f32, sr):
    import numpy as np
    import math
    import os

    try:
        native = os.environ.get("AQC_NATIVE_SR", "").strip().lower() in ("1", "true", "on", "yes", "y")
    except Exception:
        native = False

    lufs = None
    lra = None
    backend = None
    lt_series_out = None
    ds_sr = int(sr) if (native and isinstance(sr, (int, float)) and sr > 0) else 48000

    def _energy_mean_db(db_series):
        if db_series is None:
            return None
        arr = np.asarray(db_series, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return None
        lin = 10.0 ** (arr / 10.0)
        e = float(np.mean(lin)) if lin.size else None
        if e is None or not np.isfinite(e) or e <= 0:
            return None
        return float(10.0 * np.log10(e))

    def _gate_series(db_series, ref_db=None):
        arr = np.asarray(db_series, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return arr
        if ref_db is None:
            med = float(np.median(arr))
            gate = max(-70.0, med - 10.0)
        else:
            gate = max(-70.0, ref_db - 10.0)
        sel = arr[arr >= gate]
        if sel.size < max(5, int(0.05 * arr.size)):
            idx = np.argsort(arr)[-max(5, int(0.05 * arr.size)):]
            sel = arr[idx]
        return sel

    def _rms_dbfs_sliding(x, sr_local):
        n = x.size
        win = int(round(0.400 * sr_local))
        hop = max(1, int(round(0.100 * sr_local)))
        if n < win or win <= 0 or hop <= 0:
            return np.array([], dtype=float)
        out = []
        for i in range(0, n - win + 1, hop):
            seg = x[i:i + win].astype(np.float64, copy=False)
            if seg.size == 0:
                continue
            r = float(np.sqrt(np.mean(seg * seg))) if np.any(np.isfinite(seg)) else 0.0
            if r > 1e-12 and np.isfinite(r):
                out.append(20.0 * math.log10(r))
        return np.array(out, dtype=float) if out else np.array([], dtype=float)

    def _fallback_rms_loudness(x_ds, sr_ds):
        dbts = _rms_dbfs_sliding(x_ds, sr_ds)
        dbts = dbts[np.isfinite(dbts)]
        if dbts.size == 0:
            return None, None, None, None
        gated = _gate_series(dbts)
        I = _energy_mean_db(gated)
        if I is None or not np.isfinite(I):
            I = _energy_mean_db(dbts)
        LRA = None
        base_for_gate = I if (I is not None and np.isfinite(I)) else (float(np.median(dbts)) if dbts.size else None)
        if base_for_gate is not None and np.isfinite(base_for_gate):
            g2 = _gate_series(dbts, ref_db=base_for_gate)
            if g2.size >= 5:
                p95 = float(np.percentile(g2, 95))
                p10 = float(np.percentile(g2, 10))
                LRA = max(0.0, p95 - p10)
        return I, LRA, "rms-fallback", dbts

    def _parse_ffmpeg_ebur128(txt):
        import re
        txt = txt.replace("\u2212", "-").replace("\u2013", "-").replace("\u2014", "-").replace("\u00A0", " ")
        patt_I = [
            r'\bIntegrated(?:\s+loudness)?:\s*([\-]?\d+(?:[.,]\d+)?)\s*LUFS\b',
            r'\bI:\s*([\-]?\d+(?:[.,]\d+)?)\s*LUFS\b',
            r'\bI\s*=\s*([\-]?\d+(?:[.,]\d+)?)\s*LUFS\b',
        ]
        patt_LRA = [
            r'\bLoudness(?:\s+range)?:\s*([\-]?\d+(?:[.,]\d+)?)\s*LU\b',
            r'\bLRA:\s*([\-]?\d+(?:[.,]\d+)?)\s*LU\b',
            r'\bLRA\s*=\s*([\-]?\d+(?:[.,]\d+)?)\s*LU\b',
        ]
        I_val = None
        LRA_val = None
        for p in patt_I:
            mm = re.findall(p, txt, flags=re.IGNORECASE | re.MULTILINE)
            if mm:
                s = mm[-1].replace(",", ".")
                try:
                    v = float(s)
                    if np.isfinite(v):
                        I_val = v
                        break
                except Exception:
                    pass
        for p in patt_LRA:
            mm = re.findall(p, txt, flags=re.IGNORECASE | re.MULTILINE)
            if mm:
                s = mm[-1].replace(",", ".")
                try:
                    v = float(s)
                    if np.isfinite(v):
                        LRA_val = v
                        break
                except Exception:
                    pass
        return I_val, LRA_val

    def _run_ffmpeg_ebur128(path):
        import subprocess
        try:
            threads = "1"
            if native:
                fchain = 'ebur128=peak=true:framelog=0'
            else:
                pre_ds = f"aresample=sample_rate={ds_sr}:resampler=soxr:precision=28"
                fchain = f'{pre_ds},ebur128=peak=true:framelog=0'
            cmd = [
                'ffmpeg', '-hide_banner', '-nostats', '-nostdin', '-threads', threads,
                '-i', path, '-map', '0:a:0',
                '-filter_complex', fchain,
                '-f', 'null', '-'
            ]
            proc = subprocess.run(cmd, check=False, capture_output=True, text=True, encoding="utf-8", errors="replace")
            out = (proc.stderr or "") + "\n" + (proc.stdout or "")
            return _parse_ffmpeg_ebur128(out)
        except Exception:
            return None, None

    try:
        if isinstance(audio_f32, np.ndarray):
            if audio_f32.ndim == 2:
                if audio_f32.shape[1] in (1, 2):
                    x_mono = audio_f32.mean(axis=1).astype(np.float32, copy=False)
                else:
                    x_mono = audio_f32.T.mean(axis=1).astype(np.float32, copy=False)
            else:
                x_mono = audio_f32.astype(np.float32, copy=False)
        else:
            x_mono = None
        if x_mono is None or not isinstance(sr, (int, float)) or not (sr > 0):
            raise RuntimeError("invalid input")
        sr_use = int(sr)
        if sr_use != ds_sr:
            try:
                from scipy.signal import resample_poly
                g = math.gcd(int(sr_use), int(ds_sr))
                up = int(ds_sr // g)
                down = int(sr_use // g)
                x_mono = resample_poly(x_mono.astype(np.float64, copy=False), up, down).astype(np.float32, copy=False)
                sr_use = ds_sr
            except Exception:
                n = int(round(len(x_mono) * (ds_sr / float(sr_use))))
                if n <= 0:
                    n = len(x_mono)
                xi = np.linspace(0.0, len(x_mono) - 1.0, num=n, endpoint=False, dtype=np.float64)
                x_mono = np.interp(xi, np.arange(len(x_mono), dtype=np.float64), x_mono.astype(np.float64, copy=False)).astype(np.float32, copy=False)
                sr_use = ds_sr
        try:
            import pyloudnorm as pyln
            meter = pyln.Meter(sr_use, block_size=0.400, filter_class="K-weighting")
            try:
                lt = meter.loudness_time_series(x_mono)
                lt = np.array(lt, dtype=float)
                lt = lt[np.isfinite(lt)]
            except Exception:
                lt = None
            try:
                li = meter.integrated_loudness(x_mono)
            except Exception:
                li = None
            try:
                lr = meter.loudness_range(x_mono)
            except Exception:
                lr = None
            if li is not None and np.isfinite(li):
                lufs = float(li)
            elif lt is not None and lt.size >= 5:
                gated = _gate_series(lt)
                I = _energy_mean_db(gated)
                if I is None or not np.isfinite(I):
                    I = _energy_mean_db(lt)
                lufs = I if (I is not None and np.isfinite(I)) else None
            if lr is not None and np.isfinite(lr):
                lra = float(lr)
            elif lt is not None and lt.size >= 5:
                base = lufs if (lufs is not None and np.isfinite(lufs)) else float(np.median(lt))
                gated = _gate_series(lt, ref_db=base)
                if gated.size >= 5:
                    p95 = float(np.percentile(gated, 95))
                    p10 = float(np.percentile(gated, 10))
                    lra = max(0.0, p95 - p10)
            backend = "pyloudnorm"
            if lt is not None and lt.size >= 5:
                lt_series_out = lt.astype(float, copy=False)
        except Exception:
            lufs = None
            lra = None
            backend = None
        if backend is None or ((lufs is None or not np.isfinite(lufs)) and (lra is None or not np.isfinite(lra))):
            I_val, LRA_val = _run_ffmpeg_ebur128(filepath)
            if I_val is not None and np.isfinite(I_val):
                lufs = float(I_val)
            if LRA_val is not None and np.isfinite(LRA_val):
                lra = float(LRA_val)
            backend = "ffmpeg-ebur128" if (lufs is not None or lra is not None) else None
            lt_series_out = None
        if backend is None or (lufs is None or not np.isfinite(lufs) or lra is None or not np.isfinite(lra)):
            I2, LRA2, be2, lt2 = _fallback_rms_loudness(x_mono, sr_use)
            if lufs is None or not np.isfinite(lufs):
                lufs = I2
            if lra is None or not np.isfinite(lra):
                lra = LRA2
            lt_series_out = lt2 if lt2 is not None and len(lt2) >= 5 else lt_series_out
            if backend is None:
                backend = be2
    except Exception:
        I2 = LRA2 = None
        be2 = None
        lt2 = None
        try:
            x_mono = None
            if isinstance(audio_f32, np.ndarray):
                if audio_f32.ndim == 2:
                    if audio_f32.shape[1] in (1, 2):
                        x_mono = audio_f32.mean(axis=1).astype(np.float32, copy=False)
                    else:
                        x_mono = audio_f32.T.mean(axis=1).astype(np.float32, copy=False)
                else:
                    x_mono = audio_f32.astype(np.float32, copy=False)
            if x_mono is not None and isinstance(sr, (int, float)) and sr > 0:
                sr_use = int(sr)
                if sr_use != ds_sr:
                    n = int(round(len(x_mono) * (ds_sr / float(sr_use))))
                    xi = np.linspace(0.0, len(x_mono) - 1.0, num=n, endpoint=False, dtype=np.float64)
                    x_mono = np.interp(xi, np.arange(len(x_mono), dtype=np.float64), x_mono.astype(np.float64, copy=False)).astype(np.float32, copy=False)
                    sr_use = ds_sr
                I2, LRA2, be2, lt2 = _fallback_rms_loudness(x_mono, sr_use)
        except Exception:
            pass
        lufs = I2 if (I2 is not None and np.isfinite(I2)) else None
        lra = LRA2 if (LRA2 is not None and np.isfinite(LRA2)) else None
        backend = be2 or "unknown"
        lt_series_out = lt2 if lt2 is not None and len(lt2) >= 5 else None

    return (lufs if (lufs is not None and np.isfinite(lufs)) else None,
            lra  if (lra  is not None and np.isfinite(lra))  else None,
            backend or "unknown",
            lt_series_out if (lt_series_out is not None and len(lt_series_out) >= 5) else None)

def compute_st_lufs_iqr(audio_f32: np.ndarray, sr: int, lt_precomputed=None) -> float | None:
    if audio_f32 is None or sr is None or sr <= 0:
        return None
    try:
        if lt_precomputed is not None:
            lt = np.asarray(lt_precomputed, dtype=float)
            lt = lt[np.isfinite(lt)]
            if lt.size >= 5:
                med = float(np.median(lt))
                gate = max(-70.0, med - 10.0)
                ltg = lt[lt >= gate]
                if ltg.size < max(5, int(0.05 * lt.size)):
                    idx = np.argsort(lt)[-max(5, int(0.05 * lt.size)):]
                    ltg = lt[idx]
                p95 = float(np.percentile(ltg, 95))
                p5 = float(np.percentile(ltg, 5))
                d = p95 - p5
                if np.isfinite(d) and d > 0:
                    return float(min(max(d, 0.0), 30.0))
        if isinstance(audio_f32, np.ndarray):
            if audio_f32.ndim == 2:
                if audio_f32.shape[0] < audio_f32.shape[1] and audio_f32.shape[1] in (1, 2):
                    x_mono = audio_f32.mean(axis=1)
                elif audio_f32.shape[0] in (1, 2):
                    x_mono = audio_f32.T.mean(axis=1)
                else:
                    axis = 1 if audio_f32.shape[1] in (1, 2) else 0
                    x_mono = audio_f32.mean(axis=axis)
            else:
                x_mono = audio_f32
        else:
            return None
        x_mono = np.asarray(x_mono, dtype=np.float32)
    except Exception:
        return None
    n = x_mono.size
    if n < int(0.4 * sr):
        return None
    try:
        import pyloudnorm as pyln
        meter = pyln.Meter(sr, block_size=0.400, filter_class="K-weighting")
        lt = meter.loudness_time_series(x_mono)
        if lt is None:
            raise RuntimeError("no lt")
        lt = np.array(lt, dtype=float)
        lt = lt[np.isfinite(lt)]
        if lt.size < 5:
            raise RuntimeError("short lt")
        med = float(np.median(lt))
        gate = max(-70.0, med - 10.0)
        ltg = lt[lt >= gate]
        if ltg.size < max(5, int(0.05 * lt.size)):
            idx = np.argsort(lt)[-max(5, int(0.05 * lt.size)):]
            ltg = lt[idx]
        p95 = float(np.percentile(ltg, 95))
        p5 = float(np.percentile(ltg, 5))
        d = p95 - p5
        if not np.isfinite(d) or d <= 0:
            return None
        return float(min(max(d, 0.0), 30.0))
    except Exception:
        try:
            win = int(round(0.4 * sr))
            hop = max(1, int(round(0.1 * sr)))
            if win <= 0 or hop <= 0 or n < win:
                return None
            vals = []
            for i in range(0, n - win + 1, hop):
                seg = x_mono[i:i + win].astype(np.float64, copy=False)
                rms = calculate_rms(seg)
                if rms > 1e-12:
                    v = dbfs(rms)
                    if np.isfinite(v):
                        vals.append(v)
            if len(vals) < 5:
                return None
            arr = np.array(vals, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size < 5:
                return None
            med = float(np.median(arr))
            gate = max(-70.0, med - 10.0)
            arrg = arr[arr >= gate]
            if arrg.size < max(5, int(0.05 * arr.size)):
                idx = np.argsort(arr)[-max(5, int(0.05 * arr.size)):]
                arrg = arr[idx]
            p95 = float(np.percentile(arrg, 95))
            p5 = float(np.percentile(arrg, 5))
            d = p95 - p5
            if not np.isfinite(d) or d <= 0:
                return None
            return float(min(max(d, 0.0), 30.0))
        except Exception:
            return None

def select_noise_segments_robust(mono: np.ndarray, sr: int, mode: str = "normal", silence_dbfs_thr: float = -120.0, min_cov_sec: float = 1.5):
    if mono is None or sr is None or sr <= 0 or not isinstance(mono, np.ndarray):
        return np.array([], dtype=np.float64), "Bassa", []
    x = mono.astype(np.float64, copy=False)
    n = x.size
    if n < int(0.1 * sr):
        return np.array([], dtype=np.float64), "Bassa", []
    try:
        nyq = sr * 0.5
        fc = 20.0 / nyq
        if 0 < fc < 1:
            sos = butter(2, fc, btype='highpass', output='sos')
            x = sosfilt(sos, x)
    except Exception:
        pass
    edge_exclude = int((0.05 if mode == "strict" else 0.03) * n)
    if edge_exclude * 2 >= n:
        edge_exclude = 0
    xw = x[edge_exclude:n - edge_exclude] if edge_exclude > 0 else x
    base_offset = edge_exclude
    scales_sec = (0.200, 0.400, 0.800, 1.600) if mode != "strict" else (0.200, 0.400, 0.800)
    target_total_sec = 3.0
    min_cluster_sec = 0.60 if mode == "strict" else 0.50
    eps_local = 1e-20
    approved_windows = []
    clusters_count = 0
    total_dur = 0.0
    def spectral_features(seg, sr_local):
        if seg.size < 64:
            return None
        w = np.hanning(seg.size)
        y = seg * w
        spec = np.abs(np.fft.rfft(y))
        pwr = spec**2 + eps_local
        gmean = np.exp(np.mean(np.log(pwr)))
        amean = np.mean(pwr)
        sfm = gmean / amean if amean > 0 else 0.0
        freqs = np.fft.rfftfreq(y.size, 1 / sr_local)
        f_hi = min(20000.0, 0.98 * (sr_local / 2.0))
        band = (freqs >= 20.0) & (freqs <= f_hi)
        if np.any(band):
            band_amp = spec[band] + eps_local
            med = np.median(band_amp)
            mx = np.max(band_amp)
            crest_db = 20.0 * np.log10(mx / med) if med > 0 else 0.0
            try:
                from scipy.signal import fftconvolve
                log_spec = np.log(band_amp)
                k = max(31, int(round(band_amp.size * 0.01)))
                if k % 2 == 0:
                    k += 1
                if k >= band_amp.size:
                    k = max(31, band_amp.size // 3 if (band_amp.size // 3) % 2 == 1 else max(31, band_amp.size // 3 - 1))
                ker = np.ones(k, dtype=np.float64) / k
                baseline = np.exp(fftconvolve(log_spec, ker, mode='same'))
                prom_db = 20.0 * (np.log10(band_amp) - np.log10(baseline + eps_local))
                tonal_prom_db = float(np.max(prom_db)) if prom_db.size else 0.0
            except Exception:
                tonal_prom_db = crest_db
        else:
            crest_db = 0.0
            tonal_prom_db = 0.0
        rms = calculate_rms(seg)
        rms_db = dbfs(rms) if rms > 0 else -np.inf
        return rms_db, sfm, crest_db, tonal_prom_db, spec
    def _p90_gate(sig, sr_local):
        try:
            w = int(round(0.400 * sr_local))
            h = max(1, int(round(0.100 * sr_local)))
            if sig.size < w:
                return -60.0
            vals = []
            for i in range(0, sig.size - w + 1, h):
                r = calculate_rms(sig[i:i + w])
                if r > 1e-12:
                    v = dbfs(r)
                    if np.isfinite(v):
                        vals.append(v)
            if not vals:
                return -60.0
            arr = np.array(vals, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                return -60.0
            p90 = float(np.percentile(arr, 90))
            return -50.0 if p90 > -40.0 else -60.0
        except Exception:
            return -60.0
    abs_gate = _p90_gate(xw, sr)
    for wsec in scales_sec:
        win = max(128, int(round(wsec * sr)))
        hop = max(1, win // 2)
        if xw.size <= win:
            continue
        starts = list(range(0, xw.size - win + 1, hop))
        if len(starts) < 2:
            continue
        rms_db_list = []
        sfm_list = []
        crest_list = []
        prom_list = []
        specs = []
        for st in starts:
            seg = xw[st:st + win]
            feats = spectral_features(seg, sr)
            if feats is None:
                rms_db_list.append(-np.inf); sfm_list.append(0.0); crest_list.append(0.0); prom_list.append(0.0); specs.append(None)
            else:
                rms_db, sfm, crest_db, tonal_prom_db, spec_mag = feats
                rms_db_list.append(rms_db); sfm_list.append(sfm); crest_list.append(crest_db); prom_list.append(tonal_prom_db); specs.append(spec_mag)
        rms_db_arr = np.array(rms_db_list, dtype=float)
        sfm_arr = np.array(sfm_list, dtype=float)
        crest_arr = np.array(crest_list, dtype=float)
        prom_arr = np.array(prom_list, dtype=float)
        flux = np.zeros_like(rms_db_arr)
        for i in range(1, len(specs)):
            if specs[i] is None or specs[i-1] is None:
                flux[i] = np.inf
            else:
                a = specs[i-1].astype(np.float64, copy=False)
                b = specs[i].astype(np.float64, copy=False)
                if a.size != b.size:
                    m = min(a.size, b.size)
                    a = a[:m]; b = b[:m]
                da = np.log(np.maximum(a, eps_local))
                db = np.log(np.maximum(b, eps_local))
                diff = np.maximum(0.0, db - da)
                flux[i] = float(np.sum(diff))
        finite_rms = rms_db_arr[np.isfinite(rms_db_arr)]
        if finite_rms.size == 0:
            continue
        pA = np.percentile(finite_rms, 8 if mode == "strict" else 10)
        pB = np.percentile(finite_rms, 12 if mode == "strict" else 15)
        pC = np.percentile(finite_rms, 18 if mode == "strict" else 20)
        sfm_finite = sfm_arr[np.isfinite(sfm_arr)]
        if sfm_finite.size > 8:
            sfm_thr = float(np.percentile(sfm_finite, 75 if mode == "strict" else 70))
        else:
            sfm_thr = 0.45 if mode == "strict" else 0.35
        sfm_thr = max(sfm_thr, 0.42 if mode == "strict" else 0.30)
        crest_thr = 6.0 if mode == "strict" else 8.0
        prom_thr = 8.0 if mode == "strict" else 10.0
        flux_finite = flux[np.isfinite(flux)]
        if flux_finite.size >= 5:
            med = np.median(flux_finite)
            mad = np.median(np.abs(flux_finite - med)) + eps_local
            flux_thr = med + (2.0 if mode == "strict" else 2.5) * mad
        else:
            flux_thr = np.inf
        silence_gate = max(silence_dbfs_thr, abs_gate)
        silent_mask = rms_db_arr <= silence_gate
        mask = (rms_db_arr <= pA) & (sfm_arr >= sfm_thr) & (crest_arr <= crest_thr) & (prom_arr <= prom_thr) & (flux <= flux_thr) & (~silent_mask)
        if not np.any(mask):
            mask = (rms_db_arr <= pB) & (sfm_arr >= (sfm_thr * (1.00 if mode == "strict" else 0.95))) & (crest_arr <= (crest_thr + (0.5 if mode == "strict" else 1.0))) & (prom_arr <= (prom_thr + (0.5 if mode == "strict" else 1.0))) & (flux <= (flux_thr * (1.1 if mode == "strict" else 1.2))) & (~silent_mask)
        if not np.any(mask):
            mask = (rms_db_arr <= pC) & (sfm_arr >= (sfm_thr * (0.95 if mode == "strict" else 0.90))) & (crest_arr <= (crest_thr + (1.0 if mode == "strict" else 2.0))) & (prom_arr <= (prom_thr + (1.0 if mode == "strict" else 2.0))) & (flux <= (flux_thr * (1.2 if mode == "strict" else 1.5))) & (~silent_mask)
        idx = np.where(mask)[0]
        if idx.size == 0:
            continue
        runs = []
        s0 = idx[0]; prev = idx[0]
        for k in idx[1:]:
            if k == prev + 1:
                prev = k
            else:
                runs.append((s0, prev))
                s0 = k; prev = k
        runs.append((s0, prev))
        accepted = []
        for a, b in runs:
            dur = (b - a + 1) * hop / sr
            if dur >= min_cluster_sec:
                st_abs = base_offset + starts[a]
                en_abs = base_offset + starts[b] + win
                seg_rms = calculate_rms(x[st_abs:en_abs])
                if dbfs(seg_rms) > silence_gate:
                    accepted.append((st_abs, en_abs))
                    clusters_count += 1
                    total_dur += dur
            if total_dur >= target_total_sec:
                break
        approved_windows.extend(accepted)
        if total_dur >= target_total_sec:
            break
    if not approved_windows:
        wsec = 0.400
        win = max(128, int(round(wsec * sr)))
        hop = max(1, win // 2)
        if xw.size > win:
            starts = list(range(0, xw.size - win + 1, hop))
            rms_db_list = []
            for st in starts:
                seg = xw[st:st + win]
                rms = calculate_rms(seg)
                rms_db_list.append(dbfs(rms) if rms > 0 else -np.inf)
            rms_db_arr = np.array(rms_db_list, dtype=float)
            finite = np.isfinite(rms_db_arr)
            if np.any(finite):
                silence_gate = max(silence_dbfs_thr, abs_gate)
                k = max(3, int(0.10 * finite.sum()))
                idx_sorted = np.argsort(rms_db_arr[finite])[:k]
                starts_f = np.array(starts, dtype=int)[finite][idx_sorted]
                for st in starts_f:
                    st_abs = base_offset + st
                    en_abs = st_abs + win
                    seg_rms = calculate_rms(x[st_abs:en_abs])
                    if dbfs(seg_rms) > silence_gate:
                        approved_windows.append((st_abs, en_abs))
                if approved_windows:
                    total_dur = len(approved_windows) * (hop / sr)
                    clusters_count = max(1, len(approved_windows))
        if not approved_windows:
            approved_windows = [(0, n)]
    approved_windows.sort(key=lambda t: t[0])
    merged = []
    for w in approved_windows:
        if not merged:
            merged.append(list(w))
            continue
        last = merged[-1]
        if w[0] <= last[1] + int(0.05 * sr):
            last[1] = max(last[1], w[1])
        else:
            merged.append([w[0], w[1]])
    merged = [(int(a), int(b)) for a, b in merged if b > a and a >= 0 and b <= n]
    noise_segments = []
    total_concat = 0.0
    silence_gate = max(silence_dbfs_thr, abs_gate)
    for st, en in merged:
        seg = x[st:en]
        if seg.size <= 0:
            continue
        if dbfs(calculate_rms(seg)) <= silence_gate:
            continue
        noise_segments.append(seg)
        total_concat += (en - st) / sr
        if total_concat >= 10.0:
            break
    noise_samples = np.concatenate(noise_segments) if noise_segments else np.array([], dtype=np.float64)
    cov = float(sum(max(0, en - st) for st, en in merged)) / float(sr)
    if cov < min_cov_sec and mode != "strict":
        return select_noise_segments_robust(mono, sr, mode="strict", silence_dbfs_thr=silence_dbfs_thr, min_cov_sec=min_cov_sec)
    if total_concat >= 3.0 and clusters_count >= 2:
        conf = "Alta"
    elif total_concat >= 1.5 or clusters_count >= 1:
        conf = "Media"
    else:
        conf = "Bassa"
    if noise_samples.size < int(0.1 * sr):
        return select_noise_segments_robust(mono, sr, mode="strict", silence_dbfs_thr=silence_dbfs_thr, min_cov_sec=min_cov_sec) if mode != "strict" else (x, "Bassa", [(0, n)])
    return noise_samples, conf, merged

def select_noise_windows(x: np.ndarray,
                         sr: int,
                         mode: str = "auto",
                         win_sec: float = 0.40,
                         hop_sec: float = 0.10,
                         max_total_sec: float = 8.0,
                         hum_hint: dict | None = None):
    if x is None or not isinstance(x, np.ndarray) or x.size < 8 or sr is None or sr <= 0:
        return [], {
            "noise_selection_mode": "none",
            "noise_windows_count": 0,
            "noise_total_duration_sec": 0.0,
            "noise_total_duration_uncapped_sec": 0.0,
            "nf_notch_reduction_db": 0.0,
            "notch_f0": None,
            "notch_harmonics": 0,
            "music_limited": False,
            "noise_rms_p90_dbfs": None,
            "hum_estimated_hcount": 0,
        }

    x = x.astype(np.float64, copy=False)
    N = x.size
    win = max(1, int(round(win_sec * sr)))
    hop = max(1, int(round(hop_sec * sr)))
    if win > N:
        win = N
        hop = max(1, win // 2)

    starts = list(range(0, N - win + 1, hop)) or [0]
    starts_arr = np.asarray(starts, dtype=np.int64)

    def _dbfs_from_rms(v):
        try:
            return dbfs(v)
        except Exception:
            return -np.inf if v <= 0 else 20.0 * np.log10(max(v, 1e-30))

    def _rms_via_cumsum(csum_sq, s, e):
        s = int(s); e = int(e)
        if e <= s:
            return 0.0
        sumsq = csum_sq[e] - csum_sq[s]
        return float(np.sqrt(max(sumsq, 0.0) / float(e - s)))

    x2 = x * x
    csum_sq = np.empty(x2.size + 1, dtype=np.float64)
    csum_sq[0] = 0.0
    np.cumsum(x2, out=csum_sq[1:])
    rms_pre = np.sqrt((csum_sq[starts_arr + win] - csum_sq[starts_arr]) / float(win))
    rms_db_pre = np.array([_dbfs_from_rms(v) for v in rms_pre], dtype=np.float64)
    valid_pre = rms_db_pre[np.isfinite(rms_db_pre)]
    p90_db = float(np.percentile(valid_pre, 90)) if valid_pre.size else -np.inf

    def _estimate_hum_dense(sig, sr_local):
        try:
            eps = 1e-20
            nfft = int(2 ** np.ceil(np.log2(min(sig.size, 1_048_576))))
            if nfft < 16384:
                nfft = 16384
            w = np.hanning(min(sig.size, nfft))
            if sig.size < nfft:
                pad = np.zeros(nfft - sig.size, dtype=np.float64)
                y = np.concatenate([sig, pad])[:nfft] * np.hanning(nfft)
            else:
                y = sig[:nfft] * w
            spec = np.abs(np.fft.rfft(y)) + eps
            freqs = np.fft.rfftfreq(nfft, 1.0 / sr_local)
            f_hi = min(20000.0, 0.98 * (sr_local / 2.0))
            band = (freqs >= 20.0) & (freqs <= f_hi)
            if not np.any(band):
                return {"dense": False, "f0": None, "count": 0}
            mag = spec[band]
            fb = freqs[band]
            med = float(np.median(mag))
            if med <= 0:
                return {"dense": False, "f0": None, "count": 0}
            def count_harm(f0):
                cnt = 0
                k = 1
                while True:
                    fk = f0 * k
                    if fk > f_hi:
                        break
                    bw = max(1.0, fk * 0.01)
                    mask = (fb >= fk - bw) & (fb <= fk + bw)
                    if np.any(mask):
                        pk = float(np.max(mag[mask]))
                        if 20.0 * np.log10(pk / med) >= 8.0:
                            cnt += 1
                    k += 1
                return cnt
            c50 = count_harm(50.0)
            c60 = count_harm(60.0)
            if c50 >= c60:
                f0 = 50.0; cnt = c50
            else:
                f0 = 60.0; cnt = c60
            return {"dense": bool(cnt >= 8), "f0": f0 if cnt >= 2 else None, "count": cnt}
        except Exception:
            return {"dense": False, "f0": None, "count": 0}

    hum_info = hum_hint if isinstance(hum_hint, dict) else None
    if hum_info is None:
        hum_info = _estimate_hum_dense(x, sr)
    hum_dense = bool(hum_info.get("dense", False))
    hum_f0 = None
    try:
        if isinstance(hum_hint, dict) and isinstance(hum_hint.get("f0"), (int, float)) and np.isfinite(float(hum_hint.get("f0"))):
            hum_f0 = float(hum_hint.get("f0"))
        else:
            hum_f0 = float(hum_info.get("f0")) if isinstance(hum_info.get("f0"), (int, float)) else None
    except Exception:
        hum_f0 = None

    mode = (mode or "auto").strip().lower()
    if mode not in ("relaxed", "strict", "strict+notch", "auto"):
        mode = "auto"
    if mode == "auto":
        if hum_dense:
            mode_used = "strict+notch"
        elif p90_db > -40.0:
            mode_used = "strict"
        else:
            mode_used = "relaxed"
    else:
        mode_used = mode

    def _apply_notch_comb(sig, sr_local, f0, hum_is_dense, q_base=35.0):
        if not f0 or f0 <= 0:
            return sig, False, []
        try:
            from scipy.signal import iirnotch, filtfilt
            out = np.asarray(sig, dtype=np.float64)
            f_hi = min(20000.0, 0.98 * (sr_local / 2.0))
            used = []
            nmax_by_sr = int(np.floor(f_hi / float(f0))) if f0 > 0 else 0
            nmax_by_sr = max(0, nmax_by_sr)
            max_harm = min(nmax_by_sr, 20 if hum_is_dense else 12)
            if max_harm <= 0:
                return out, False, used
            for k in range(1, max_harm + 1):
                fk = f0 * k
                if fk > f_hi:
                    break
                if k <= 3:
                    qk = q_base * 1.7
                elif k <= 8:
                    qk = q_base
                else:
                    qk = q_base * 0.85
                try:
                    b, a = iirnotch(w0=fk, Q=max(5.0, qk), fs=sr_local)
                    out = filtfilt(b, a, out)
                    used.append(float(fk))
                except Exception:
                    break
            return out, (len(used) > 0), used
        except Exception:
            return sig, False, []

    x_notched = x
    notch_applied = False
    notch_freqs = []
    if mode_used == "strict+notch":
        x_notched, notch_applied, notch_freqs = _apply_notch_comb(x, sr, hum_f0 or 50.0, hum_dense, q_base=35.0)

    if notch_applied:
        x2_post = x_notched * x_notched
        csum_sq_post = np.empty(x2_post.size + 1, dtype=np.float64)
        csum_sq_post[0] = 0.0
        np.cumsum(x2_post, out=csum_sq_post[1:])
        rms_post = np.sqrt((csum_sq_post[starts_arr + win] - csum_sq_post[starts_arr]) / float(win))
        rms_db_post = np.array([_dbfs_from_rms(v) for v in rms_post], dtype=np.float64)
    else:
        csum_sq_post = None
        rms_db_post = rms_db_pre

    if mode_used == "relaxed":
        thr_db = float(np.percentile(rms_db_pre[np.isfinite(rms_db_pre)], 20)) - 2.0
        ref_arr = rms_db_pre
        use_post = False
    elif mode_used == "strict":
        thr_db = float(np.percentile(rms_db_pre[np.isfinite(rms_db_pre)], 10)) - 3.0
        ref_arr = rms_db_pre
        use_post = False
    else:
        valid_post = rms_db_post[np.isfinite(rms_db_post)]
        valid_pre2 = rms_db_pre[np.isfinite(rms_db_pre)]
        thr_a = float(np.percentile(valid_post, 15)) if valid_post.size else -np.inf
        thr_b = (float(np.percentile(valid_pre2, 10)) - 1.0) if valid_pre2.size else -np.inf
        thr_db = min(thr_a, thr_b)
        ref_arr = rms_db_post
        use_post = True

    abs_gate = -50.0 if p90_db > -40.0 else -60.0
    thr_eff = min(thr_db, abs_gate)
    sel_mask = np.isfinite(ref_arr) & (ref_arr <= thr_eff)
    selected_idx = np.nonzero(sel_mask)[0].tolist()
    if not selected_idx:
        order = np.argsort(ref_arr)
        cap_frames = max(1, int(np.ceil(max_total_sec / win_sec)))
        selected_idx = [i for i in order if ref_arr[i] <= abs_gate][:cap_frames] or order[:cap_frames]

    def _merge_indices(idxs, start_list, w, h):
        if not idxs:
            return []
        idxs = sorted(idxs)
        merged = []
        cur_s = start_list[idxs[0]]
        cur_e = min(cur_s + w, N)
        last = idxs[0]
        for i in idxs[1:]:
            s = start_list[i]
            e = min(s + w, N)
            if s <= start_list[last] + h:
                cur_e = e
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = s, e
            last = i
        merged.append((cur_s, cur_e))
        return merged

    merged_windows_uncapped = _merge_indices(selected_idx, starts, win, hop)

    def _win_score(win_tuple, use_post_arr):
        s, e = win_tuple
        mask = (starts_arr >= s) & (starts_arr < e)
        vals = (rms_db_post if use_post_arr else rms_db_pre)[mask]
        if vals.size == 0:
            if use_post_arr and notch_applied and (csum_sq_post is not None):
                return _dbfs_from_rms(_rms_via_cumsum(csum_sq_post, s, e))
            else:
                return _dbfs_from_rms(_rms_via_cumsum(csum_sq, s, e))
        return float(np.median(vals))

    ranked = sorted(merged_windows_uncapped, key=lambda w_: _win_score(w_, use_post))

    capped = []
    tot = 0.0
    cap = float(max_total_sec)
    for (s, e) in ranked:
        dur = (e - s) / sr
        if tot + dur > cap:
            resid = cap - tot
            if resid > 0.05:
                new_e = s + int(resid * sr)
                capped.append((s, new_e))
                tot += resid
            break
        capped.append((s, e))
        tot += dur
    if not capped and ranked:
        s, e = ranked[0]
        keep = min(e, s + int(min(cap, win_sec) * sr))
        capped = [(s, keep)]
        tot = (keep - s) / sr

    def _reduction_db_precomp(wins, csum_pre_local, csum_post_local):
        if not wins or csum_post_local is None:
            return 0.0
        pre_vals = []
        post_vals = []
        for s, e in wins:
            rpre = _rms_via_cumsum(csum_pre_local, s, e)
            rpos = _rms_via_cumsum(csum_post_local, s, e)
            if rpre > 0 and rpos > 0:
                pre_vals.append(_dbfs_from_rms(rpre))
                post_vals.append(_dbfs_from_rms(rpos))
        if not pre_vals or not post_vals:
            return 0.0
        return float(np.median(pre_vals) - np.median(post_vals))

    if mode_used == "strict+notch":
        reduction_db = _reduction_db_precomp(capped, csum_sq, csum_sq_post)
    else:
        reduction_db = 0.0

    uncapped_sec = float(sum((e - s) for (s, e) in merged_windows_uncapped)) / float(sr)
    music_limited = False

    if len(capped) < 2 or tot < 3.0:
        order = np.argsort(ref_arr)
        need_sec = 3.0
        picked = []
        picked_mask = np.zeros_like(ref_arr, dtype=bool)
        for idx in order:
            if ref_arr[idx] > abs_gate:
                continue
            if picked_mask[idx]:
                continue
            s = starts[idx]
            e = min(N, s + win)
            picked.append((s, e))
            start_frame = max(0, idx - int(round(0.5 * (win / max(1, hop)))))
            end_frame = min(ref_arr.size - 1, idx + int(round(0.5 * (win / max(1, hop)))))
            picked_mask[start_frame:end_frame + 1] = True
            if sum((q[1] - q[0]) for q in picked) / float(sr) >= need_sec:
                break
        if picked:
            picked_merged = _merge_indices([starts.index(p[0]) for p in picked if p[0] in starts], starts, win, hop)
            cap2 = []
            tot2 = 0.0
            for (s, e) in picked_merged:
                dur = (e - s) / sr
                if tot2 + dur > cap:
                    resid = cap - tot2
                    if resid > 0.05:
                        new_e = s + int(resid * sr)
                        cap2.append((s, new_e))
                        tot2 += resid
                    break
                cap2.append((s, e))
                tot2 += dur
            if (len(cap2) > len(capped)) or (tot2 > tot):
                capped = cap2
                tot = tot2
        if len(capped) < 2 or tot < 3.0:
            music_limited = True

    meta = {
        "noise_selection_mode": mode_used,
        "noise_windows_count": int(len(capped)),
        "noise_total_duration_sec": float(tot),
        "noise_total_duration_uncapped_sec": float(uncapped_sec),
        "nf_notch_reduction_db": float(round(reduction_db, 2)),
        "notch_f0": float(hum_f0) if (notch_applied and hum_f0) else (float(hum_f0) if hum_dense else None),
        "notch_harmonics": int(len(notch_freqs)) if notch_applied else 0,
        "music_limited": bool(music_limited),
        "noise_rms_p90_dbfs": float(round(p90_db, 2)) if np.isfinite(p90_db) else None,
        "hum_estimated_hcount": int(hum_info.get("count", 0)),
    }
    return capped, meta

def compute_nf_interval_fields(r):
    def _sf(x):
        try:
            f = float(x)
            return f if np.isfinite(f) else None
        except Exception:
            return None

    nf_bb = _sf(r.get('nf_broadband_dbfs'))
    nf_raw = _sf(r.get('noise_floor_raw_dbfs'))
    nf_cap = _sf(r.get('noise_floor_dbfs'))
    center = nf_bb if nf_bb is not None else (nf_raw if nf_raw is not None else nf_cap)

    if center is None:
        r['nf_interval_low_dbfs'] = None
        r['nf_interval_high_dbfs'] = None
        r['nf_interval_center_dbfs'] = None
        r['nf_interval_unc_db'] = None
        r['nf_interval_conf_label'] = None
        return r

    nwin = int(r.get("noise_windows_count") or 0)
    ndur = _sf(r.get("noise_total_duration_sec")) or 0.0
    diff = _sf(r.get("noise_consistency_diff_db")) or 0.0
    nf_cross = _sf(r.get("noise_floor_cross_rms_dbfs"))
    nf_cap_appl = bool(r.get("noise_floor_sanity_applied"))

    if (nwin >= 3 and ndur >= 3.0 and diff <= 6.0):
        unc = 2.0; conf = "Alta"
    elif (nwin >= 2 and ndur >= 2.0 and diff <= 10.0):
        unc = 3.0; conf = "Media"
    elif diff <= 20.0:
        unc = 4.0; conf = "Media"
    elif diff <= 30.0:
        unc = 5.0; conf = "Bassa"
    else:
        unc = 6.0; conf = "Bassa"

    if nf_cross is not None:
        delta = abs(center - nf_cross)
        if delta > unc:
            unc = min(8.0, delta + 1.0)
    if nf_cap_appl:
        unc = max(unc, 4.0)

    r['nf_interval_center_dbfs'] = float(center)
    r['nf_interval_unc_db'] = float(unc)
    r['nf_interval_low_dbfs'] = float(center - unc)
    r['nf_interval_high_dbfs'] = float(center + unc)
    r['nf_interval_conf_label'] = conf
    return r

def compute_noise_metrics_robust(mono: np.ndarray,
                                 sr: int,
                                 windows: list[tuple[int, int]] | None,
                                 noise_samples: np.ndarray | None = None,
                                 windows_meta: dict | None = None):
    import os
    if mono is None or sr is None or sr <= 0 or not isinstance(mono, np.ndarray) or mono.size < 8:
        return {
            "nf_dbfs": -np.inf,
            "nf_broadband_dbfs": -np.inf,
            "spur_db": 0.0,
            "nf_cross_rms_dbfs": -np.inf,
            "nf_consistency_diff_db": 0.0,
            "spur_label": None,
            "spur_fundamental_hz": None,
            "spur_harmonics_count": 0,
            "noise_windows_count": 0,
            "noise_total_duration_sec": 0.0,
            "noise_total_duration_uncapped_sec": 0.0,
            "nf_notch_reduction_db": 0.0,
            "notch_f0": None,
            "notch_harmonics": 0,
            "music_limited": False,
            "noise_rms_p90_dbfs": None,
            "hum_estimated_hcount": 0
        }

    x_in = mono.astype(np.float64, copy=False)
    N_in = x_in.size

    wm = windows_meta if isinstance(windows_meta, dict) else {}
    notch_f0 = wm.get("notch_f0")
    notch_harms = int(wm.get("notch_harmonics") or 0)
    nf_notch_reduction_db = float(wm.get("nf_notch_reduction_db")) if isinstance(wm.get("nf_notch_reduction_db"), (int, float)) and np.isfinite(wm.get("nf_notch_reduction_db")) else 0.0
    noise_rms_p90_dbfs = wm.get("noise_rms_p90_dbfs")
    music_limited = bool(wm.get("music_limited")) if "music_limited" in wm else False

    sr_eff = int(sr)
    x = x_in
    win_map = windows

    try:
        native = os.environ.get("AQC_NATIVE_SR", "").strip().lower() in ("1", "true", "on", "yes", "y")
    except Exception:
        native = False

    if not native and sr > 48000:
        try:
            from scipy.signal import resample_poly
            g = math.gcd(int(sr), 48000)
            up = 48000 // g
            down = int(sr) // g
            if up > 0 and down > 0:
                x = resample_poly(x_in, up, down).astype(np.float64, copy=False)
                sr_eff = int(round(sr * (up / down)))
                if isinstance(windows, (list, tuple)) and windows:
                    ratio = sr_eff / float(sr)
                    win_map = [(int(round(st * ratio)), int(round(en * ratio))) for st, en in windows]
            else:
                sr_eff = int(sr)
                win_map = windows
        except Exception:
            sr_eff = int(sr)
            win_map = windows
    else:
        sr_eff = int(sr)
        win_map = windows

    N = x.size

    if isinstance(win_map, (list, tuple)) and win_map:
        noise_windows_count = len(win_map)
        noise_total_duration_sec = float(sum(max(0, int(en) - int(st)) for st, en in win_map)) / float(sr_eff)
    else:
        noise_windows_count = 0
        noise_total_duration_sec = 0.0
    noise_total_duration_uncapped_sec = float(wm.get("noise_total_duration_uncapped_sec") or noise_total_duration_sec)

    if noise_samples is None or noise_samples.size < int(0.1 * sr_eff):
        segs = []
        total = 0.0
        if win_map:
            for st, en in win_map:
                st = max(0, int(st)); en = min(int(en), N)
                if en > st:
                    seg = x[st:en]
                    if seg.size:
                        segs.append(seg)
                        total += (en - st) / sr_eff
                        if total >= 10.0:
                            break
        noise_samples = np.concatenate(segs).astype(np.float64, copy=False) if segs else x

    def _nf_cross_rms(sig: np.ndarray, sr_local: int):
        win = int(round(0.4 * sr_local))
        hop = max(1, int(round(0.1 * sr_local)))
        if sig.size < win or win <= 0 or hop <= 0:
            r = calculate_rms(sig)
            return dbfs(r) if r > 0 else -np.inf
        x2 = sig * sig
        csum = np.empty(x2.size + 1, dtype=np.float64)
        csum[0] = 0.0
        np.cumsum(x2, out=csum[1:])
        frames = list(range(0, sig.size - win + 1, hop))
        if not frames:
            r = calculate_rms(sig)
            return dbfs(r) if r > 0 else -np.inf
        idx = np.array(frames)
        rms_v = np.sqrt((csum[idx + win] - csum[idx]) / float(win))
        with np.errstate(divide='ignore'):
            rms_db = 20.0 * np.log10(np.maximum(rms_v, 1e-30))
        rms_db = rms_db[np.isfinite(rms_db)]
        if rms_db.size >= 10:
            return float(np.percentile(rms_db, 10))
        if rms_db.size > 0:
            return float(np.median(rms_db))
        r = calculate_rms(sig)
        return dbfs(r) if r > 0 else -np.inf

    nf_cross = _nf_cross_rms(x, sr_eff)

    f_hi = min(20000.0, 0.98 * (sr_eff / 2.0))

    def _welch_once(sig: np.ndarray, sr_local: int):
        try:
            from scipy.signal import welch
            target = 65536
            nperseg = min(sig.size, target)
            if nperseg < 2048:
                nperseg = max(512, nperseg)
            noverlap = nperseg // 2
            freqs, Pxx = welch(sig, fs=sr_local, window='hann', nperseg=nperseg, noverlap=noverlap, scaling='spectrum', detrend='constant')
            return freqs.astype(np.float64, copy=False), Pxx.astype(np.float64, copy=False)
        except Exception:
            r = calculate_rms(sig)
            return None, None

    freqs_w, Pxx_w = _welch_once(noise_samples, sr_eff)

    def _nf_total_from_welch(freqs, Pxx):
        if freqs is None or Pxx is None:
            r = calculate_rms(noise_samples)
            return dbfs(r) if r > 0 else -np.inf
        band = (freqs >= 20.0) & (freqs <= f_hi)
        if not np.any(band):
            r = calculate_rms(noise_samples)
            return dbfs(r) if r > 0 else -np.inf
        spec = Pxx[band]
        if spec.size < 16:
            power_band = float(np.sum(spec))
            nf_rms = math.sqrt(power_band) if power_band > 0 else 0.0
            return dbfs(nf_rms) if nf_rms > 0 else -np.inf
        try:
            from scipy.signal import fftconvolve
            eps = 1e-30
            log_spec = np.log(np.maximum(spec, eps))
            k = max(31, int(round(spec.size * 0.01)))
            if k % 2 == 0:
                k += 1
            if k >= spec.size:
                k = max(31, spec.size // 3 if (spec.size // 3) % 2 == 1 else max(31, spec.size // 3 - 1))
            ker = np.ones(k, dtype=np.float64) / k
            smooth_log = fftconvolve(log_spec, ker, mode='same')
            baseline = np.exp(smooth_log)
            thr = baseline * (10.0 ** (8.0 / 10.0))
            spec_clip = np.minimum(spec, thr)
            power_band = float(np.sum(spec_clip))
            nf_rms = math.sqrt(power_band) if power_band > 0 else 0.0
            return dbfs(nf_rms) if nf_rms > 0 else -np.inf
        except Exception:
            r = calculate_rms(noise_samples)
            return dbfs(r) if r > 0 else -np.inf

    nf_dbfs = _nf_total_from_welch(freqs_w, Pxx_w)

    def _nf_broadband_from_welch(freqs, Pxx, hum_f0: float | None):
        if freqs is None or Pxx is None:
            return nf_dbfs
        band = (freqs >= 20.0) & (freqs <= f_hi)
        if not np.any(band):
            r = calculate_rms(noise_samples)
            return dbfs(r) if r > 0 else -np.inf
        spec = Pxx.copy()
        if hum_f0 is not None and hum_f0 > 0:
            f0 = float(hum_f0)
            k = 1
            while True:
                fk = f0 * k
                if fk > f_hi:
                    break
                if k <= 3:
                    pct = 0.025
                elif k <= 8:
                    pct = 0.015
                else:
                    pct = 0.010
                bw = max(1.0, fk * pct)
                core = (freqs >= fk - bw) & (freqs <= fk + bw)
                ring = (freqs >= fk - 5.0 * bw) & (freqs <= fk + 5.0 * bw) & (~core)
                if np.any(core):
                    if np.any(ring):
                        back = float(np.median(spec[ring]))
                    else:
                        back = float(np.median(spec[band]))
                    if np.isfinite(back):
                        spec[core] = np.minimum(spec[core], back)
                k += 1
        band_spec = spec[band]
        power_band = float(np.sum(band_spec))
        nf_rms = math.sqrt(power_band) if power_band > 0 else 0.0
        return dbfs(nf_rms) if nf_rms > 0 else -np.inf

    hum_f0_for_nf = None
    spur_label_tmp = str(wm.get("spur_label") or "")
    if isinstance(notch_f0, (int, float)) and np.isfinite(float(notch_f0)) and float(notch_f0) > 0:
        hum_f0_for_nf = float(notch_f0)
    elif "HUM" in spur_label_tmp:
        if "50" in spur_label_tmp:
            hum_f0_for_nf = 50.0
        elif "60" in spur_label_tmp:
            hum_f0_for_nf = 60.0

    nf_broadband = _nf_broadband_from_welch(freqs_w, Pxx_w, hum_f0_for_nf)

    diff = float(abs(nf_cross - nf_dbfs)) if np.isfinite(nf_cross) and np.isfinite(nf_dbfs) else 0.0

    eps = 1e-20
    anal_segments = []
    if win_map and len(win_map) >= 3:
        for st, en in win_map:
            st = max(0, int(st)); en = min(int(en), N)
            if en > st:
                seg = x[st:en]
                if seg.size >= int(0.25 * sr_eff):
                    anal_segments.append(seg)
    if not anal_segments:
        ns = noise_samples
        W = int(round(min(1.0, max(0.3, ns.size / sr_eff / 10.0)) * sr_eff))
        if W < int(0.25 * sr_eff):
            W = int(0.25 * sr_eff)
        if ns.size < W:
            anal_segments = [ns]
        else:
            step = W
            for i in range(0, ns.size - W + 1, step):
                anal_segments.append(ns[i:i + W])
            if not anal_segments:
                anal_segments = [ns]

    def _parabolic_peak(f, y, i):
        if i <= 0 or i >= y.size - 1:
            return f[i], y[i]
        y_m1 = y[i - 1]; y_0 = y[i]; y_p1 = y[i + 1]
        denom = (y_m1 - 2.0 * y_0 + y_p1)
        if abs(denom) < 1e-30:
            return f[i], y[i]
        delta = 0.5 * (y_m1 - y_p1) / denom
        delta = float(np.clip(delta, -0.5, 0.5))
        df = f[1] - f[0] if f.size > 1 else 0.0
        return float(f[i] + delta * df), float(y[i] - 0.25 * (y_m1 - y_p1) * delta)

    tops_freq = []
    hum50_counts = []
    hum60_counts = []
    whine_freqs = []

    for seg in anal_segments:
        Nw = seg.size
        nfft = int(2 ** math.ceil(math.log2(max(64, min(Nw * 2, 1_048_576)))))
        w = np.hanning(Nw)
        y = seg * w
        spec = np.abs(np.fft.rfft(y, n=nfft)) + eps
        freqs = np.fft.rfftfreq(nfft, 1 / sr_eff)
        band_mask = (freqs >= 20.0) & (freqs <= f_hi)
        if not np.any(band_mask):
            continue
        band_spec = spec[band_mask]
        band_freqs = freqs[band_mask]
        try:
            from scipy.signal import fftconvolve
            log_spec = np.log(band_spec)
            k = max(31, int(round(band_spec.size * 0.01)))
            if k % 2 == 0:
                k += 1
            if k >= band_spec.size:
                k = max(31, band_spec.size // 3 if (band_spec.size // 3) % 2 == 1 else max(31, band_spec.size // 3 - 1))
            ker = np.ones(k, dtype=np.float64) / k
            smooth_log = fftconvolve(log_spec, ker, mode='same')
            baseline = np.exp(smooth_log)
            prom_db = 20.0 * (np.log10(band_spec) - np.log10(baseline))
        except Exception:
            med = float(np.median(band_spec))
            prom_db = 20.0 * np.log10(band_spec / max(med, eps))
        idx_max = int(np.argmax(prom_db))
        f_refined, _ = _parabolic_peak(band_freqs, band_spec, idx_max)
        tops_freq.append(float(f_refined))

        def _harm_count(f0):
            cnt = 0
            k = 1
            while True:
                fk = f0 * k
                if fk > f_hi:
                    break
                bw = max(1.0, fk * 0.01)
                mask_k = (band_freqs >= fk - bw) & (band_freqs <= fk + bw)
                if np.any(mask_k):
                    pk = float(np.max(prom_db[mask_k]))
                    if pk >= 6.0:
                        cnt += 1
                k += 1
            return cnt

        hum50_counts.append(_harm_count(50.0))
        hum60_counts.append(_harm_count(60.0))

        mask_w = band_freqs >= 200.0
        if np.any(mask_w):
            prom_w = prom_db[mask_w]
            freqs_w2 = band_freqs[mask_w]
            idxw = int(np.argmax(prom_w))
            fw, _ = _parabolic_peak(freqs_w2, np.exp(prom_w / 20.0), idxw) if (idxw > 0 and idxw < prom_w.size - 1) else (float(freqs_w2[idxw]), None)
            pw = float(prom_w[idxw])
            if pw >= 10.0:
                whine_freqs.append(fw)

    def _stability_fraction(freq_list, tol_hz_min=5.0, tol_rel=0.005):
        if not freq_list:
            return 0.0, None
        arr = np.array(freq_list, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return 0.0, None
        center = float(np.median(arr))
        tol = max(tol_hz_min, center * tol_rel)
        sel = np.abs(arr - center) <= tol
        frac = float(np.sum(sel)) / float(arr.size)
        return frac, center

    hum50_frac = float(np.mean([1.0 if c >= 2 else 0.0 for c in hum50_counts])) if hum50_counts else 0.0
    hum60_frac = float(np.mean([1.0 if c >= 2 else 0.0 for c in hum60_counts])) if hum60_counts else 0.0
    whine_frac, whine_center = _stability_fraction(whine_freqs)
    tops_frac, tops_center = _stability_fraction(tops_freq)

    spur_label = None
    spur_f0 = None
    spur_hcount = 0
    cap_harm = 20

    if hum50_frac >= 0.6 or hum60_frac >= 0.6:
        if hum50_frac >= hum60_frac:
            spur_f0 = 50.0
            counts = [c for c in hum50_counts if c >= 1]
        else:
            spur_f0 = 60.0
            counts = [c for c in hum60_counts if c >= 1]
        if counts:
            spur_hcount = int(round(float(np.median(counts))))
        spur_hcount = int(min(cap_harm, max(0, spur_hcount)))
        if spur_hcount >= 15:
            spur_label = f"HUM{int(spur_f0)} (denso)"
        elif spur_hcount >= 2:
            spur_label = f"HUM{int(spur_f0)} + H{spur_hcount}"
        else:
            spur_label = f"HUM{int(spur_f0)}"
    elif whine_frac >= 0.6 and whine_center and np.isfinite(whine_center):
        spur_label = f"WHINE {whine_center/1000.0:.1f}k"
        spur_f0 = float(whine_center)
        spur_hcount = 1

    est_hc = int(wm.get("hum_estimated_hcount") or 0)
    if spur_label is None:
        if (est_hc >= 8) or ((isinstance(notch_f0, (int, float)) and np.isfinite(float(notch_f0)) and float(notch_f0) > 0.0) and (nf_notch_reduction_db >= 1.0 or notch_harms >= 4)):
            f0_guess = float(notch_f0) if (isinstance(notch_f0, (int, float)) and np.isfinite(float(notch_f0)) and float(notch_f0) > 0.0) else 50.0
            spur_f0 = f0_guess
            spur_hcount = int(min(cap_harm, max(spur_hcount, est_hc if est_hc > 0 else 2)))
            if spur_hcount >= 15:
                spur_label = f"HUM{int(spur_f0)} (denso)"
            elif spur_hcount >= 2:
                spur_label = f"HUM{int(spur_f0)} + H{spur_hcount}"
            else:
                spur_label = f"HUM{int(spur_f0)}"

    y_all = noise_samples * np.hanning(noise_samples.size)
    nfft_all = int(2 ** math.ceil(math.log2(max(64, min(y_all.size * 2, 1_048_576)))))
    spec_all = np.abs(np.fft.rfft(y_all, n=nfft_all)) + 1e-20
    freqs_all = np.fft.rfftfreq(nfft_all, 1 / sr_eff)
    band_all = (freqs_all >= 20.0) & (freqs_all <= f_hi)
    spur_db = 0.0
    if np.any(band_all):
        band_spec_all = spec_all[band_all]
        try:
            from scipy.signal import fftconvolve
            log_spec_all = np.log(band_spec_all)
            k_all = max(31, int(round(band_spec_all.size * 0.01)))
            if k_all % 2 == 0:
                k_all += 1
            if k_all >= band_spec_all.size:
                k_all = max(31, band_spec_all.size // 3 if (band_spec_all.size // 3) % 2 == 1 else max(31, band_spec_all.size // 3 - 1))
            ker_all = np.ones(k_all, dtype=np.float64) / k_all
            smooth_log_all = fftconvolve(log_spec_all, ker_all, mode='same')
            baseline_all = np.exp(smooth_log_all)
            med_all = float(np.median(band_spec_all))
            if med_all > 0:
                if spur_f0 and np.isfinite(spur_f0):
                    tol = max(5.0, spur_f0 * 0.005)
                    mask_sel = (freqs_all >= (spur_f0 - tol)) & (freqs_all <= (spur_f0 + tol)) & band_all
                    if np.any(mask_sel):
                        sub_f = freqs_all[mask_sel]
                        sub_y = band_spec_all[(freqs_all[band_all] >= (spur_f0 - tol)) & (freqs_all[band_all] <= (spur_f0 + tol))]
                        idx = int(np.argmax(sub_y))
                        f_pk, y_pk = _parabolic_peak(sub_f, sub_y, idx)
                        spur_db = float(max(0.0, 20.0 * math.log10(max(y_pk, 1e-20) / med_all)))
                    else:
                        idx = int(np.argmax(band_spec_all))
                        spur_db = float(max(0.0, 20.0 * math.log10(float(band_spec_all[idx]) / med_all)))
                else:
                    idx = int(np.argmax(band_spec_all))
                    spur_db = float(max(0.0, 20.0 * math.log10(float(band_spec_all[idx]) / med_all)))
        except Exception:
            med_all = float(np.median(band_spec_all))
            if med_all > 0:
                idx = int(np.argmax(band_spec_all))
                spur_db = float(max(0.0, 20.0 * math.log10(float(band_spec_all[idx]) / med_all)))

    if not np.isfinite(nf_dbfs):
        nf_dbfs = -np.inf
    if not np.isfinite(nf_broadband):
        nf_broadband = nf_dbfs
    if not np.isfinite(spur_db) or spur_db < 0:
        spur_db = 0.0

    return {
        "nf_dbfs": float(nf_dbfs),
        "nf_broadband_dbfs": float(nf_broadband),
        "spur_db": float(spur_db),
        "nf_cross_rms_dbfs": float(nf_cross),
        "nf_consistency_diff_db": float(diff),
        "spur_label": spur_label,
        "spur_fundamental_hz": None if spur_f0 is None else float(spur_f0),
        "spur_harmonics_count": int(spur_hcount),
        "noise_windows_count": int(noise_windows_count),
        "noise_total_duration_sec": float(noise_total_duration_sec),
        "noise_total_duration_uncapped_sec": float(noise_total_duration_uncapped_sec),
        "nf_notch_reduction_db": float(nf_notch_reduction_db),
        "notch_f0": float(notch_f0) if isinstance(notch_f0, (int, float)) and np.isfinite(float(notch_f0)) else (50.0 if spur_label and spur_label.startswith("HUM") and "50" in spur_label else (60.0 if spur_label and spur_label.startswith("HUM") and "60" in spur_label else None)),
        "notch_harmonics": int(notch_harms),
        "music_limited": bool(music_limited),
        "noise_rms_p90_dbfs": float(noise_rms_p90_dbfs) if isinstance(noise_rms_p90_dbfs, (int, float)) and np.isfinite(float(noise_rms_p90_dbfs)) else None,
        "hum_estimated_hcount": int(wm.get("hum_estimated_hcount") or 0)
    }

def order_jobs_by_weight(files_to_analyze_fully_info):
    import os, math, subprocess
    try:
        native = os.environ.get("AQC_NATIVE_SR", "").strip().lower() in ("1", "true", "on", "yes", "y")
    except Exception:
        native = False
    def _float_env(name, default):
        v = os.environ.get(name)
        if v is None:
            return default
        try:
            f = float(v)
            if not math.isfinite(f):
                return default
            return f
        except Exception:
            return default
    sr_exp = _float_env("AQC_JOB_WEIGHT_SR_EXP", 1.0 if native else 0.6)
    dur_exp = _float_env("AQC_JOB_WEIGHT_DUR_EXP", 1.0)
    ch_exp = _float_env("AQC_JOB_WEIGHT_CH_EXP", 0.2)
    def _probe_info(path):
        try:
            info = sf.info(path)
            sr = int(getattr(info, "samplerate", 0) or 0)
            frames = int(getattr(info, "frames", 0) or 0)
            ch = int(getattr(info, "channels", 2) or 2)
            dur = float(frames) / float(sr) if (sr and frames) else 0.0
            return sr, dur, ch
        except Exception:
            try:
                out = subprocess.run(
                    ['ffprobe', '-v', 'error', '-select_streams', 'a:0',
                     '-show_entries', 'stream=sample_rate,duration,channels',
                     '-of', 'default=noprint_wrappers=1:nokey=1', path],
                    check=False, capture_output=True, text=True, encoding='utf-8', errors='replace'
                )
                vals = [t.strip() for t in (out.stdout or "").splitlines() if t.strip()]
                sr = 0
                dur = 0.0
                ch = 2
                for t in vals:
                    if t.isdigit():
                        if sr == 0:
                            sr = int(t)
                        else:
                            ch = int(t)
                    else:
                        try:
                            v = float(t.replace(",", "."))
                            if v > 0:
                                dur = v
                        except Exception:
                            pass
                if sr <= 0:
                    sr = 48000
                if dur <= 0:
                    try:
                        sz = os.path.getsize(path)
                        dur = max(60.0, min(1800.0, sz / (512.0 * 1024.0)))
                    except Exception:
                        dur = 180.0
                if ch <= 0:
                    ch = 2
                return sr, dur, ch
            except Exception:
                return 48000, 180.0, 2
    items = []
    for it in files_to_analyze_fully_info or []:
        try:
            p = it[0]
        except Exception:
            continue
        sr, dur, ch = _probe_info(p)
        sr_norm = max(48000.0, float(sr))
        w = (max(1.0, float(dur)) ** dur_exp) * ((sr_norm / 48000.0) ** sr_exp) * (max(1.0, float(ch)) ** ch_exp)
        items.append((w, it))
    if not items:
        return list(files_to_analyze_fully_info or [])
    items.sort(key=lambda t: (t[0], os.path.basename(t[1][0])), reverse=True)
    return [it for _, it in items]

def calculate_score_float(r):
    import os
    pol = _policy_resolve()
    thr_all = pol["thresholds"]
    wH_base = dict(pol["weights"]["H"])
    wQ_base = dict(pol["weights"]["Q"])
    mix_base = pol["score_mix"]
    conf_w = pol["confidence_weights"]

    def _tier_from_thr(val, thr):
        if thr is None or val is None:
            return None
        try:
            v = float(val)
            if not np.isfinite(v):
                return None
        except (TypeError, ValueError):
            return None
        good = thr.get("good"); warn = thr.get("warn"); bad = thr.get("bad")
        hib = thr.get("higher_is_better", True)
        if good is None or warn is None or bad is None:
            return None
        if hib:
            if v >= good:          return 4
            elif v >= warn:        return 3
            elif v >  bad:         return 2
            else:                  return 1
        else:
            if v <= good:          return 4
            elif v <= warn:        return 3
            elif v <  bad:         return 2
            else:                  return 1

    def _safe_float_local(x, d=None):
        try:
            f = float(x)
            return f if np.isfinite(f) else d
        except Exception:
            return d

    def _strictize_thr(key, thr):
        import copy, os
        prof = os.environ.get("AQC_PROFILE", "").strip().lower()
        if not thr or not isinstance(thr, dict) or not (prof.startswith("strict") or prof == "audiophile-strict"):
            return thr
        t = copy.deepcopy(thr)
        if key == "isp_margin_db":
            t["good"] = max(t.get("good", 1.0), 1.5)
            t["warn"] = max(t.get("warn", 0.5), 1.0)
        elif key == "spectral_balance_dev_db":
            t["good"] = min(t.get("good", 2.0), 1.5)
            t["warn"] = min(t.get("warn", 4.0), 3.0)
        elif key == "lra_est":
            t["good"] = max(t.get("good", 12.0), 14.0)
            t["warn"] = max(t.get("warn", 6.0), 10.0)
        return t

    tp = _safe_float_local(r.get("true_peak_est_dbtp", r.get("peak_dbfs_overall")), None)
    hard_thr = pol["clip_classes"]["hard_tp_over"]
    if tp is not None and np.isfinite(tp):
        if tp > hard_thr:
            clip_class = "hard"
        elif tp > 0.0:
            clip_class = "borderline"
        else:
            clip_class = "clean"
    else:
        clip_class = "clean"
    r["clipping_class"] = clip_class

    nf_val = _safe_float_local(r.get("noise_floor_dbfs"), None)
    nf_bb_val = _safe_float_local(r.get("nf_broadband_dbfs", nf_val), None)
    nf_profile_key = _policy_select_nf_profile(nf_val, r, pol)
    eval_bits = 24 if nf_profile_key and nf_profile_key.endswith("_24") else 16
    r["hygiene_eval_bits"] = eval_bits

    conf_label_raw = (r.get("noise_confidence") or "").strip().lower()
    if conf_label_raw.startswith("alt"):   conf_label = "Alta"
    elif conf_label_raw.startswith("med"): conf_label = "Media"
    elif conf_label_raw.startswith("bas"): conf_label = "Bassa"
    else:                                  conf_label = "Alta"
    conf_scale = conf_w.get(conf_label.split()[0].capitalize(), 1.0)

    spur_label = str(r.get("noise_spur_label") or "")
    spur_hc = int(r.get("noise_spur_harmonics_count") or 0)
    hum_dense = spur_label.startswith("HUM") and spur_hc >= 10

    music_limited = bool(r.get("music_limited")) if r.get("music_limited") is not None else False
    nf_delta = _safe_float_local(r.get("noise_consistency_diff_db"), 0.0) or 0.0
    nwin = int(r.get("noise_windows_count") or 0)
    ndur = _safe_float_local(r.get("noise_total_duration_sec"), 0.0) or 0.0

    nf_use_for_score = nf_bb_val if (hum_dense and nf_bb_val is not None and np.isfinite(nf_bb_val)) else nf_val
    nf_gate_hi = nf_use_for_score is not None and np.isfinite(nf_use_for_score) and nf_use_for_score > -35.0

    extreme_uncertain  = (nf_delta > 30.0) or ((nwin < 2) and (ndur < 3.0))
    moderate_uncertain = (nf_delta > 20.0) or (nwin < 2 or ndur < 3.0)

    mixH_eff = float(mix_base["H"])
    mixQ_eff = float(mix_base["Q"])
    if extreme_uncertain:
        mixH_eff = 0.78
        mixQ_eff = 0.22
    elif moderate_uncertain:
        mixH_eff = 0.76
        mixQ_eff = 0.24
    if hum_dense:
        mixH_eff = max(mixH_eff, 0.80)
        mixQ_eff = 1.0 - mixH_eff
    if music_limited:
        mixH_eff = max(mixH_eff, 0.70)
        mixQ_eff = 1.0 - mixH_eff
    if conf_label.startswith("Bassa"):
        mixH_eff = max(mixH_eff, 0.75)
        mixQ_eff = 1.0 - mixH_eff

    w_nf0 = wH_base["noise_floor_dbfs"] * conf_scale
    w_sp0 = wH_base["noise_spur_db"]    * conf_scale

    w_nf_eff = w_nf0
    w_sp_eff = w_sp0
    w_isp_eff = wH_base["isp_margin_db"]
    w_dc_eff = wH_base["dc_offset_dbfs"]

    if hum_dense:
        w_nf_eff = w_nf0 * 0.6
        w_sp_eff = w_sp0 * 1.05
    if music_limited:
        w_nf_eff *= 0.7
        w_sp_eff *= 0.90
    if conf_label.startswith("Bassa"):
        if nf_delta > 10.0:
            w_nf_eff *= 0.6
            w_sp_eff *= 0.85
        elif nf_delta > 8.0:
            w_nf_eff *= 0.7
            w_sp_eff *= 0.90
        elif nf_delta > 6.0:
            w_nf_eff *= 0.8
            w_sp_eff *= 0.95
    if extreme_uncertain:
        w_nf_eff = 0.0
        w_sp_eff *= 1.00
        w_isp_eff *= 1.15
    elif moderate_uncertain:
        w_nf_eff *= 0.50
        w_sp_eff *= 0.95
        w_isp_eff *= 1.10

    wH_eff_map = {
        "noise_floor_dbfs": w_nf_eff,
        "noise_spur_db":    w_sp_eff,
        "isp_margin_db":    w_isp_eff,
        "dc_offset_dbfs":   w_dc_eff,
    }

    h_num = 0.0
    h_den = 0.0

    safe_env = os.environ.get("AQC_NF_SAFE", "").strip().lower()
    nf_safe = (safe_env not in ("0", "false", "off", "no", "n"))

    thr_nf = thr_all.get(nf_profile_key) if nf_profile_key else None
    if nf_safe:
        thr16 = thr_all.get("noise_floor_dbfs_16")
        thr24 = thr_all.get("noise_floor_dbfs_24")
        s16 = policy_map_to_unit_score(nf_use_for_score, thr16) if thr16 is not None else None
        s24 = policy_map_to_unit_score(nf_use_for_score, thr24) if thr24 is not None else None
        cand = [x for x in (s16, s24) if x is not None]
        s_nf = max(cand) if cand else policy_map_to_unit_score(nf_use_for_score, thr_nf)
        r["nf_scoring_mode"] = "SAFE"
    else:
        s_nf = policy_map_to_unit_score(nf_use_for_score, thr_nf)
        r["nf_scoring_mode"] = "PROFILE_24" if (nf_profile_key and nf_profile_key.endswith("_24")) else "PROFILE_16"

    if s_nf is not None and wH_eff_map["noise_floor_dbfs"] > 0:
        h_num += s_nf * wH_eff_map["noise_floor_dbfs"]
        h_den += wH_eff_map["noise_floor_dbfs"]

    val_spur = _safe_float_local(r.get("noise_spur_db"), None)
    thr_sp = _policy_metric_thresholds("noise_spur_db", val_spur, pol, r=r)
    s_spur = policy_map_to_unit_score(val_spur, thr_sp)
    if s_spur is not None and wH_eff_map["noise_spur_db"] > 0:
        h_num += s_spur * wH_eff_map["noise_spur_db"]
        h_den += wH_eff_map["noise_spur_db"]

    val_isp = _safe_float_local(r.get("isp_margin_db"), None)
    thr_is = _policy_metric_thresholds("isp_margin_db", val_isp, pol, r=r)
    thr_is = _strictize_thr("isp_margin_db", thr_is)
    s_isp = policy_map_to_unit_score(val_isp, thr_is)
    if s_isp is not None and wH_eff_map["isp_margin_db"] > 0:
        h_num += s_isp * wH_eff_map["isp_margin_db"]
        h_den += wH_eff_map["isp_margin_db"]

    val_dc = _safe_float_local(r.get("dc_offset_dbfs"), None)
    thr_dc = _policy_metric_thresholds("dc_offset_dbfs", val_dc, pol, r=r)
    s_dc = policy_map_to_unit_score(val_dc, thr_dc)
    if s_dc is not None and wH_eff_map["dc_offset_dbfs"] > 0:
        h_num += s_dc * wH_eff_map["dc_offset_dbfs"]
        h_den += wH_eff_map["dc_offset_dbfs"]

    H = (h_num / h_den) if h_den > 0 else 0.5
    r["hygiene_score"] = H

    plr_val = _safe_float_local(r.get("plr_effective_db", r.get("plr_est")), None)
    wQ_eff_map = dict(wQ_base)
    if conf_label.startswith("Bassa") or music_limited or extreme_uncertain or (H is not None and H < 0.45):
        total = sum(wQ_eff_map.values())
        wQ_eff_map["spectral_balance_dev_db"] = 0.20
        wQ_eff_map["stereo_width_iqr_db"] = 0.01
        wQ_eff_map["lra_est"] = 0.14
        wQ_eff_map["dr_tt_avg"] = 0.30
        wQ_eff_map["plr_est"] = 0.20
        wQ_eff_map["st_lufs_iqr_db"] = 0.15
        ssum = sum(wQ_eff_map.values())
        if ssum > 0 and abs(ssum - total) > 1e-12:
            k = total / ssum
            for kk in wQ_eff_map:
                wQ_eff_map[kk] *= k

    q_specs = [
        ("dr_tt_avg",               wQ_eff_map["dr_tt_avg"]),
        ("plr_effective_db",        wQ_eff_map["plr_est"]),
        ("lra_est",                 wQ_eff_map["lra_est"]),
        ("st_lufs_iqr_db",          wQ_eff_map["st_lufs_iqr_db"]),
        ("spectral_balance_dev_db", wQ_eff_map["spectral_balance_dev_db"]),
        ("stereo_width_iqr_db",     wQ_eff_map["stereo_width_iqr_db"]),
    ]
    q_num = 0.0
    q_den = 0.0
    for key, w in q_specs:
        if w <= 0:
            continue
        if key == "plr_effective_db":
            val = plr_val
            thr = _policy_metric_thresholds("plr_est", val, pol, r=r)
        else:
            val = _safe_float_local(r.get(key), None)
            thr = _policy_metric_thresholds(key, val, pol, r=r)
        if key in ("isp_margin_db", "lra_est", "spectral_balance_dev_db"):
            thr = _strictize_thr("isp_margin_db" if key == "isp_margin_db" else key, thr)
        s = policy_map_to_unit_score(val, thr)
        if key == "stereo_width_iqr_db" and s is not None and val is not None and np.isfinite(val):
            if val > 10.0:
                factor = max(0.4, min(1.0, 10.0 / float(val)))
                s = s * factor
        if s is None:
            continue
        q_num += s * w
        q_den += w
    Q = (q_num / q_den) if q_den > 0 else 0.5
    r["quality_score"] = Q

    try:
        if ('nf_interval_unc_db' not in r) or (r.get('nf_interval_unc_db') is None):
            compute_nf_interval_fields(r)
    except Exception:
        pass

    fine_points = 0.0
    nf_unc = _safe_float_local(r.get('nf_interval_unc_db'), None)
    if nf_unc is not None and np.isfinite(nf_unc):
        if nf_unc <= 2.0:
            fine_points += min(0.8, (2.0 - nf_unc) * 0.4)
        elif nf_unc >= 5.0:
            fine_points -= min(0.8, (nf_unc - 5.0) * 0.2)
    f05 = _safe_float_local(r.get('isp_under_05db_frac_50ms', r.get('isp_under_05db_frac')), None)
    f02 = _safe_float_local(r.get('isp_under_02db_frac_50ms', r.get('isp_under_02db_frac')), None)
    if f05 is not None and np.isfinite(f05):
        fine_points -= min(0.6, 1.2 * max(0.0, f05))
    if f02 is not None and np.isfinite(f02):
        fine_points -= min(0.6, 1.6 * max(0.0, f02))
    tag = (r.get('isp_multi_consistency') or '').strip().lower()
    if tag == "safe":
        fine_points += 0.2
    elif tag == "spiky":
        fine_points -= 0.2
    elif tag == "dense":
        fine_points -= 0.15

    spur_db_val = _safe_float_local(r.get("noise_spur_db"), 0.0) or 0.0
    h_norm_spur = min(1.0, max(0.0, (spur_db_val - 6.0) / 12.0))
    h_norm_hc = min(1.0, max(0.0, spur_hc) / 15.0)
    nf_red = _safe_float_local(r.get("nf_notch_reduction_db"), 0.0) or 0.0
    h_norm_notch = min(1.0, max(0.0, nf_red) / 6.0)
    hum_raw = 0.55 * h_norm_spur + 0.30 * h_norm_hc + 0.40 * h_norm_notch
    hum_penalty = (1.05 * (1.0 - math.exp(-1.35 * hum_raw))) + (0.25 * max(0.0, hum_raw - 0.85))
    micro_tail = 0.05 * min(1.0, max(0.0, (spur_db_val - 30.0) / 6.0))
    hum_penalty += micro_tail
    if hum_penalty > 1.5:
        hum_penalty = 1.5
    fine_points -= hum_penalty

    if music_limited and not extreme_uncertain:
        fine_points -= 0.20
    fine_points = max(-1.8, min(1.8, fine_points))

    base_score_raw = 100.0 * (mixH_eff * H + mixQ_eff * Q)
    base_score = base_score_raw + fine_points
    r["score_pre_cap"] = round(base_score, 2)
    r["score_mix_eff"] = {"H": mixH_eff, "Q": mixQ_eff}
    r["weights_H_eff"] = dict(wH_eff_map)
    r["weights_Q_eff"] = dict(wQ_eff_map)

    tier_metrics = []
    tier_nf_thr = thr_all.get(nf_profile_key) if nf_profile_key else None
    tier_metrics.append(_tier_from_thr(nf_use_for_score, tier_nf_thr))
    tier_sp = _policy_metric_thresholds("noise_spur_db", _safe_float_local(r.get("noise_spur_db"), None), pol, r=r)
    tier_is = _policy_metric_thresholds("isp_margin_db", _safe_float_local(r.get("isp_margin_db"), None), pol, r=r)
    tier_dc = _policy_metric_thresholds("dc_offset_dbfs", _safe_float_local(r.get("dc_offset_dbfs"), None), pol, r=r)
    tier_metrics.append(_tier_from_thr(_safe_float_local(r.get("noise_spur_db"), None), tier_sp))
    tier_metrics.append(_tier_from_thr(_safe_float_local(r.get("isp_margin_db"), None), _strictize_thr("isp_margin_db", tier_is)))
    tier_metrics.append(_tier_from_thr(_safe_float_local(r.get("dc_offset_dbfs"), None), tier_dc))

    unknown_count = sum(1 for t in tier_metrics if t is None)

    hygiene_class = "DQ" if clip_class == "hard" else "OK"
    cap = 100.0
    r["hygiene_class"] = hygiene_class
    r["hygiene_cap"] = cap
    r["hygiene_unknown_count"] = unknown_count

    final_score = base_score
    final_score = max(0.0, min(100.0, final_score))
    r["score_float"] = round(final_score, 2)
    r["score"] = int(round(r["score_float"]))
    r["fine_score"] = round(fine_points, 2)
    r["assessment"] = get_verbal_assessment(r["score"], r)
    return r["score_float"]

def analyze_audio(filepath, is_temporary=False):
    results = {}
    try:
        ext = os.path.splitext(filepath)[1].lower()
        if ext in ('.dsf', '.dff'):
            print(f"-- {os.path.basename(filepath)}: formato DSD non supportato direttamente per l'analisi. Converti in PCM (FLAC/WAV) e riprova.")
            return None

        with sf.SoundFile(filepath) as f:
            if f.channels != 2:
                print(f"-- {os.path.basename(filepath)}: non stereo. Ignorato.")
                return None
            sr = f.samplerate
            if sr < 32000:
                print(f"-- {os.path.basename(filepath)}: SR troppo basso. Ignorato.")
                return None
            audio = f.read(dtype="float64")
            info = sf.info(filepath)
            stype = info.subtype

        if audio is None or audio.shape[0] < int(0.1 * sr):
            print(f"-- {os.path.basename(filepath)}: file troppo corto. Ignorato.")
            return None
        if not np.all(np.isfinite(audio)):
            print(f"-- {os.path.basename(filepath)}: contiene NaN/inf. Ignorato.")
            return None

        L, R = audio.T
        mono = (L + R) * 0.5
        audio_f32 = audio.astype(np.float32, copy=False)

        views = prepare_analysis_views(L, R, mono, sr, base_target_sr=48000)
        Lb = views["L_base"]
        Rb = views["R_base"]
        Mb = views["mono_base"]
        sr_b = views["sr_base"]

        results["filename"] = os.path.basename(filepath) + (" (tmp)" if is_temporary else "")
        results["filepath"] = filepath
        results["samplerate"] = sr
        results["subtype"] = stype
        results["channels"] = 2
        results["engine_version"] = "2.4.1-audiophile"

        if is_temporary:
            results["conversion_container"] = ext.lstrip('.')
            if sr == 44100 and stype and "16" in stype:
                results["conversion_profile"] = "C"
            elif sr >= 88200 and (stype and ("24" in stype or "32" in stype)):
                results["conversion_profile"] = "H"
            else:
                results["conversion_profile"] = None
            results["conversion_target_sr"] = sr
            results["conversion_verified"] = True

        with sf.SoundFile(filepath) as fdet:
            nominal_bits = _detect_nominal_bits(fdet)
        results["bit_depth"] = nominal_bits
        results["effective_bit_depth"] = (estimate_effective_bits(mono, nominal_bits) if nominal_bits else None)

        pkL, pkR = np.max(np.abs(L)), np.max(np.abs(R))
        results["peak_dbfs_fl_sample"] = dbfs(pkL) if pkL > 0 else -np.inf
        results["peak_dbfs_fr_sample"] = dbfs(pkR) if pkR > 0 else -np.inf
        results["rms_dbfs_fl"] = dbfs(calculate_rms(L)) if np.any(L) else -np.inf
        results["rms_dbfs_fr"] = dbfs(calculate_rms(R)) if np.any(R) else -np.inf

        tp = _true_peak_internal(audio, sr, os_factor=16)
        results["true_peak_est_dbtp"] = tp if tp is not None and np.isfinite(tp) else None
        results["peak_dbfs_overall"] = results["true_peak_est_dbtp"]
        results["isp_margin_db"] = _isp_margin_db(audio, sr, os_factor=16)

        isp_stats = compute_isp_stats(audio, sr, os_factor=16, win_s=0.05, hop_s=0.025)
        if isinstance(isp_stats, dict):
            results.update({
                "isp_window_count": isp_stats.get("isp_window_count"),
                "isp_margin_p05_db": isp_stats.get("isp_margin_p05_db"),
                "isp_margin_p50_db": isp_stats.get("isp_margin_p50_db"),
                "isp_margin_p95_db": isp_stats.get("isp_margin_p95_db"),
                "isp_under_1db_count": isp_stats.get("isp_under_1db_count"),
                "isp_under_05db_count": isp_stats.get("isp_under_05db_count"),
                "isp_under_02db_count": isp_stats.get("isp_under_02db_count"),
                "isp_under_1db_frac": isp_stats.get("isp_under_1db_frac"),
                "isp_under_05db_frac": isp_stats.get("isp_under_05db_frac"),
                "isp_under_02db_frac": isp_stats.get("isp_under_02db_frac"),
                "isp_window_count_20ms": isp_stats.get("isp_window_count_20ms"),
                "isp_margin_p05_db_20ms": isp_stats.get("isp_margin_p05_db_20ms"),
                "isp_margin_p50_db_20ms": isp_stats.get("isp_margin_p50_db_20ms"),
                "isp_margin_p95_db_20ms": isp_stats.get("isp_margin_p95_db_20ms"),
                "isp_under_1db_count_20ms": isp_stats.get("isp_under_1db_count_20ms"),
                "isp_under_05db_count_20ms": isp_stats.get("isp_under_05db_count_20ms"),
                "isp_under_02db_count_20ms": isp_stats.get("isp_under_02db_count_20ms"),
                "isp_under_1db_frac_20ms": isp_stats.get("isp_under_1db_frac_20ms"),
                "isp_under_05db_frac_20ms": isp_stats.get("isp_under_05db_frac_20ms"),
                "isp_under_02db_frac_20ms": isp_stats.get("isp_under_02db_frac_20ms"),
                "isp_window_count_50ms": isp_stats.get("isp_window_count_50ms"),
                "isp_margin_p05_db_50ms": isp_stats.get("isp_margin_p05_db_50ms"),
                "isp_margin_p50_db_50ms": isp_stats.get("isp_margin_p50_db_50ms"),
                "isp_margin_p95_db_50ms": isp_stats.get("isp_margin_p95_db_50ms"),
                "isp_under_1db_count_50ms": isp_stats.get("isp_under_1db_count_50ms"),
                "isp_under_05db_count_50ms": isp_stats.get("isp_under_05db_count_50ms"),
                "isp_under_02db_count_50ms": isp_stats.get("isp_under_02db_count_50ms"),
                "isp_under_1db_frac_50ms": isp_stats.get("isp_under_1db_frac_50ms"),
                "isp_under_05db_frac_50ms": isp_stats.get("isp_under_05db_frac_50ms"),
                "isp_under_02db_frac_50ms": isp_stats.get("isp_under_02db_frac_50ms"),
                "isp_multi_ratio_05": isp_stats.get("isp_multi_ratio_05"),
                "isp_multi_ratio_02": isp_stats.get("isp_multi_ratio_02"),
                "isp_multi_consistency": isp_stats.get("isp_multi_consistency"),
            })

        isp_p05_ref = results.get("isp_margin_p05_db_50ms", results.get("isp_margin_p05_db"))
        if isp_p05_ref is not None and np.isfinite(isp_p05_ref) and isp_p05_ref < 0.5:
            tp32 = _true_peak_internal(audio, sr, os_factor=32)
            if tp32 is not None and np.isfinite(tp32):
                results["true_peak_est_dbtp"] = tp32
                results["peak_dbfs_overall"] = tp32
                results["isp_margin_db"] = _isp_margin_db(audio, sr, os_factor=32)
                isp_stats32 = compute_isp_stats(audio, sr, os_factor=32, win_s=0.05, hop_s=0.025)
                if isinstance(isp_stats32, dict):
                    results.update({
                        "isp_window_count": isp_stats32.get("isp_window_count"),
                        "isp_margin_p05_db": isp_stats32.get("isp_margin_p05_db"),
                        "isp_margin_p50_db": isp_stats32.get("isp_margin_p50_db"),
                        "isp_margin_p95_db": isp_stats32.get("isp_margin_p95_db"),
                        "isp_under_1db_count": isp_stats32.get("isp_under_1db_count"),
                        "isp_under_05db_count": isp_stats32.get("isp_under_05db_count"),
                        "isp_under_02db_count": isp_stats32.get("isp_under_02db_count"),
                        "isp_under_1db_frac": isp_stats32.get("isp_under_1db_frac"),
                        "isp_under_05db_frac": isp_stats32.get("isp_under_05db_frac"),
                        "isp_under_02db_frac": isp_stats32.get("isp_under_02db_frac"),
                        "isp_window_count_20ms": isp_stats32.get("isp_window_count_20ms"),
                        "isp_margin_p05_db_20ms": isp_stats32.get("isp_margin_p05_db_20ms"),
                        "isp_margin_p50_db_20ms": isp_stats32.get("isp_margin_p50_db_20ms"),
                        "isp_margin_p95_db_20ms": isp_stats32.get("isp_margin_p95_db_20ms"),
                        "isp_under_1db_count_20ms": isp_stats32.get("isp_under_1db_count_20ms"),
                        "isp_under_05db_count_20ms": isp_stats32.get("isp_under_05db_count_20ms"),
                        "isp_under_02db_count_20ms": isp_stats32.get("isp_under_02db_count_20ms"),
                        "isp_under_1db_frac_20ms": isp_stats32.get("isp_under_1db_frac_20ms"),
                        "isp_under_05db_frac_20ms": isp_stats32.get("isp_under_05db_frac_20ms"),
                        "isp_under_02db_frac_20ms": isp_stats32.get("isp_under_02db_frac_20ms"),
                        "isp_window_count_50ms": isp_stats32.get("isp_window_count_50ms"),
                        "isp_margin_p05_db_50ms": isp_stats32.get("isp_margin_p05_db_50ms"),
                        "isp_margin_p50_db_50ms": isp_stats32.get("isp_margin_p50_db_50ms"),
                        "isp_margin_p95_db_50ms": isp_stats32.get("isp_margin_p95_db_50ms"),
                        "isp_under_1db_count_50ms": isp_stats32.get("isp_under_1db_count_50ms"),
                        "isp_under_05db_count_50ms": isp_stats32.get("isp_under_05db_count_50ms"),
                        "isp_under_02db_count_50ms": isp_stats32.get("isp_under_02db_count_50ms"),
                        "isp_under_1db_frac_50ms": isp_stats32.get("isp_under_1db_frac_50ms"),
                        "isp_under_05db_frac_50ms": isp_stats32.get("isp_under_05db_frac_50ms"),
                        "isp_under_02db_frac_50ms": isp_stats32.get("isp_under_02db_frac_50ms"),
                        "isp_multi_ratio_05": isp_stats32.get("isp_multi_ratio_05"),
                        "isp_multi_ratio_02": isp_stats32.get("isp_multi_ratio_02"),
                        "isp_multi_consistency": isp_stats32.get("isp_multi_consistency"),
                    })

        results["dc_offset_dbfs"] = _dc_offset_dbfs(mono)

        try:
            std_l, std_r = np.std(L), np.std(R)
            if std_l > 1e-9 and std_r > 1e-9:
                corr, _ = pearsonr(L, R)
                results["stereo_correlation"] = float(corr) if np.isfinite(corr) else 0.0
            else:
                results["stereo_correlation"] = 1.0 if std_l < 1e-9 and std_r < 1e-9 else 0.0
        except Exception:
            results["stereo_correlation"] = None

        loudL, loudR, _ = compute_dr_tt_true(Lb, Rb, sr_b)
        peak_for_dr = None
        if results["true_peak_est_dbtp"] is not None and np.isfinite(results["true_peak_est_dbtp"]):
            peak_for_dr = min(0.0, results["true_peak_est_dbtp"])
        dr_l = peak_for_dr - dbfs(loudL) if (peak_for_dr is not None and loudL > 1e-12) else None
        dr_r = peak_for_dr - dbfs(loudR) if (peak_for_dr is not None and loudR > 1e-12) else None
        results["dr_tt_fl"] = dr_l
        results["dr_tt_fr"] = dr_r
        results["dr_tt_avg"] = float(max(0.0, np.mean([dr_l, dr_r]))) if (dr_l is not None and dr_r is not None and np.isfinite(dr_l) and np.isfinite(dr_r)) else 0.0

        lufs, lra, loud_backend, lt_series = measure_loudness(filepath, audio_f32, sr)
        results["loudness_lufs"] = lufs
        results["lra_est"] = lra
        results["loudness_backend"] = loud_backend

        def _compute_plr_active(mono_f32, sr_local, tp_db, lt_precomputed=None):
            if tp_db is None or not np.isfinite(tp_db):
                return None
            if lt_precomputed is not None and isinstance(lt_precomputed, (list, tuple, np.ndarray)):
                lt_arr = np.asarray(lt_precomputed, dtype=float)
                lt_arr = lt_arr[np.isfinite(lt_arr)]
                if lt_arr.size >= 5:
                    med = float(np.median(lt_arr))
                    gate = max(-70.0, med - 10.0)
                    active = lt_arr[lt_arr > gate]
                    if active.size < max(5, int(0.1 * lt_arr.size)):
                        k = max(5, int(0.1 * lt_arr.size))
                        idx = np.argsort(lt_arr)[-k:]
                        active = lt_arr[idx]
                    p95 = float(np.percentile(active, 95))
                    if np.isfinite(p95):
                        return float(tp_db - p95)
            lt = None
            try:
                import pyloudnorm as pyln
                x_mono = mono_f32.astype(np.float32, copy=False)
                meter = pyln.Meter(sr_local, block_size=0.400, filter_class="K-weighting")
                lt = meter.loudness_time_series(x_mono)
                if lt is not None:
                    lt = np.array(lt, dtype=float)
                    lt = lt[np.isfinite(lt)]
            except Exception:
                lt = None
            if lt is None or lt.size < 5:
                win = int(round(0.4 * sr_local))
                hop = max(1, int(round(0.1 * sr_local)))
                vals = []
                nloc = mono_f32.size
                for i in range(0, nloc - win + 1, hop):
                    seg = mono_f32[i:i + win]
                    rms = calculate_rms(seg.astype(np.float64, copy=False))
                    if rms > 1e-12:
                        v = dbfs(rms)
                        if np.isfinite(v):
                            vals.append(v)
                if len(vals) < 5:
                    return None
                lt = np.array(vals, dtype=float)
            med = float(np.median(lt))
            gate = max(-70.0, med - 10.0)
            active = lt[lt > gate]
            if active.size < max(5, int(0.1 * lt.size)):
                k = max(5, int(0.1 * lt.size))
                idx = np.argsort(lt)[-k:]
                active = lt[idx]
            p95 = float(np.percentile(active, 95))
            if not np.isfinite(p95):
                return None
            return float(tp_db - p95)

        plr_active = _compute_plr_active(Mb.astype(np.float32, copy=False), sr_b, results.get("true_peak_est_dbtp"), lt_precomputed=lt_series)
        results["plr_active_db"] = plr_active if plr_active is not None and np.isfinite(plr_active) else None

        results["st_lufs_iqr_db"] = compute_st_lufs_iqr(Mb.astype(np.float32, copy=False), sr_b, lt_precomputed=lt_series) or None

        if lufs is not None and np.isfinite(lufs) and results.get("true_peak_est_dbtp") is not None and np.isfinite(results.get("true_peak_est_dbtp")):
            results["plr_est"] = float(results["true_peak_est_dbtp"] - lufs)
        else:
            results["plr_est"] = None

        if results.get("plr_est") is not None and results.get("plr_active_db") is not None:
            results["plr_effective_db"] = float(min(results["plr_est"], results["plr_active_db"]))
        else:
            results["plr_effective_db"] = results.get("plr_est") if results.get("plr_est") is not None else results.get("plr_active_db")

        try:
            N_b = len(Mb)
            if N_b >= 1024:
                window = np.hanning(N_b)
                windowed = Mb * window
                coherent_gain = window.sum() / N_b
                nfft = int(2 ** math.ceil(math.log2(N_b)))
                yf = np.fft.rfft(windowed, n=nfft) / (N_b * coherent_gain)
                xf = np.fft.rfftfreq(nfft, 1 / sr_b)
                mag = np.abs(yf) + 1e-20
                results["spectral_balance_dev_db"] = _calc_spectral_balance_dev(mag, xf)
            else:
                results["spectral_balance_dev_db"] = None
        except Exception:
            results["spectral_balance_dev_db"] = None

        results["reverb_tail_ratio_db"] = _reverb_tail_ratio_db(Mb, sr_b)
        results["stereo_width_iqr_db"] = _stereo_width_iqr_db(Lb, Rb, sr_b)

        results["hf_rms_var_db"] = None
        results["hf_var_norm_pct"] = None
        if sr >= 96000 and (sr / 2) > 30000:
            try:
                low_j_hz = 30000
                high_j_hz = min(90000.0, sr / 2.0 * 0.98)
                low_j_norm = low_j_hz / (sr / 2.0)
                high_j_norm = high_j_hz / (sr / 2.0)
                if high_j_norm > low_j_norm:
                    sos = butter(2, [low_j_norm, high_j_norm], btype='band', output='sos')
                    try:
                        from scipy.signal import sosfiltfilt as _sff
                        band_filtered = _sff(sos, mono)
                    except Exception:
                        band_filtered = sosfilt(sos, mono)
                    win_samples = max(1, int(0.005 * sr))
                    if len(band_filtered) > win_samples:
                        rms_vals_db = []
                        for i in range(0, len(band_filtered) - win_samples + 1, win_samples):
                            segment = band_filtered[i: i + win_samples]
                            rms = calculate_rms(segment)
                            if rms > 1e-12:
                                rms_db = dbfs(rms)
                                if np.isfinite(rms_db):
                                    rms_vals_db.append(rms_db)
                        if len(rms_vals_db) > 5:
                            arr = np.array(rms_vals_db, dtype=float)
                            arr = arr[np.isfinite(arr)]
                            if arr.size > 5:
                                lo = int(max(0, math.floor(0.10 * arr.size)))
                                hi = int(min(arr.size, math.ceil(0.90 * arr.size)))
                                arr_sorted = np.sort(arr)
                                core = arr_sorted[lo:hi] if hi > lo else arr_sorted
                                var_db = float(np.std(core)) if core.size > 1 else float(np.std(arr_sorted))
                                med_db = float(np.median(arr_sorted))
                                results["hf_rms_var_db"] = var_db
                                results["hf_var_norm_pct"] = float((var_db / max(1e-6, abs(med_db))) * 100.0)
            except Exception:
                pass

        results["thdn_db"] = None
        results["jitter_ppm"] = None
        try:
            ACCEPTED_TONES = (400, 997, 1000, 1002, 10000)
            TOL_HZ = 3.0
            detection_duration_sec = min(5.0, len(mono) / sr)
            win_det = int(detection_duration_sec * sr)
            if win_det >= int(0.1 * sr):
                seg_d = mono[:win_det]
                yf_d = np.fft.rfft(seg_d * np.hanning(win_det))
                xf_d = np.fft.rfftfreq(win_det, 1 / sr)
                mag_d = np.abs(yf_d) + 1e-20
                if len(mag_d) > 1:
                    idx_max = np.argmax(mag_d[1:]) + 1
                    dominant_freq = xf_d[idx_max]
                    dominant_mag = mag_d[idx_max]
                    is_test_tone = any(abs(dominant_freq - ref_tone) <= TOL_HZ for ref_tone in ACCEPTED_TONES)
                    median_mag_d = np.median(mag_d[1:]) if len(mag_d) > 1 else 0.0
                    is_dominant_peak = (median_mag_d > 1e-12 and dominant_mag > median_mag_d * 10)
                    if is_test_tone and is_dominant_peak:
                        thdn_duration_sec = min(30.0, len(mono) / sr)
                        win_thd = int(thdn_duration_sec * sr)
                        seg_thd = mono[:win_thd]
                        yf_thd = np.fft.rfft(seg_thd * np.hanning(win_thd))
                        mag_thd = np.abs(yf_thd) + 1e-20
                        xf_thd = np.fft.rfftfreq(win_thd, 1 / sr)
                        idx_fund = np.argmin(np.abs(xf_thd - dominant_freq))
                        fundamental_mag = mag_thd[idx_fund]
                        mask_noise_harm = np.ones(len(mag_thd), dtype=bool)
                        mask_noise_harm[0] = False
                        exclude_bw_hz = max(5.0, dominant_freq * 0.01)
                        fund_low = dominant_freq - exclude_bw_hz
                        fund_high = dominant_freq + exclude_bw_hz
                        mask_noise_harm[(xf_thd >= fund_low) & (xf_thd <= fund_high)] = False
                        noise_harm_power = float(np.sum(np.square(mag_thd[mask_noise_harm])))
                        fund_power = float(fundamental_mag ** 2)
                        if noise_harm_power > 0 and fund_power > 1e-24:
                            thdn_ratio = math.sqrt(noise_harm_power / fund_power)
                            results["thdn_db"] = dbfs(thdn_ratio) if thdn_ratio > 1e-9 else -np.inf
                        try:
                            analytic = hilbert(mono[:int(min(1.0, len(mono) / sr) * sr)])
                            inst_phase = np.unwrap(np.angle(analytic))
                            phase_diff = np.diff(inst_phase)
                            phase_diff_corr = (phase_diff + np.pi) % (2 * np.pi) - np.pi
                            inst_freq = (sr / (2 * np.pi)) * phase_diff_corr
                            inst_freq = np.insert(inst_freq, 0, np.mean(inst_freq))
                            mean_f = np.mean(inst_freq)
                            if abs(mean_f - dominant_freq) < TOL_HZ * 2 and np.all(np.isfinite(inst_freq)):
                                freq_std = np.std(inst_freq)
                                if abs(mean_f) > 1:
                                    jitter_ppm = (freq_std / abs(mean_f)) * 1e6
                                    results["jitter_ppm"] = float(jitter_ppm) if np.isfinite(jitter_ppm) else None
                        except Exception:
                            pass
        except Exception:
            pass

        wins_auto, meta_auto = select_noise_windows(Mb, sr_b, mode="auto", win_sec=0.40, hop_sec=0.10, max_total_sec=8.0)
        nm_auto = compute_noise_metrics_robust(Mb, sr_b, wins_auto, noise_samples=None, windows_meta=meta_auto)
        use_nm = nm_auto
        use_meta = meta_auto
        use_windows = wins_auto

        spur_lbl_auto = str(nm_auto.get("spur_label") or "")
        spur_hc_auto = int(nm_auto.get("spur_harmonics_count") or 0)
        hum_est_hc = int(nm_auto.get("hum_estimated_hcount") or 0)
        hum_dense_auto = (spur_lbl_auto.startswith("HUM") and spur_hc_auto >= 10) or (hum_est_hc >= 10)

        force_sn = False
        nf_red_auto = float(meta_auto.get("nf_notch_reduction_db") or 0.0)
        if hum_dense_auto or (np.isfinite(nf_red_auto) and nf_red_auto >= 1.0):
            force_sn = True

        if force_sn and use_meta.get("noise_selection_mode") != "strict+notch":
            f0_hint = nm_auto.get("spur_fundamental_hz")
            if not isinstance(f0_hint, (int, float)) or not np.isfinite(float(f0_hint)) or float(f0_hint) <= 0:
                if "50" in spur_lbl_auto:
                    f0_hint = 50.0
                elif "60" in spur_lbl_auto:
                    f0_hint = 60.0
                else:
                    f0_hint = 50.0
            wins_sn, meta_sn = select_noise_windows(Mb, sr_b, mode="strict+notch", win_sec=0.40, hop_sec=0.10, max_total_sec=8.0, hum_hint={"f0": float(f0_hint)})
            nm_sn = compute_noise_metrics_robust(Mb, sr_b, wins_sn, noise_samples=None, windows_meta=meta_sn)
            use_nm = nm_sn
            use_meta = meta_sn
            use_windows = wins_sn
        else:
            nf_auto = float(nm_auto.get("nf_dbfs", -np.inf))
            diff_auto = float(nm_auto.get("nf_consistency_diff_db", 0.0))
            if (np.isfinite(nf_auto) and nf_auto > -40.0) and use_meta.get("noise_selection_mode") != "strict":
                wins_s, meta_s = select_noise_windows(Mb, sr_b, mode="strict", win_sec=0.40, hop_sec=0.10, max_total_sec=8.0)
                nm_s = compute_noise_metrics_robust(Mb, sr_b, wins_s, noise_samples=None, windows_meta=meta_s)
                nf_s = float(nm_s.get("nf_dbfs", -np.inf))
                diff_s = float(nm_s.get("nf_consistency_diff_db", 0.0))
                better = False
                if np.isfinite(nf_s) and np.isfinite(nf_auto) and (nf_s + 0.5) < nf_auto:
                    better = True
                if not better and np.isfinite(diff_s) and np.isfinite(diff_auto) and (diff_s + 0.3) < diff_auto:
                    better = True
                if better:
                    use_nm = nm_s
                    use_meta = meta_s
                    use_windows = wins_s

        results["noise_windows_preview"] = [(int(s), int(e)) for (s, e) in (use_windows[:3] if isinstance(use_windows, list) else [])]

        results["noise_selection_mode"] = use_meta.get("noise_selection_mode")
        results["noise_windows_count"] = int(use_nm.get("noise_windows_count") or 0)
        results["noise_total_duration_sec"] = float(use_nm.get("noise_total_duration_sec") or 0.0)
        results["noise_total_duration_uncapped_sec"] = float(use_nm.get("noise_total_duration_uncapped_sec") or 0.0)
        results["nf_notch_reduction_db"] = float(use_nm.get("nf_notch_reduction_db") or 0.0)
        results["notch_f0"] = use_nm.get("notch_f0")
        results["notch_harmonics"] = int(use_nm.get("notch_harmonics") or 0)
        results["music_limited"] = bool(use_meta.get("music_limited") or False)
        results["noise_rms_p90_dbfs"] = use_nm.get("noise_rms_p90_dbfs")

        nf_raw = float(use_nm.get("nf_dbfs", -np.inf))
        nf_bb = float(use_nm.get("nf_broadband_dbfs", nf_raw))
        nf_cross_rms = use_nm.get("nf_cross_rms_dbfs")
        results["noise_consistency_diff_db"] = float(use_nm.get("nf_consistency_diff_db", 0.0))

        nf_cross_p90 = None
        try:
            win = int(round(0.4 * sr_b))
            hop = max(1, int(round(0.1 * sr_b)))
            vals = []
            n_b = Mb.size
            for i in range(0, n_b - win + 1, hop):
                seg = Mb[i:i + win]
                rms = calculate_rms(seg)
                if rms > 1e-12:
                    v = dbfs(rms)
                    if np.isfinite(v):
                        vals.append(v)
            if len(vals) >= 10:
                arr = np.array(vals, dtype=float)
                nf_cross_p90 = float(np.percentile(arr, 90))
        except Exception:
            nf_cross_p90 = None

        nwin = results["noise_windows_count"]
        ndur = results["noise_total_duration_sec"]
        diff_final = results.get("noise_consistency_diff_db")
        extreme_uncertain = (isinstance(diff_final, (int, float)) and np.isfinite(diff_final) and diff_final > 30.0) or (nwin < 2 and ndur < 3.0)
        moderate_uncertain = (isinstance(diff_final, (int, float)) and np.isfinite(diff_final) and diff_final > 20.0) or (nwin < 2 or ndur < 3.0)

        nf_sane_cap = None
        nf_sanity_applied = False
        nf_cap_reason = None
        nf_for_cap = nf_bb
        if nf_cross_p90 is not None and np.isfinite(nf_cross_p90):
            if results.get("music_limited") or extreme_uncertain:
                base_cap = nf_cross_p90 - 10.0
                floor_cap = -60.0
                nf_sane_cap = max(base_cap, floor_cap)
                nf_cap_reason = f"p90-10 (floor {floor_cap:.0f})"
            elif moderate_uncertain:
                base_cap = nf_cross_p90 - 8.0
                floor_cap = -55.0
                nf_sane_cap = max(base_cap, floor_cap)
                nf_cap_reason = f"p90-8 (floor {floor_cap:.0f})"
            else:
                base_cap = nf_cross_p90 - 6.0
                floor_cap = -50.0
                nf_sane_cap = max(base_cap, floor_cap)
                nf_cap_reason = f"p90-6 (floor {floor_cap:.0f})"
            if nf_for_cap < nf_sane_cap:
                nf_for_cap = nf_sane_cap
                nf_sanity_applied = True

        results["noise_floor_raw_dbfs"] = float(nf_raw)
        results["nf_broadband_dbfs"] = float(nf_bb)
        results["noise_floor_dbfs"] = float(nf_for_cap)
        results["noise_floor_cross_rms_dbfs"] = nf_cross_rms
        results["noise_floor_cross_p90_dbfs"] = nf_cross_p90
        results["noise_floor_sanity_cap_dbfs"] = nf_sane_cap
        results["noise_floor_sanity_applied"] = bool(nf_sanity_applied)
        results["noise_floor_cap_reason"] = nf_cap_reason

        results["noise_spur_db"] = float(max(0.0, use_nm.get("spur_db", 0.0)))
        results["noise_spur_label"] = use_nm.get("spur_label")
        results["noise_spur_fundamental_hz"] = use_nm.get("spur_fundamental_hz")
        results["noise_spur_harmonics_count"] = int(min(20, use_nm.get("spur_harmonics_count") or 0))
        if results["noise_spur_label"] and results["noise_spur_label"].startswith("HUM"):
            if results["noise_spur_harmonics_count"] > 0:
                base = results["noise_spur_label"].split()[0]
                if results["noise_spur_harmonics_count"] >= 15:
                    results["noise_spur_label"] = f"{base} (denso)"
                else:
                    results["noise_spur_label"] = f"{base} + H{results['noise_spur_harmonics_count']}"

        if isinstance(diff_final, (int, float)) and np.isfinite(diff_final):
            if diff_final > 30.0:
                use_conf = "Bassa (NF escluso)"
            elif diff_final > 20.0:
                use_conf = "Media"
            else:
                use_conf = "Alta"
        else:
            use_conf = "Alta"
        if results["noise_windows_count"] < 2 or results["noise_total_duration_sec"] < 3.0:
            use_conf = "Bassa (NF escluso)" if extreme_uncertain else "Bassa"
        if results.get("music_limited") and not extreme_uncertain:
            use_conf = "Media" if results["noise_windows_count"] >= 2 and results["noise_total_duration_sec"] >= 3.0 else "Bassa"
        results["noise_confidence"] = use_conf

        results = compute_nf_interval_fields(results)

        tc, rt = compute_transient_metrics(Mb, sr_b)
        results["transient_crest_med"] = tc if (tc is None or np.isfinite(tc)) else None
        results["transient_rise_med_ms"] = rt if (rt is None or np.isfinite(rt)) else None

        results["clipping_detected"] = bool(results["true_peak_est_dbtp"] is not None and results["true_peak_est_dbtp"] > 0.0)

        score_float = calculate_score_float(results)
        results["score_float"] = score_float
        results["score"] = int(round(score_float))
        results["assessment"] = get_verbal_assessment(results["score"], results)
        return results

    except sf.LibsndfileError as e_sf:
        print(f"!!! Errore SoundFile '{os.path.basename(filepath)}': {e_sf}")
        try:
            if "format not recognised" in str(e_sf).lower() and os.path.splitext(filepath)[1].lower() in ('.dsf', '.dff'):
                print("Suggerimento: converti il DSD in PCM (WAV 24 bit o FLAC 24 bit) tramite il pre-process automatico.")
        except Exception:
            pass
        return None
    except MemoryError:
        print(f"!!! Errore Memoria '{os.path.basename(filepath)}'.")
        return None
    except Exception as exc:
        print(f"!!! Errore Inatteso '{os.path.basename(filepath)}': {type(exc).__name__} - {exc}")
        import traceback
        traceback.print_exc()
        return None


def rank_results_hygiene_first(results, policy=None):
    """
    Ranking dei risultati.
    - Default: ordina SOLO per score (AQC_TIEBREAK=none) e opzionalmente tiene i DQ in fondo (AQC_RANK_DQ_LAST=1).
    - Se tie-break disattivato, in caso di punteggio esattamente uguale usa un tie-break metrico:
      spur più basso, ISP margin più alto, DC offset più basso (deterministico, niente residuo).
    - Se AQC_TIEBREAK=residual ripristina la banda top con ri-ordinamento per residuo.
    """
    if not results:
        return [], None, "score"

    import os

    def _sf(x, d=0.0):
        try:
            f = float(x)
            return f if np.isfinite(f) else d
        except Exception:
            return d

    # Switch di comportamento
    tie_mode = (os.environ.get("AQC_TIEBREAK", "none").strip().lower())
    dq_last  = (os.environ.get("AQC_RANK_DQ_LAST", "1").strip().lower() in ("1", "true", "yes", "on"))

    def _gate_threshold(top_item):
        # Se tie-break disattivato → niente “banda top”
        if tie_mode in ("none", "off", "0"):
            return 0.0
        try:
            diff = _sf(top_item.get('noise_consistency_diff_db'), 0.0) or 0.0
            wins = int(top_item.get('noise_windows_count') or 0)
            dur  = _sf(top_item.get('noise_total_duration_sec'), 0.0) or 0.0
            conf = (top_item.get('noise_confidence') or '').strip().lower()
            be   = (top_item.get('loudness_backend') or '').strip().lower()
            extreme  = (diff > 30.0) or (wins < 2 and dur < 3.0)
            moderate = (diff > 20.0) or (wins < 2 or 3.0 > dur)
            thr = 1.0
            if extreme:
                thr = 0.4
            elif moderate:
                thr = 0.7
            if conf.startswith('bas'):
                thr = min(thr, 0.7)
            if be == 'rms-fallback':
                thr = min(thr, 0.6)
            return thr
        except Exception:
            return 0.7

    from scipy.signal import butter, sosfilt

    def _design_filters(sr_local):
        nyq = sr_local * 0.5
        lo_hi = min(200.0 / nyq, 0.999)
        mi_lo = min(200.0 / nyq, 0.999)
        mi_hi = min(5000.0 / nyq, 0.999)
        hi_lo = min(5000.0 / nyq, 0.999)
        try:
            sos_lo = butter(4, lo_hi, btype='lowpass', output='sos')
            sos_mid = butter(4, [mi_lo, mi_hi], btype='band', output='sos')
            sos_hi = butter(4, hi_lo, btype='highpass', output='sos')
        except Exception:
            sos_lo = sos_mid = sos_hi = None
        return sos_lo, sos_mid, sos_hi

    def _segment_residual(segA, segB, sr_local):
        m = int(round(0.0005 * sr_local))
        best_res = float('inf')
        best_shift = 0
        best_sign = 1.0
        best_g_db = 0.0
        if m > 0:
            shifts = range(-m, m + 1, max(1, m // 8))
        else:
            shifts = [0]
        for sh in shifts:
            if sh > 0:
                b = segB[sh:]
                a = segA[:b.size]
            elif sh < 0:
                b = segB[:sh]
                a = segA[-sh:]
            else:
                a = segA
                b = segB
            if a.size < 64 or b.size < 64:
                continue
            for sgn in (1.0, -1.0):
                bb = b * sgn
                den = float(np.dot(bb, bb)) + 1e-20
                g = float(np.dot(bb, a)) / den
                g = float(np.clip(g, 10.0 ** (-6 / 20.0), 10.0 ** (6 / 20.0)))
                rA = calculate_rms(a)
                if rA <= 1e-20:
                    continue
                resid = calculate_rms(a - g * bb)
                if np.isfinite(resid) and resid < best_res:
                    best_res = resid
                    best_shift = sh
                    best_sign = sgn
                    best_g_db = 20.0 * math.log10(max(g, 1e-20))
        if not np.isfinite(best_res) or best_res <= 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        denA = calculate_rms(segA)
        resid_db = dbfs(best_res / max(denA, 1e-20))
        return resid_db, best_shift, best_sign, denA, best_g_db

    def _band_residuals(segA, segB, sr_local, best_shift, best_sign, sos_lo, sos_mid, sos_hi):
        if best_shift > 0:
            b = segB[best_shift:]
            a = segA[:b.size]
        elif best_shift < 0:
            b = segB[:best_shift]
            a = segA[-best_shift:]
        else:
            a = segA
            b = segB
        b = b * best_sign
        if a.size < 64 or b.size < 64:
            return 0.0, 0.0, 0.0

        def _resid_db(a_f, b_f):
            den = float(np.dot(b_f, b_f)) + 1e-20
            g = float(np.dot(b_f, a_f)) / den
            g = float(np.clip(g, 10.0 ** (-6 / 20.0), 10.0 ** (6 / 20.0)))
            rA = calculate_rms(a_f)
            if rA <= 1e-20:
                return 0.0
            resid = calculate_rms(a_f - g * b_f)
            return dbfs(resid / max(rA, 1e-20))

        try:
            loA = sosfilt(sos_lo, a) if sos_lo is not None else a
            loB = sosfilt(sos_lo, b) if sos_lo is not None else b
            miA = sosfilt(sos_mid, a) if sos_mid is not None else a
            miB = sosfilt(sos_mid, b) if sos_mid is not None else b
            hiA = sosfilt(sos_hi, a) if sos_hi is not None else a
            hiB = sosfilt(sos_hi, b) if sos_hi is not None else b
            rL = _resid_db(loA, loB)
            rM = _resid_db(miA, miB)
            rH = _resid_db(hiA, hiB)
        except Exception:
            rL = rM = rH = 0.0
        return rL, rM, rH

    def _best_gain_and_residual_robust(pathA, pathB):
        try:
            with sf.SoundFile(pathA) as fA:
                if fA.channels != 2:
                    return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                srA = fA.samplerate
                XA = fA.read(dtype="float64")
            with sf.SoundFile(pathB) as fB:
                if fB.channels != 2:
                    return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                srB = fB.samplerate
                XB = fB.read(dtype="float64")
        except Exception:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        if XA is None or XB is None:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        xA = (XA[:, 0] + XA[:, 1]) * 0.5 if XA.ndim == 2 else XA
        xB = (XB[:, 0] + XB[:, 1]) * 0.5 if XB.ndim == 2 else XB
        if srA != srB:
            try:
                from scipy.signal import resample_poly
                if srA > srB:
                    g = math.gcd(srA, srB)
                    xA = resample_poly(xA, srB // g, srA // g)
                    sr = srB
                else:
                    g = math.gcd(srA, srB)
                    xB = resample_poly(xB, srA // g, srB // g)
                    sr = srA
            except Exception:
                sr = min(srA, srB)
                nA = int(len(xA) * sr / srA)
                nB = int(len(xB) * sr / srB)
                xA = np.interp(np.linspace(0, len(xA) - 1, nA, endpoint=False), np.arange(len(xA)), xA)
                xB = np.interp(np.linspace(0, len(xB) - 1, nB, endpoint=False), np.arange(len(xB)), xB)
        else:
            sr = srA
        try:
            nyq = sr * 0.5
            cutoff = min(20000.0, 0.98 * nyq)
            if cutoff > 1000.0 and nyq > cutoff:
                Wn = cutoff / nyq
                sos_lp = butter(6, Wn, btype='lowpass', output='sos')
                xA = sosfilt(sos_lp, xA)
                xB = sosfilt(sos_lp, xB)
        except Exception:
            pass
        ds_factor = max(1, int(round(sr / 8000.0)))
        try:
            a_ds = xA if ds_factor == 1 else xA[::ds_factor]
            b_ds = xB if ds_factor == 1 else xB[::ds_factor]
            n = int(2 ** math.ceil(math.log2(len(a_ds) + len(b_ds) - 1)))
            Fa = np.fft.rfft(a_ds, n=n)
            Fb = np.fft.rfft(b_ds, n=n)
            R = Fa * np.conj(Fb)
            denom = np.abs(R) + 1e-20
            R /= denom
            corr = np.fft.irfft(R, n=n)
            lags = np.arange(-len(b_ds) + 1, len(a_ds))
            if corr.size != lags.size:
                corr = np.concatenate([corr[-(len(b_ds) - 1):], corr[:len(a_ds)]])
            idx = int(np.argmax(corr))
            off = int(lags[idx] * ds_factor)
        except Exception:
            off = 0
        if off >= 0:
            stA = off
            stB = 0
        else:
            stA = 0
            stB = -off
        n_overlap = min(len(xA) - stA, len(xB) - stB)
        if n_overlap <= int(0.5 * sr):
            m = min(len(xA), len(xB), int(0.5 * sr))
            if m <= 0:
                return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            stA = 0
            stB = 0
            n_overlap = m
        a_al = xA[stA:stA + n_overlap]
        b_al = xB[stB:stB + n_overlap]
        dur = n_overlap / sr
        sos_lo, sos_mid, sos_hi = _design_filters(sr)
        scales = [3.0, 7.0, 15.0]
        scales = [s for s in scales if dur >= s * 1.2]
        if not scales:
            scales = [max(1.5, min(5.0, dur * 0.5))]
        med_resid = []
        med_lo = []
        med_mid = []
        med_hi = []
        gains_all = []
        for W_s in scales:
            W = int(round(W_s * sr))
            if W <= 64 or W > len(a_al):
                W = max(64, min(len(a_al), int(round(0.5 * len(a_al)))))
            K = max(3, int((len(a_al) - W) / W)) if len(a_al) > W else 1
            if K <= 1:
                starts = [max(0, (len(a_al) - W) // 2)]
            else:
                positions = np.linspace(0.15, 0.85, K)
                starts = [int(round(p * (len(a_al) - W))) for p in positions]
            res_vals = []
            rL_vals = []
            rM_vals = []
            rH_vals = []
            g_vals = []
            for st in starts:
                segA = a_al[st:st + W]
                segB = b_al[st:st + W]
                if segA.size != segB.size or segA.size < 128:
                    continue
                resid_db, best_shift, best_sign, _, best_g_db = _segment_residual(segA, segB, sr)
                rL, rM, rH = _band_residuals(segA, segB, sr, best_shift, best_sign, sos_lo, sos_mid, sos_hi)
                res_vals.append(resid_db)
                rL_vals.append(rL)
                rM_vals.append(rM)
                rH_vals.append(rH)
                g_vals.append(best_g_db)
            if not res_vals:
                continue
            med_resid.append(float(np.median(np.array(res_vals, dtype=float))))
            med_lo.append(float(np.median(np.array(rL_vals, dtype=float))))
            med_mid.append(float(np.median(np.array(rM_vals, dtype=float))))
            med_hi.append(float(np.median(np.array(rH_vals, dtype=float))))
            if g_vals:
                gains_all.extend(list(np.array(g_vals, dtype=float)))
        if not med_resid:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        resid_db = float(np.median(np.array(med_resid, dtype=float)))
        rL_db = float(np.median(np.array(med_lo, dtype=float)))
        rM_db = float(np.median(np.array(med_mid, dtype=float)))
        rH_db = float(np.median(np.array(med_hi, dtype=float)))
        gdb = float(np.median(np.array(gains_all, dtype=float))) if gains_all else 0.0
        drift_ppm = 0.0
        sim = 0.0
        if np.isfinite(resid_db):
            sim = (min(max((-resid_db) - 20.0, 0.0), 40.0)) / 40.0
        return gdb, resid_db, drift_ppm, 1.0, rL_db, rM_db, rH_db, sim

    # Tie-break metrico (senza residuo): spur↓, ISP↑, DC↓
    def _tie_key(r):
        spur = r.get('noise_spur_db')
        isp = r.get('isp_margin_db')
        dc  = r.get('dc_offset_dbfs')
        spur_key = float(spur) if (isinstance(spur, (int, float)) and np.isfinite(spur)) else float('inf')
        isp_key  = -float(isp) if (isinstance(isp, (int, float)) and np.isfinite(isp)) else 0.0  # ISP maggiore è meglio
        dc_key   = float(dc) if (isinstance(dc, (int, float)) and np.isfinite(dc)) else 0.0      # più negativo è meglio
        sid = str(r.get('filepath') or r.get('filename') or '')
        return (round(spur_key, 2), round(isp_key, 3), round(dc_key, 1), sid)

    def _tp_refine(result_obj):
        # Manteniamo disponibile: usato solo se tie-break residual è attivo
        try:
            with sf.SoundFile(result_obj['filepath']) as f:
                if f.channels != 2:
                    return
                sr = f.samplerate
                audio = f.read(dtype="float64")
            if audio is None or audio.shape[0] < int(0.1 * sr):
                return
            tp32 = _true_peak_internal(audio, sr, os_factor=32)
            if tp32 is None or not np.isfinite(tp32):
                return
            result_obj["true_peak_est_dbtp"] = tp32
            result_obj["peak_dbfs_overall"] = tp32
            result_obj["isp_margin_db"] = _isp_margin_db(audio, sr, os_factor=32)
            isp_stats = compute_isp_stats(audio, sr, os_factor=32, win_s=0.05, hop_s=0.025)
            if isinstance(isp_stats, dict):
                result_obj.update({
                    "isp_window_count": isp_stats.get("isp_window_count"),
                    "isp_margin_p05_db": isp_stats.get("isp_margin_p05_db"),
                    "isp_margin_p50_db": isp_stats.get("isp_margin_p50_db"),
                    "isp_margin_p95_db": isp_stats.get("isp_margin_p95_db"),
                    "isp_under_1db_count": isp_stats.get("isp_under_1db_count"),
                    "isp_under_05db_count": isp_stats.get("isp_under_05db_count"),
                    "isp_under_02db_count": isp_stats.get("isp_under_02db_count"),
                    "isp_under_1db_frac": isp_stats.get("isp_under_1db_frac"),
                    "isp_under_05db_frac": isp_stats.get("isp_under_05db_frac"),
                    "isp_under_02db_frac": isp_stats.get("isp_under_02db_frac"),
                })
            calculate_score_float(result_obj)
        except Exception:
            return

    def _residual_round_robin(cands, top_n=5):
        c = cands[:min(top_n, len(cands))]
        M = len(c)
        med_resid = [0.0] * M
        for i in range(M):
            vals = []
            for j in range(M):
                if i == j:
                    continue
                gdb, resid_db, drift_ppm, conf, rL_db, rM_db, rH_db, sim = _best_gain_and_residual_robust(c[i].get('filepath'), c[j].get('filepath'))
                if isinstance(resid_db, (int, float)) and np.isfinite(resid_db):
                    vals.append(resid_db)
            med_resid[i] = float(np.median(vals)) if vals else 0.0
        return list(zip(c, med_resid))

    def _order_and_refine(lst):
        if not lst:
            return []
        # Se tie-break disattivato: ordina per score discendente
        # e usa il tie-break metrico (spur↓, ISP↑, DC↓) solo per punteggi identici (stable sort).
        if tie_mode in ("none", "off", "0"):
            # 1) Ordine metrico ASC (il "più pulito" passa davanti in caso di pari score)
            temp = sorted(lst, key=_tie_key)
            # 2) Ordine per score DESC (stabile: mantiene l'ordine metrico dove lo score è identico)
            ordered = sorted(temp, key=lambda r: _sf(r.get('score_float'), _sf(r.get('score'))), reverse=True)
            return ordered

        # Vecchio comportamento: banda top + refine TP + residual tie-break
        ordered = sorted(lst, key=lambda r: (_sf(r.get('score_float'), _sf(r.get('score'))),
                                             str(r.get('filename') or "")), reverse=True)
        top_score = _sf(ordered[0].get('score_float'), _sf(ordered[0].get('score')))
        thr = _gate_threshold(ordered[0])
        band = []
        for r in ordered:
            sc = _sf(r.get('score_float'), _sf(r.get('score')))
            if top_score - sc <= thr:
                band.append(r)
            else:
                break
        if band:
            for itm in band:
                _tp_refine(itm)
            ordered = sorted(lst, key=lambda r: (_sf(r.get('score_float'), _sf(r.get('score'))),
                                                 str(r.get('filename') or "")), reverse=True)
            top_score = _sf(ordered[0].get('score_float'), _sf(ordered[0].get('score')))
            thr = _gate_threshold(ordered[0])
            band = []
            for r in ordered:
                sc = _sf(r.get('score_float'), _sf(r.get('score')))
                if top_score - sc <= thr:
                    band.append(r)
                else:
                    break
        if len(band) >= 2:
            for i in range(len(band) - 1):
                a = band[i]
                b = band[i + 1]
                gdb, resid_db, drift_ppm, conf, rL_db, rM_db, rH_db, sim = _best_gain_and_residual_robust(a.get('filepath'), b.get('filepath'))
                a['pair_gain_to_next_db'] = gdb
                a['pair_residual_to_next_db'] = resid_db
                a['pair_drift_to_next_ppm'] = 0.0
                a['pair_compare_conf'] = conf
                a['pair_residual_low_db'] = rL_db
                a['pair_residual_mid_db'] = rM_db
                a['pair_residual_high_db'] = rH_db
                a['pair_similarity'] = sim
                a['pair_same_master_next'] = bool(resid_db <= -35.0) if np.isfinite(resid_db) else False
                b['pair_gain_to_prev_db'] = -gdb if np.isfinite(gdb) else 0.0
                b['pair_residual_to_prev_db'] = resid_db if np.isfinite(resid_db) else 0.0
                b['pair_drift_to_prev_ppm'] = 0.0
                b['pair_compare_conf'] = conf
                b['pair_residual_low_db'] = rL_db
                b['pair_residual_mid_db'] = rM_db
                b['pair_residual_high_db'] = rH_db
                b['pair_similarity'] = sim
                b['pair_same_master_prev'] = bool(resid_db <= -35.0) if np.isfinite(resid_db) else False
            rr = _residual_round_robin(band, top_n=min(5, len(band)))
            if rr and len(rr) >= 2:
                med_map = {id(x): m for x, m in rr}
                # fra i pari punteggio nella banda, usa residuo e poi metrica
                band_sorted = sorted(band, key=lambda r: (med_map.get(id(r), 0.0), _tie_key(r)))
                tail = [x for x in ordered if x not in band]
                ordered = band_sorted + tail
        return ordered

    # Separazione DQ opzionale
    dq_list = [r for r in results if (str(r.get('clipping_class') or '').strip().lower() == 'hard')]
    clean_list = [r for r in results if (str(r.get('clipping_class') or '').strip().lower() != 'hard')]

    if not dq_last:
        ordered_all = _order_and_refine(list(results))
        winner = ordered_all[0] if ordered_all else None
        return ordered_all, winner, "score"

    ordered_clean = _order_and_refine(clean_list)
    ordered_dq = _order_and_refine(dq_list) if dq_list else []

    combined = ordered_clean + ordered_dq
    winner = ordered_clean[0] if ordered_clean else (ordered_dq[0] if ordered_dq else None)
    return combined, winner, "score"

# --- Funzioni di Input/Conversione/Utility (Invariate da versione precedente) ---
def get_audio_files_from_directory():
    import os, sys
    SUPPORTED = ('.wav', '.flac', '.aiff', '.aif', '.aifc', '.dsf', '.dff')
    env_dir = os.environ.get("AQC_INPUT_DIR", "").strip().strip('"').strip("'")
    auto = bool(env_dir) and os.path.isdir(env_dir)
    if auto:
        dir_path = env_dir
    else:
        while True:
            dir_path = input("Percorso della cartella da analizzare: ").strip().strip('"').strip("'")
            if not dir_path:
                print("Percorso vuoto. Uscita.")
                sys.exit(0)
            if not os.path.isdir(dir_path):
                print("Non è una directory valida. Riprova.\n")
                continue
            break
    rec_env = os.environ.get("AQC_SCAN_RECURSIVE", "").strip().lower()
    recursive = rec_env in ("1", "true", "on", "yes", "y", "")  # default: ricorsivo
    files = []
    if recursive:
        for root, _, fnames in os.walk(dir_path):
            for f in fnames:
                p = os.path.join(root, f)
                if os.path.isfile(p) and f.lower().endswith(SUPPORTED):
                    files.append(p)
    else:
        for f in os.listdir(dir_path):
            p = os.path.join(dir_path, f)
            if os.path.isfile(p) and f.lower().endswith(SUPPORTED):
                files.append(p)
    files.sort(key=lambda x: (os.path.dirname(x), os.path.basename(x)))
    if not files:
        print("La cartella non contiene file audio supportati. Fine.")
        sys.exit(0)
    lim_env = os.environ.get("AQC_FILE_LIMIT", "").strip()
    if lim_env.isdigit():
        k = int(lim_env)
        if k > 0:
            files = files[:k]
    print(f"Trovati {len(files)} file audio supportati{(' in '+dir_path) if auto else ''}.")
    return files

def check_ffmpeg():
    ffmpeg_ok = False
    ffprobe_ok = False
    try:
        subprocess.run(['ffmpeg', '-version'], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        ffmpeg_ok = True
        print("✓ FFmpeg trovato.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        ffmpeg_ok = False
    if not ffmpeg_ok:
        print("✗ FFmpeg è obbligatorio. Installa FFmpeg e riprova.")
        sys.exit(1)

    try:
        subprocess.run(['ffprobe', '-version'], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        ffprobe_ok = True
        print("✓ FFprobe trovato.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        ffprobe_ok = False
        print("✗ FFprobe non trovato: alcune introspezioni (SR/codec) potrebbero essere limitate.")

    print("ℹ️ Catena conversione DSD→PCM: aresample=soxr (precisione 28). Il filtro dsd2pcm non è richiesto.")
    return True

def get_audio_md5(filepath, ffmpeg_available, dsd_target_sr=TEMP_FLAC_SR, dsd_target_bits=TEMP_FLAC_BITS, dsd_target_quality=TEMP_FLAC_QUALITY):
    import os, tempfile, subprocess, shutil
    base_name = os.path.basename(filepath)
    is_dsd = filepath.lower().endswith(('.dsf', '.dff'))
    temp_file_created = None
    if not ffmpeg_available:
        print(f"!! MD5/Analisi skip: FFmpeg non disponibile per '{base_name}'.")
        return None, filepath, None

    file_to_hash_internally = filepath
    try:
        threads = os.environ.get("FFMPEG_THREADS") or os.environ.get("OMP_NUM_THREADS") or "1"
    except Exception:
        threads = "1"

    def _filesize(p):
        try:
            return os.path.getsize(p)
        except Exception:
            return None

    nonint = os.environ.get("AQC_NONINTERACTIVE", "").strip().lower() in ("1", "true", "on", "yes", "y")

    if is_dsd:
        print(f"\nMD5 Pre-Process: '{base_name}' è un file DSD.")
        if nonint:
            chosen_format_for_dsd = (os.environ.get("AQC_DSD_CONTAINER", "flac").strip().lower() or "flac")
            quality_mode = (os.environ.get("AQC_DSD_QUALITY", "hires").strip().lower() or "hires")
            if chosen_format_for_dsd not in ("flac", "wav"):
                chosen_format_for_dsd = "flac"
            if quality_mode not in ("hires", "cd"):
                quality_mode = "hires"
            print(f"  Modalità non-interattiva: container={chosen_format_for_dsd.upper()}, qualità={quality_mode}")
        else:
            chosen_format_for_dsd = 'flac'
            while True:
                choice_fmt = input("Contenitore per l'analisi? FLAC o WAV [F/W] (Default F): ").strip().lower()
                if choice_fmt in ('', 'f', 'flac'):
                    chosen_format_for_dsd = 'flac'
                    break
                elif choice_fmt in ('w', 'wav'):
                    chosen_format_for_dsd = 'wav'
                    break
                else:
                    print("Scelta non valida. Digita F (o Invio) per FLAC, W per WAV.")
            quality_mode = 'hires'
            while True:
                choice_q = input("Qualità di conversione? Hi-Res 24 bit [H] oppure CD 44.1/16 [C] (Default H): ").strip().lower()
                if choice_q in ('', 'h', 'hi', 'hires'):
                    quality_mode = 'hires'
                    break
                elif choice_q in ('c', 'cd'):
                    quality_mode = 'cd'
                    break
                else:
                    print("Scelta non valida. Digita H (o Invio) per Hi-Res 24 bit, C per CD 44.1/16.")

        detected_sr = None
        detected_codec = None
        try:
            probe = subprocess.run(
                ['ffprobe', '-v', 'error', '-select_streams', 'a:0',
                 '-show_entries', 'stream=codec_name,sample_rate',
                 '-of', 'default=noprint_wrappers=1:nokey=1', filepath],
                check=True, capture_output=True, text=True, encoding='utf-8', errors='replace'
            )
            lines = [x.strip() for x in (probe.stdout or "").splitlines() if x.strip()]
            for tok in lines:
                if tok.isdigit():
                    detected_sr = int(float(tok))
                else:
                    detected_codec = tok
        except Exception:
            detected_sr = None
            detected_codec = None

        if quality_mode == 'cd':
            target_bits = 16
            target_sr_base = 44100
        else:
            target_bits = 24
            if detected_sr and detected_sr > 0:
                if isinstance(detected_codec, str) and detected_codec.lower().startswith('dsd'):
                    dsd_sr_true = int(detected_sr * 8)
                else:
                    dsd_sr_true = int(detected_sr)
                target_sr_base = int(round(dsd_sr_true / 16.0))
            else:
                target_sr_base = 176400

        print(f"  Rilevato SR={detected_sr or 'N/A'} Hz (codec={detected_codec or 'N/A'}) → Profilo {'Hi-Res 24 bit' if target_bits==24 else 'CD 44.1/16'} SR base: {target_sr_base} Hz")

        soxr_opts = "aresample=resampler=soxr:precision=28"
        if target_bits == 16:
            soxr_opts += ":dither_method=triangular"

        if chosen_format_for_dsd == 'flac':
            flac_sr_candidates = [352800, 176400, 88200, 44100, 384000, 192000, 96000, 48000]
            flac_sr_candidates = [sr for sr in flac_sr_candidates if sr <= target_sr_base and sr <= 655350]
            if not flac_sr_candidates:
                flac_sr_candidates = [44100]

            success = False
            temp_pcm_path = None
            for target_sr in flac_sr_candidates:
                fd, temp_pcm_path = tempfile.mkstemp(suffix='.flac')
                os.close(fd)
                cmd = [
                    'ffmpeg', '-y', '-hide_banner', '-nostats', '-nostdin',
                    '-threads', str(threads),
                    '-i', filepath,
                    '-map', '0:a:0', '-vn', '-sn', '-dn',
                    '-af', soxr_opts, '-ar', str(target_sr),
                ]
                if target_bits == 16:
                    cmd += ['-sample_fmt', 's16', '-c:a', 'flac', '-compression_level', str(max(5, int(dsd_target_quality))), '-map_metadata', '0', temp_pcm_path]
                else:
                    cmd += ['-sample_fmt', 's32', '-bits_per_raw_sample', '24', '-c:a', 'flac', '-compression_level', str(max(5, int(dsd_target_quality))), '-map_metadata', '0', temp_pcm_path]

                proc = subprocess.run(cmd, check=False, capture_output=True, text=True, encoding='utf-8', errors='replace')
                sz = _filesize(temp_pcm_path)
                if proc.returncode != 0 or not sz:
                    try:
                        if temp_pcm_path and os.path.exists(temp_pcm_path):
                            os.remove(temp_pcm_path)
                        temp_pcm_path = None
                    except OSError:
                        pass
                    continue

                vcmd = ['ffmpeg', '-v', 'error', '-hide_banner', '-nostdin', '-i', temp_pcm_path, '-map', '0:a:0', '-f', 'null', '-']
                vproc = subprocess.run(vcmd, check=False, capture_output=True, text=True, encoding='utf-8', errors='replace')
                if vproc.returncode == 0:
                    temp_file_created = temp_pcm_path
                    file_to_hash_internally = temp_file_created
                    success = True
                    if target_sr != target_sr_base:
                        print(f"  FLAC: SR ridotto a {target_sr} Hz per superare la verifica")
                    print(f"  -> Convertito: '{os.path.basename(temp_file_created)}'")
                    break
                else:
                    try:
                        if temp_pcm_path and os.path.exists(temp_pcm_path):
                            os.remove(temp_pcm_path)
                        temp_pcm_path = None
                    except OSError:
                        pass

            if not success:
                print(f"!! Verifica FLAC fallita a tutti i SR disponibili ≤ {target_sr_base} Hz")
                return None, filepath, None

        else:
            wav_sr_candidates = [352800, 176400, 88200, 44100, 384000, 192000, 96000, 48000]
            wav_sr_candidates = [sr for sr in wav_sr_candidates if sr <= target_sr_base]
            if not wav_sr_candidates:
                wav_sr_candidates = [44100]

            success = False
            temp_pcm_path = None
            for target_sr in wav_sr_candidates:
                fd, temp_pcm_path = tempfile.mkstemp(suffix='.wav')
                os.close(fd)
                cmd = [
                    'ffmpeg', '-y', '-hide_banner', '-nostats', '-nostdin',
                    '-threads', str(threads),
                    '-i', filepath,
                    '-map', '0:a:0', '-vn', '-sn', '-dn',
                    '-af', soxr_opts, '-ar', str(target_sr),
                    '-c:a', 'pcm_s24le' if target_bits == 24 else 'pcm_s16le',
                    '-map_metadata', '0',
                    temp_pcm_path
                ]
                proc = subprocess.run(cmd, check=False, capture_output=True, text=True, encoding='utf-8', errors='replace')
                sz = _filesize(temp_pcm_path)
                if proc.returncode != 0 or not sz:
                    try:
                        if temp_pcm_path and os.path.exists(temp_pcm_path):
                            os.remove(temp_pcm_path)
                        temp_pcm_path = None
                    except OSError:
                        pass
                    continue

                vcmd = ['ffmpeg', '-v', 'error', '-hide_banner', '-nostdin', '-i', temp_pcm_path, '-map', '0:a:0', '-f', 'null', '-']
                vproc = subprocess.run(vcmd, check=False, capture_output=True, text=True, encoding='utf-8', errors='replace')
                if vproc.returncode == 0:
                    temp_file_created = temp_pcm_path
                    file_to_hash_internally = temp_file_created
                    success = True
                    if target_sr != target_sr_base:
                        print(f"  WAV: SR ridotto a {target_sr} Hz per superare la verifica")
                    print(f"  -> Convertito: '{os.path.basename(temp_file_created)}'")
                    break
                else:
                    try:
                        if temp_pcm_path and os.path.exists(temp_pcm_path):
                            os.remove(temp_pcm_path)
                        temp_pcm_path = None
                    except OSError:
                        pass

            if not success:
                print(f"!! Verifica WAV fallita a tutti i SR disponibili ≤ {target_sr_base} Hz")
                return None, filepath, None

    cmd_md5 = [
        'ffmpeg', '-hide_banner', '-nostats', '-nostdin',
        '-fflags', '+bitexact', '-threads', str(threads),
        '-i', file_to_hash_internally, '-map', '0:a',
        '-vn', '-sn', '-dn', '-map_metadata', '-1',
        '-c:a', 'pcm_s32le', '-f', 'md5', '-'
    ]
    try:
        process = subprocess.run(cmd_md5, check=True, capture_output=True, text=True, encoding='utf-8', errors='replace')
        md5_output = (process.stdout or "").strip()
        if md5_output.startswith("MD5="):
            md5_hash = md5_output.split('=')[1]
            path_for_full_analysis = temp_file_created if temp_file_created else filepath
            return md5_hash, path_for_full_analysis, temp_file_created
        else:
            print(f"!! Output MD5 non atteso per '{os.path.basename(file_to_hash_internally)}': {md5_output}")
            if temp_file_created and os.path.exists(temp_file_created):
                return None, temp_file_created, temp_file_created
            return None, filepath, None
    except subprocess.CalledProcessError as e:
        if "Invalid data found when processing input" in (e.stderr or ""):
            print(f"  !! File corrotto rilevato: '{os.path.basename(file_to_hash_internally)}' (dati audio non validi)")
        else:
            print(f"!! Errore FFmpeg calcolo MD5 per '{os.path.basename(file_to_hash_internally)}': {(e.stderr or '')[:400]}")
        if temp_file_created and os.path.exists(temp_file_created):
            return None, temp_file_created, temp_file_created
        return None, filepath, None
    except FileNotFoundError:
        print(f"!! FFmpeg non trovato per calcolo MD5 di '{os.path.basename(file_to_hash_internally)}'.")
        if temp_file_created and os.path.exists(temp_file_created):
            return None, temp_file_created, temp_file_created
        return None, filepath, None

# Dizionario di tolleranze per arrotondare ogni metrica
SIG_TOL = {
    'true_peak_est_dbtp'      : 0.05,
    'isp_margin_db'           : 0.05,
    'peak_dbfs_fl_sample'     : 0.02,
    'peak_dbfs_fr_sample'     : 0.02,
    'dc_offset_dbfs'          : 0.5,
    'noise_floor_dbfs'        : 0.2,
    'noise_index_db'          : 0.2,
    'noise_spur_db'           : 0.2,
    'dr_tt_avg'               : 0.05,
    'plr_est'                 : 0.05,
    'lra_est'                 : 0.10,
    'loudness_lufs'           : 0.10,
    'st_lufs_iqr_db'          : 0.10,
    'spectral_balance_dev_db' : 0.10,
    'reverb_tail_ratio_db'    : 0.10,
    'stereo_width_iqr_db'     : 0.10,
    'jitter_ppm'              : 2,
    'hf_rms_var_db'           : 0.05,
    'hf_var_norm_pct'         : 0.5,
    'transient_crest_med'     : 0.05,
    'transient_rise_med_ms'   : 0.05,
    'stereo_correlation'      : 0.001,
}

_SIGNATURE_KEYS = (
    [
        'samplerate',
        'bit_depth',
        'subtype',
        'clipping_detected',
    ] +
    list(SIG_TOL.keys())
)

def _round_with_tol(val, tol):
    if val is None:
        return None
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    try:
        f = float(val)
    except (TypeError, ValueError):
        return str(val).lower()

    if not np.isfinite(f):
        return f

    if tol == 0:
        return f
    return round(f / tol) * tol

def make_signature(r):
    sig_parts = []
    for key in _SIGNATURE_KEYS:
        val = r.get(key)
        tol = SIG_TOL.get(key, 1e-6)
        sig_parts.append(_round_with_tol(val, tol))

    extras_tols = {
        'noise_floor_raw_dbfs': 0.2,
        'nf_broadband_dbfs': 0.2,
        'noise_windows_count': 0,
        'noise_total_duration_sec': 0.1,
        'noise_total_duration_uncapped_sec': 0.1,
        'score_mix_eff_H': 0.01,
        'score_mix_eff_Q': 0.01,
        'noise_selection_mode': 0,
        'loudness_backend': 0,
        'notch_f0': 0.1,
        'notch_harmonics': 0,
        'nf_notch_reduction_db': 0.1,
        'music_limited': 0,
        'noise_rms_p90_dbfs': 0.1,
        'noise_floor_sanity_cap_dbfs': 0.1,
        'noise_floor_sanity_applied': 0,
        'noise_floor_cap_reason': 0,
    }

    nf_raw = r.get('noise_floor_raw_dbfs')
    nf_bb = r.get('nf_broadband_dbfs', r.get('noise_floor_dbfs'))
    nwc = r.get('noise_windows_count')
    ndur = r.get('noise_total_duration_sec')
    ndur_unc = r.get('noise_total_duration_uncapped_sec')
    mix_eff = r.get('score_mix_eff') if isinstance(r.get('score_mix_eff'), dict) else {}
    mixH = mix_eff.get('H')
    mixQ = mix_eff.get('Q')
    nmode = r.get('noise_selection_mode')
    ldb = r.get('loudness_backend')
    notch_f0 = r.get('notch_f0')
    notch_h = r.get('notch_harmonics')
    nf_red = r.get('nf_notch_reduction_db')
    ml = r.get('music_limited')
    p90 = r.get('noise_rms_p90_dbfs')
    nf_cap = r.get('noise_floor_sanity_cap_dbfs')
    nf_cap_applied = r.get('noise_floor_sanity_applied')
    nf_cap_reason = r.get('noise_floor_cap_reason')

    extra_vals = [
        ('noise_floor_raw_dbfs', nf_raw),
        ('nf_broadband_dbfs', nf_bb),
        ('noise_windows_count', nwc),
        ('noise_total_duration_sec', ndur),
        ('noise_total_duration_uncapped_sec', ndur_unc),
        ('score_mix_eff_H', mixH),
        ('score_mix_eff_Q', mixQ),
        ('noise_selection_mode', nmode),
        ('loudness_backend', ldb),
        ('notch_f0', notch_f0),
        ('notch_harmonics', notch_h),
        ('nf_notch_reduction_db', nf_red),
        ('music_limited', ml),
        ('noise_rms_p90_dbfs', p90),
        ('noise_floor_sanity_cap_dbfs', nf_cap),
        ('noise_floor_sanity_applied', nf_cap_applied),
        ('noise_floor_cap_reason', nf_cap_reason),
    ]

    for k, v in extra_vals:
        tol = extras_tols[k]
        sig_parts.append(_round_with_tol(v, tol))

    return tuple(sig_parts)

def move_winner_files(winner_representative_result, md5_audio_groups, temp_files_registry, temp_files_to_keep_set):
    """
    Sposta i file del gruppo vincitore (tutti i file originali del gruppo MD5 del vincitore).
    Se un file DSD è stato convertito in FLAC/WAV e quel temporaneo (come rappresentante) vince,
    è il file temporaneo che viene spostato, con estensione coerente al contenitore creato.
    temp_files_to_keep_set: un SET a cui aggiungere i path dei file temporanei che vengono spostati.
    """
    if winner_representative_result is None:
        return
    dest_dir = os.getcwd()

    analyzed_file_path = winner_representative_result['filepath']
    original_path_of_winner_rep = analyzed_file_path
    for orig_p, temp_p in temp_files_registry.items():
        if temp_p == analyzed_file_path:
            original_path_of_winner_rep = orig_p
            break

    original_paths_to_consider_moving = []
    found_group = False
    for md5_hash, original_paths_in_group in md5_audio_groups.items():
        if original_path_of_winner_rep in original_paths_in_group:
            original_paths_to_consider_moving = original_paths_in_group
            print(f"\n--- Spostamento Vincitore e suoi {len(original_paths_in_group)-1} Duplicati Audio (MD5: {md5_hash[:8]}...) ---")
            found_group = True
            break
    if not found_group:
        original_paths_to_consider_moving = [original_path_of_winner_rep]
        print(f"\n--- Spostamento Vincitore ('{os.path.basename(original_path_of_winner_rep)}') ---")

    for original_file_path_in_winner_group in original_paths_to_consider_moving:
        path_to_actually_move = temp_files_registry.get(original_file_path_in_winner_group, original_file_path_in_winner_group)

        if not os.path.exists(path_to_actually_move):
            print(f"!! Attenzione: File sorgente '{os.path.basename(path_to_actually_move)}' (da originale '{os.path.basename(original_file_path_in_winner_group)}') non trovato. Impossibile spostare.")
            continue

        base_for_dest = os.path.basename(original_file_path_in_winner_group)
        if path_to_actually_move != original_file_path_in_winner_group and original_file_path_in_winner_group.lower().endswith(('.dsf', '.dff')):
            base_no_ext = os.path.splitext(os.path.basename(original_file_path_in_winner_group))[0]
            ext_temp = os.path.splitext(path_to_actually_move)[1].lower()
            if ext_temp in ('.flac', '.wav'):
                base_for_dest = base_no_ext + ext_temp
            else:
                base_for_dest = base_no_ext + ".flac"

        dst_name_part, dst_ext_part = os.path.splitext(base_for_dest)
        final_dest_path = os.path.join(dest_dir, base_for_dest)
        n = 1
        while os.path.exists(final_dest_path):
            final_dest_path = os.path.join(dest_dir, f"{dst_name_part}_{n}{dst_ext_part}")
            n += 1
            if n > 100:
                print(f"!! Errore: Troppi file con nome simile a '{base_for_dest}'. Spostamento annullato per questo file.")
                final_dest_path = None
                break
        if not final_dest_path:
            continue

        try:
            shutil.move(path_to_actually_move, final_dest_path)
            print(f"Spostato '{os.path.basename(path_to_actually_move)}' → '{final_dest_path}'")
            if path_to_actually_move in temp_files_registry.values():
                temp_files_to_keep_set.add(path_to_actually_move)
        except Exception as e:
            print(f"Impossibile spostare '{os.path.basename(path_to_actually_move)}': {e}")

def _map_to_score_range(value, params):
    """
    Mappa un valore numerico nella fascia [0.0 .. 1.0].
    • good  = 1
    • bad   = 0
    • higher_is_better  True/False
    """
    if value is None or not params:
        return 0.0

    # gestisci valori booleani come 1 (True) / 0 (False)
    if isinstance(value, (bool, np.bool_)):
        return 1.0 if value else 0.0

    if not np.isfinite(value):
        return 0.0

    good = params.get('good')
    bad  = params.get('bad')
    hib  = params.get('higher_is_better', True)

    if good is None or bad is None:
        return 0.0

    value = float(value)

    if hib:
        if value >= good:
            return 1.0
        if value <= bad:
            return 0.0
        return (value - bad) / (good - bad)
    else:
        if value <= good:
            return 1.0
        if value >= bad:
            return 0.0
        return (bad - value) / (bad - good)

def prompt_num_workers(files_meta=None):
    import os, math
    max_cpu = os.cpu_count() or 1
    try:
        import psutil
        avail = getattr(psutil.virtual_memory(), "available", None)
    except Exception:
        avail = None
    try:
        env_per_worker = os.environ.get("AQC_PER_WORKER_MB")
        per_worker_mb = float(env_per_worker) if env_per_worker is not None else 350.0
        if not (per_worker_mb > 0):
            per_worker_mb = 350.0
    except Exception:
        per_worker_mb = 350.0
    try:
        env_frac = os.environ.get("AQC_MEM_USE_FRACTION")
        mem_use_fraction = float(env_frac) if env_frac is not None else 0.75
        if not (0.1 <= mem_use_fraction <= 0.95):
            mem_use_fraction = 0.75
    except Exception:
        mem_use_fraction = 0.75
    if avail is not None and isinstance(avail, (int, float)) and avail > 0:
        mem_cap = int((avail * mem_use_fraction) // (per_worker_mb * 1024 * 1024))
        mem_cap = max(1, mem_cap)
    else:
        mem_cap = max(1, max_cpu // 2)
    cpu_cap = max(1, int(math.ceil(max_cpu * 0.75)))
    hard_cap = max(1, min(mem_cap, cpu_cap, max_cpu))
    rec = max(1, min(hard_cap, max_cpu // 2))
    N = None
    if isinstance(files_meta, (list, tuple)) and files_meta:
        N = len(files_meta)
        srs = []
        durs = []
        for item in files_meta:
            try:
                p = item[0]
                try:
                    info = sf.info(p)
                    sr = int(getattr(info, "samplerate", 0) or 0)
                    frames = int(getattr(info, "frames", 0) or 0)
                    dur = float(frames) / float(sr) if (sr and frames) else 0.0
                except Exception:
                    sr = 0
                    dur = 0.0
                if sr <= 0 or not np.isfinite(sr):
                    try:
                        import subprocess
                        out = subprocess.run(
                            ['ffprobe', '-v', 'error', '-select_streams', 'a:0',
                             '-show_entries', 'stream=sample_rate,duration',
                             '-of', 'default=noprint_wrappers=1:nokey=1', p],
                            check=False, capture_output=True, text=True, encoding='utf-8', errors='replace'
                        )
                        vals = [t.strip() for t in (out.stdout or "").splitlines() if t.strip()]
                        sr_guess = 0
                        dur_guess = 0.0
                        for t in vals:
                            if t.isdigit():
                                sr_guess = int(t)
                            else:
                                try:
                                    v = float(t.replace(",", "."))
                                    if v > 0:
                                        dur_guess = v
                                except Exception:
                                    pass
                        sr = sr_guess if sr_guess > 0 else 48000
                        dur = dur_guess if dur_guess > 0 else 180.0
                    except Exception:
                        sr = 48000
                        dur = 180.0
                srs.append(float(sr))
                durs.append(float(dur))
            except Exception:
                srs.append(48000.0)
                durs.append(180.0)
        if srs and durs:
            srs_arr = np.array(srs, dtype=float)
            durs_arr = np.array(durs, dtype=float)
            sr_p90 = float(np.percentile(srs_arr, 90))
            dur_p90 = float(np.percentile(durs_arr, 90))
            if sr_p90 >= 176400:
                sr_factor = 0.6
            elif sr_p90 >= 96000:
                sr_factor = 0.7
            else:
                sr_factor = 1.0
            if dur_p90 >= 1800.0:
                dur_factor = 0.6
            elif dur_p90 >= 600.0:
                dur_factor = 0.8
            else:
                dur_factor = 1.0
            cap_by_jobs = N
            rec_est = int(max(1, min(hard_cap, cap_by_jobs, round(hard_cap * sr_factor * dur_factor))))
            rec = max(1, rec_est)
    try:
        auto = (os.environ.get("AQC_AUTO_WORKERS", "").strip().lower() in ("1", "true", "on", "yes", "y", "auto", "max"))
        if auto:
            if os.environ.get("AQC_AUTO_WORKERS", "").strip().lower() == "max":
                if N is None and isinstance(files_meta, (list, tuple)):
                    N = len(files_meta)
                return max(1, min(hard_cap, N if N is not None else hard_cap))
            return rec
    except Exception:
        pass
    max_display = max_cpu
    while True:
        try:
            hint = f"{rec}"
            ram_txt = f"mem≈{per_worker_mb:.0f}MB/worker, frazione RAM={int(mem_use_fraction*100)}%"
            rng = f"[1-{max_display}]"
            extra = ""
            if isinstance(N, int):
                extra = f", job={N}"
            raw = input(f"Quanti core vuoi usare per l'analisi? {rng} (Invio={hint} consigliato, {ram_txt}{extra}): ").strip().lower()
            if raw == "":
                return rec
            if raw in ("auto",):
                return rec
            if raw in ("max",):
                nmax = max(1, min(hard_cap, max_cpu if N is None else min(max_cpu, N)))
                return nmax
            n = int(raw)
            if 1 <= n <= max_display:
                if n > mem_cap:
                    print(f"Attenzione: {n} worker eccedono il limite di memoria sicuro (cap={mem_cap}); uso {mem_cap}.")
                    return mem_cap
                if N is not None and n > N:
                    print(f"Nota: richiesti {n} worker ma i job sono {N}; uso {N}.")
                    return N
                return n
        except (ValueError, EOFError, KeyboardInterrupt):
            pass
        print(f"Valore non valido. Inserisci un numero tra 1 e {max_display} (consigliato: {rec}, cap memoria: {mem_cap}).")

def _analyze_wrapper(args):
    path_for_analysis, original_ref_path, is_temp_for_call = args
    res = analyze_audio(path_for_analysis, is_temporary=is_temp_for_call)
    if res:
        res['_original_ref_path'] = original_ref_path
    return res


def get_verbal_assessment(score_100, r):
    pol = _policy_resolve()
    def s(key, val=None):
        v = r.get(key) if val is None else val
        sym = get_metric_status(key, v, policy=pol, r=r)
        return (sym or "").strip()
    msgs = []
    clipc = (r.get('clipping_class') or 'clean').lower()
    if clipc == 'hard':
        msgs.append("Escluso (DQ) per clipping severo.")
    elif clipc == 'borderline':
        msgs.append("Clipping borderline rilevato.")
    if isinstance(score_100, (int, float)):
        if score_100 >= 85:      qual = "eccellente"
        elif score_100 >= 70:    qual = "molto buona"
        elif score_100 >= 55:    qual = "buona"
        elif score_100 >= 40:    qual = "discreta"
        else:                    qual = "migliorabile"
        msgs.append(f"Qualità tecnica {qual}.")
    conf_disp = (r.get("noise_confidence") or "").strip()
    conf_raw = conf_disp.lower()
    if conf_raw.startswith("alt"):
        conf_label = "Alta"
    elif conf_raw.startswith("med"):
        conf_label = "Media"
    elif conf_raw.startswith("bas"):
        conf_label = "Bassa"
    else:
        conf_label = "Alta"
    if conf_label != "Alta":
        msgs.append(f"Misure del rumore con confidenza {conf_disp.lower()}.")
    nwin = int(r.get("noise_windows_count") or 0)
    ndur = float(r.get("noise_total_duration_sec") or 0.0)
    diff = float(r.get("noise_consistency_diff_db") or 0.0)
    extreme_uncertain = (diff > 30.0) or (nwin < 2 and ndur < 3.0)
    moderate_uncertain = (diff > 20.0) or (nwin < 2 or ndur < 3.0)
    nf = r.get('noise_floor_dbfs')
    nf_raw = r.get('noise_floor_raw_dbfs')
    nf_bb = r.get('nf_broadband_dbfs', nf)
    spur_db = r.get('noise_spur_db')
    spur_label = r.get('noise_spur_label') or ""
    spur_hc = int(r.get('noise_spur_harmonics_count') or 0)
    nf_red = r.get('nf_notch_reduction_db')
    hum_dense = spur_label.startswith("HUM") and spur_hc >= 10
    nf_eval = nf_bb if (hum_dense and isinstance(nf_bb, (int, float)) and np.isfinite(nf_bb)) else nf
    if extreme_uncertain:
        msgs.append("Selezione rumore poco rappresentativa: NF escluso dal punteggio, si considera principalmente lo Spur.")
    elif moderate_uncertain:
        msgs.append("Selezione rumore limitata: peso NF ridotto.")
    nf_cap_applied = bool(r.get("noise_floor_sanity_applied"))
    nf_cap_val = r.get("noise_floor_sanity_cap_dbfs")
    nf_cap_reason = r.get("noise_floor_cap_reason")
    if nf_cap_applied and isinstance(nf_cap_val, (int, float)) and np.isfinite(nf_cap_val):
        if nf_cap_reason:
            msgs.append(f"NF sanity cap applicato a {nf_cap_val:.1f} dBFS ({nf_cap_reason}).")
        else:
            msgs.append(f"NF sanity cap applicato a {nf_cap_val:.1f} dBFS.")
    if extreme_uncertain:
        if isinstance(nf_bb, (int, float)) and np.isfinite(nf_bb):
            msgs.append(f"NF escluso: valore diagnostico broadband {nf_bb:.1f} dBFS.")
        elif isinstance(nf, (int, float)) and np.isfinite(nf):
            msgs.append(f"NF escluso: valore diagnostico {nf:.1f} dBFS.")
    else:
        if isinstance(nf_eval, (int, float)) and np.isfinite(nf_eval):
            if nf_eval <= -80:
                msgs.append("Noise floor molto basso.")
            elif nf_eval <= -60:
                msgs.append("Noise floor basso.")
            elif nf_eval > -40:
                msgs.append("Noise floor elevato, probabilmente influenzato dal contenuto musicale.")
            else:
                msgs.append("Noise floor moderato.")
    if hum_dense and isinstance(nf_bb, (int, float)) and np.isfinite(nf_bb) and isinstance(nf, (int, float)) and np.isfinite(nf) and (nf - nf_bb) >= 2.0:
        msgs.append(f"Valutazione NF su broadband con HUM separato (BB {nf_bb:.1f} dBFS).")
    if isinstance(nf_bb, (int, float)) and np.isfinite(nf_bb) and isinstance(nf, (int, float)) and np.isfinite(nf) and (nf_bb + 3.0) < nf:
        msgs.append(f"Tappeto broadband più basso rispetto al totale (al netto dei toni: {nf_bb:.1f} dBFS).")
    if isinstance(nf_red, (int, float)) and np.isfinite(nf_red) and nf_red >= 1.0:
        msgs.append(f"Parte del rumore è hum: notch riduce ~{nf_red:.1f} dB.")
    low_i = r.get('nf_interval_low_dbfs')
    high_i = r.get('nf_interval_high_dbfs')
    unc_i = r.get('nf_interval_unc_db')
    conf_i = r.get('nf_interval_conf_label')
    if low_i is not None and high_i is not None and unc_i is not None:
        msgs.append(f"NF intervallo credibile [{low_i:.1f},{high_i:.1f}] dBFS (±{unc_i:.1f}, conf={conf_i or 'N/A'}).")
    if isinstance(spur_db, (int, float)) and np.isfinite(spur_db):
        if spur_label.startswith("HUM"):
            if spur_db >= 25.0 or spur_hc >= 15:
                msgs.append("Hum di rete molto marcato e denso.")
            elif spur_db >= 12.0 or spur_hc >= 5:
                msgs.append("Hum di rete evidente.")
            else:
                msgs.append("Tracce di hum di rete.")
        elif spur_db >= 12.0:
            msgs.append("Toni spurî evidenti nel rumore.")
        elif spur_db >= 6.0:
            msgs.append("Possibili toni spurî nel rumore.")
    sel_mode = (r.get("noise_selection_mode") or "").strip().lower()
    notch_h = int(r.get("notch_harmonics") or 0)
    if sel_mode == "strict+notch":
        if notch_h > 0:
            msgs.append("Selettore rumore con notch anti‑hum applicato.")
        else:
            msgs.append("Selettore rumore con notch anti‑hum.")
    elif sel_mode == "strict":
        msgs.append("Selettore rumore in modalità strict.")
    if bool(r.get("music_limited")) and not extreme_uncertain:
        msgs.append("Contenuto sempre pieno: selezione rumore in regime music‑limited.")
    dr_sym  = s('dr_tt_avg')
    plr_sym = s('plr_est', r.get('plr_effective_db', r.get('plr_est')))
    lra_sym = s('lra_est')
    sti_sym = s('st_lufs_iqr_db')
    ton_sym = s('spectral_balance_dev_db')
    wid_sym = s('stereo_width_iqr_db')
    if dr_sym == '✅':
        msgs.append("Dinamica ampia.")
    elif dr_sym == '❌':
        msgs.append("Gamma dinamica ridotta.")
    if plr_sym == '❌':
        msgs.append("PLR limitato (headroom ridotto).")
    elif plr_sym == '✅':
        msgs.append("PLR solido.")
    if lra_sym == '❌':
        msgs.append("Loudness range contenuto.")
    elif lra_sym == '✅':
        msgs.append("Buon range di loudness.")
    if sti_sym == '❌':
        msgs.append("Compressione short‑term marcata.")
    elif sti_sym == '✅':
        msgs.append("Buona variabilità short‑term.")
    if ton_sym == '❌':
        msgs.append("Bilanciamento tonale squilibrato.")
    elif ton_sym == '✅':
        msgs.append("Tonalità ben bilanciata.")
    if wid_sym == '❌':
        msgs.append("Variabilità di ampiezza stereo contenuta.")
    elif wid_sym == '✅':
        msgs.append("Buona variabilità di ampiezza stereo.")
    dc = r.get('dc_offset_dbfs')
    if isinstance(dc, (int, float)) and np.isfinite(dc):
        if dc > -90.0:
            msgs.append("DC offset misurabile.")
    jit_val = r.get('jitter_ppm')
    hf_var = r.get('hf_rms_var_db')
    if isinstance(jit_val, (int, float)) and np.isfinite(jit_val):
        jit_sym = s('jitter_ppm', jit_val)
        if jit_sym == '❌':
            msgs.append("Jitter elevato.")
        elif jit_sym == '⚠️':
            msgs.append("Jitter moderato.")
    elif isinstance(hf_var, (int, float)) and np.isfinite(hf_var):
        hf_sym = s('hf_rms_var_db', hf_var)
        if hf_sym == '❌':
            msgs.append("Fluttuazioni HF elevate.")
        elif hf_sym == '⚠️':
            msgs.append("Fluttuazioni HF moderate.")
    return " ".join(msgs)

def announce_winner(winner_representative_result, md5_audio_groups, temp_files_registry):
    import os
    import numpy as np

    def fmt_num(val, pattern=".1f", unit=""):
        return f"{val:{pattern}}{unit}" if val is not None and np.isfinite(val) else "N/A"
    def fmt_signed(val, pattern=".2f"):
        if val is None or not isinstance(val, (int, float)) or not np.isfinite(val):
            return "N/A"
        sign = "+" if float(val) >= 0 else ""
        return f"{sign}{val:{pattern}}"
    def fmt_dc(val):
        if val is None or (isinstance(val, float) and not np.isfinite(val)):
            return "N/A"
        try:
            v = float(val)
            if v <= -180.0:
                return "≤-180.0 dBFS"
            return f"{v:.1f} dBFS"
        except Exception:
            return "N/A"
    def fmt_time_range(pair, sr):
        try:
            s, e = pair
            s = int(s); e = int(e); sr = float(sr)
            if sr <= 0 or e <= s:
                return None
            def mmss(t):
                m = int(t // 60)
                s = int(t % 60)
                return f"{m:02d}:{s:02d}"
            return f"{mmss(s/sr)}–{mmss(e/sr)}"
        except Exception:
            return None
    def _sf(x):
        try:
            f = float(x)
            return f if np.isfinite(f) else None
        except Exception:
            return None
    def _nf_interval_from_r(r):
        nf_bb = _sf(r.get('nf_broadband_dbfs'))
        nf_raw = _sf(r.get('noise_floor_raw_dbfs'))
        nf_cap = _sf(r.get('noise_floor_dbfs'))
        center = nf_bb if nf_bb is not None else (nf_raw if nf_raw is not None else nf_cap)
        if center is None:
            return None
        nwin = int(r.get("noise_windows_count") or 0)
        ndur = _sf(r.get("noise_total_duration_sec")) or 0.0
        diff = _sf(r.get("noise_consistency_diff_db")) or 0.0
        nf_cross = _sf(r.get("noise_floor_cross_rms_dbfs"))
        nf_cap_appl = bool(r.get("noise_floor_sanity_applied"))
        if (nwin >= 3 and ndur >= 3.0 and diff <= 6.0):
            unc = 2.0; conf = "Alta"
        elif (nwin >= 2 and ndur >= 2.0 and diff <= 10.0):
            unc = 3.0; conf = "Media"
        elif diff <= 20.0:
            unc = 4.0; conf = "Media"
        elif diff <= 30.0:
            unc = 5.0; conf = "Bassa"
        else:
            unc = 6.0; conf = "Bassa"
        if nf_cross is not None:
            delta = abs(center - nf_cross)
            if delta > unc:
                unc = min(8.0, delta + 1.0)
        if nf_cap_appl:
            unc = max(unc, 4.0)
        low = center - unc
        high = center + unc
        return (low, high, center, unc, conf)

    tie_mode = (os.environ.get("AQC_TIEBREAK", "none").strip().lower())
    dq_last  = (os.environ.get("AQC_RANK_DQ_LAST", "1").strip().lower() in ("1", "true", "yes", "on"))
    if tie_mode in ("none", "off", "0"):
        crit = "score (solo score)"
    elif tie_mode in ("residual", "residuo", "default"):
        crit = "score + tie-break (residuo)"
    else:
        crit = f"score + tie-break ({tie_mode})"
    crit += " | DQ in coda" if dq_last else " | DQ inclusi"

    def _conf_debug_from_winner(r):
        if tie_mode in ("none", "off", "0"):
            return "Alta (score)"
        resid = r.get('pair_residual_to_next_db')
        try:
            if isinstance(resid, (int, float)) and np.isfinite(resid):
                if resid <= -3.0:
                    return "Alta (residuo)"
                if resid <= -1.5:
                    return "Media (residuo)"
                return "Bassa (residuo)"
        except Exception:
            pass
        return "Alta (score)"

    print("\n--- Vincitore ---")
    if winner_representative_result is None:
        print("Errore: nessun vincitore determinato.")
        return

    r = winner_representative_result
    analyzed_file_path = r['filepath']
    original_path_of_winner_rep = analyzed_file_path
    for orig_p, temp_p in temp_files_registry.items():
        if temp_p == analyzed_file_path:
            original_path_of_winner_rep = orig_p
            break

    print(f"🥇 '{os.path.basename(original_path_of_winner_rep)}' (Score: {r.get('score', 'N/A')}/100) 🥇")
    print(f"Criterio: {crit}")
    if analyzed_file_path != original_path_of_winner_rep:
        print(f"   (Analisi basata su: '{os.path.basename(analyzed_file_path)}')")

    clipc = r.get('clipping_class', 'clean')
    sc = r.get('score_float')
    pre = r.get('score_pre_cap')
    fine = r.get('fine_score')
    tie = r.get('tie_bonus')
    H = r.get('hygiene_score')
    Q = r.get('quality_score')
    mix_eff = r.get('score_mix_eff') or {}
    conf_raw = (r.get("noise_confidence") or "").strip()
    conf_disp = conf_raw if conf_raw else "Alta"
    state = "DQ" if (clipc or "").lower() == "hard" else None

    extras = []
    if isinstance(pre, (int, float)):
        extras.append(f"pre {pre:.2f}")
    if isinstance(fine, (int, float)) and np.isfinite(fine):
        extras.append(f"fine {fine:.2f}")
    if isinstance(tie, (int, float)) and np.isfinite(tie) and abs(tie) > 0:
        extras.append(f"tie {fmt_signed(tie)}")
    extras_str = f" ({', '.join(extras)})" if extras else ""

    print(f"Score finale: {fmt_num(sc)}{extras_str}")
    if state:
        print(f"Stato: {state}")
    if isinstance(H, (int, float)) or isinstance(Q, (int, float)):
        print(f"Componenti: H={fmt_num(H, pattern='.3f')} | Q={fmt_num(Q, pattern='.3f')}")
    if isinstance(mix_eff, dict) and mix_eff:
        print(f"Mix H/Q effettivo: {fmt_num(mix_eff.get('H'), pattern='.2f')} / {fmt_num(mix_eff.get('Q'), pattern='.2f')}")
    print(f"Conf. rumore: {conf_disp}")

    print(f"Conf. debug: {_conf_debug_from_winner(r)}")

    winner_md5_group_paths = None
    for md5_hash, original_paths_in_group in md5_audio_groups.items():
        if original_path_of_winner_rep in original_paths_in_group:
            winner_md5_group_paths = original_paths_in_group
            print(f"\nQuesto file appartiene a un gruppo di {len(winner_md5_group_paths)} file audio-identici (MD5: {md5_hash[:8]}...).")
            if len(winner_md5_group_paths) > 1:
                print("   Altri file nel gruppo:")
                for opath in winner_md5_group_paths:
                    if opath != original_path_of_winner_rep:
                        print(f"     ↳ {os.path.basename(opath)}")
            break
    if not winner_md5_group_paths:
        print("\n(Nessun altro file audio-identico trovato per il vincitore, o MD5 non calcolato).")

    print("\nMotivazioni principali (basate sull’analisi del file rappresentativo):")
    assessment = r.get('assessment', 'Punteggio tecnico complessivo più alto.')
    sentences = [s.strip() for s in str(assessment).split('.') if s.strip()]
    motivations = ". ".join(sentences[:2]) + "." if len(sentences) >= 2 else assessment
    print(f"  • {motivations}")

    g_to_next = r.get('pair_gain_to_next_db')
    r_to_next = r.get('pair_residual_to_next_db')
    d_to_next = r.get('pair_drift_to_next_ppm')
    sm_next = r.get('pair_same_master_next')
    sim_next = r.get('pair_similarity')
    rL = r.get('pair_residual_low_db')
    rM = r.get('pair_residual_mid_db')
    rH = r.get('pair_residual_high_db')
    conf_cmp = r.get('pair_compare_conf')

    if isinstance(r_to_next, (int, float)) and np.isfinite(r_to_next) and sm_next is None:
        sm_next = bool(r_to_next <= -35.0)
    if d_to_next is None or not isinstance(d_to_next, (int, float)) or not np.isfinite(d_to_next):
        d_to_next = 0.0
    if sim_next is None or not isinstance(sim_next, (int, float)) or not np.isfinite(sim_next):
        sim_next = 0.0
    if conf_cmp is None or not isinstance(conf_cmp, (int, float)) or not np.isfinite(conf_cmp):
        conf_cmp = 0.0

    if g_to_next is not None or r_to_next is not None or sm_next is not None or d_to_next is not None:
        gm = fmt_num(g_to_next, pattern=".2f", unit=" dB")
        rm = fmt_num(r_to_next, pattern=".1f", unit=" dBFS")
        dm = fmt_num(d_to_next, pattern=".1f", unit=" ppm")
        print(f"\nConfronto con il 2° classificato:")
        print(f"  • Offset di gain ottimale: {gm}")
        print(f"  • Residuo relativo: {rm}")
        print(f"  • Drift: {dm}")
        print(f"  • Stessa master (solo volume): {'SÌ' if bool(sm_next) else 'NO'}")
        print(f"  • Similarità: {fmt_num(sim_next, pattern='.3f')}")
        print(f"  • Conf. confronto: {fmt_num(conf_cmp, pattern='.2f')}")
        if any(isinstance(v, (int, float)) and np.isfinite(v) for v in (rL, rM, rH)):
            print(f"  • Residuo bande (dB): Low={fmt_num(rL, pattern='.1f')} | Mid={fmt_num(rM, pattern='.1f')} | High={fmt_num(rH, pattern='.1f')}")

    print("\nDistribuzione ISP (50 ms):")
    wc50 = r.get("isp_window_count_50ms", r.get("isp_window_count"))
    p05_50 = r.get("isp_margin_p05_db_50ms", r.get("isp_margin_p05_db"))
    p50_50 = r.get("isp_margin_p50_db_50ms", r.get("isp_margin_p50_db"))
    p95_50 = r.get("isp_margin_p95_db_50ms", r.get("isp_margin_p95_db"))
    u1c_50 = r.get("isp_under_1db_count_50ms", r.get("isp_under_1db_count"))
    u05c_50 = r.get("isp_under_05db_count_50ms", r.get("isp_under_05db_count"))
    u02c_50 = r.get("isp_under_02db_count_50ms", r.get("isp_under_02db_count"))
    u1f_50 = r.get("isp_under_1db_frac_50ms", r.get("isp_under_1db_frac"))
    u05f_50 = r.get("isp_under_05db_frac_50ms", r.get("isp_under_05db_frac"))
    u02f_50 = r.get("isp_under_02db_frac_50ms", r.get("isp_under_02db_frac"))
    print(f"  • Finestre: {wc50 if isinstance(wc50,(int,float)) else 'N/A'} | P05/P50/P95: {fmt_num(p05_50,'.2f',' dB')}/{fmt_num(p50_50,'.2f',' dB')}/{fmt_num(p95_50,'.2f',' dB')}")
    print(f"  • <1.0/<0.5/<0.2 dB: {fmt_num(u1c_50,'.0f')} ({fmt_num((u1f_50 or 0)*100,'.1f','%')}) / {fmt_num(u05c_50,'.0f')} ({fmt_num((u05f_50 or 0)*100,'.1f','%')}) / {fmt_num(u02c_50,'.0f')} ({fmt_num((u02f_50 or 0)*100,'.1f','%')})")

    print("\nDistribuzione ISP (20 ms):")
    wc20 = r.get("isp_window_count_20ms")
    p05_20 = r.get("isp_margin_p05_db_20ms")
    p50_20 = r.get("isp_margin_p50_db_20ms")
    p95_20 = r.get("isp_margin_p95_db_20ms")
    u1c_20 = r.get("isp_under_1db_count_20ms")
    u05c_20 = r.get("isp_under_05db_count_20ms")
    u02c_20 = r.get("isp_under_02db_count_20ms")
    u1f_20 = r.get("isp_under_1db_frac_20ms")
    u05f_20 = r.get("isp_under_05db_frac_20ms")
    u02f_20 = r.get("isp_under_02db_frac_20ms")
    ratio05 = r.get("isp_multi_ratio_05")
    ratio02 = r.get("isp_multi_ratio_02")
    tag = r.get("isp_multi_consistency")
    print(f"  • Finestre: {wc20 if isinstance(wc20,(int,float)) else 'N/A'} | P05/P50/P95: {fmt_num(p05_20,'.2f',' dB')}/{fmt_num(p50_20,'.2f',' dB')}/{fmt_num(p95_20,'.2f',' dB')}")
    print(f"  • <1.0/<0.5/<0.2 dB: {fmt_num(u1c_20,'.0f')} ({fmt_num((u1f_20 or 0)*100,'.1f','%')}) / {fmt_num(u05c_20,'.0f')} ({fmt_num((u05f_20 or 0)*100,'.1f','%')}) / {fmt_num(u02c_20,'.0f')} ({fmt_num((u02f_20 or 0)*100,'.1f','%')})")
    print(f"  • Multi-consistenza: ratio0.5={fmt_num(ratio05,'.2f')} | ratio0.2={fmt_num(ratio02,'.2f')} | tag={tag or 'N/A'}")

    nf = r.get('noise_floor_dbfs')
    nf_raw = r.get('noise_floor_raw_dbfs')
    nf_bb = r.get('nf_broadband_dbfs', nf)
    spur = r.get('noise_spur_db')
    isp = r.get('isp_margin_db')
    dc = r.get('dc_offset_dbfs')
    nf_cross = r.get('noise_floor_cross_rms_dbfs')
    nf_diff = r.get('noise_consistency_diff_db')
    nf_p90 = r.get('noise_floor_cross_p90_dbfs')
    nf_cap = r.get('noise_floor_sanity_cap_dbfs')
    nf_ap = r.get('noise_floor_sanity_applied')
    spur_label = r.get('noise_spur_label')
    spur_f0 = r.get('noise_spur_fundamental_hz')
    spur_hc = r.get('noise_spur_harmonics_count')
    ns_mode = (r.get("noise_selection_mode") or "").strip().lower()
    notch_h = int(r.get("notch_harmonics") or 0)
    ns_mode_label = ns_mode
    if ns_mode == "strict+notch" and notch_h > 0:
        ns_mode_label = "strict+notch (applied)"

    print(f"\n> Selettore rumore: {ns_mode_label} | Finestre: {fmt_num(r.get('noise_windows_count'),'.0f')} | Durata: {fmt_num(r.get('noise_total_duration_sec'),'.2f',' s')}")
    wins_preview = r.get("noise_windows_preview")
    if isinstance(wins_preview, (list, tuple)) and wins_preview:
        stamps = []
        for pair in wins_preview[:3]:
            s = fmt_time_range(pair, r.get('samplerate'))
            if s:
                stamps.append(s)
        if stamps:
            print(f"  • Finestre (anteprima): {', '.join(stamps)}")

    hum_dense = False
    try:
        lbl_tmp = str(spur_label or "")
        hc_tmp = int(spur_hc or 0)
        hum_dense = lbl_tmp.startswith("HUM") and hc_tmp >= 10
    except Exception:
        hum_dense = False

    print(f"\n> Riepilogo metriche del file analizzato: "
          f"DR={fmt_num(r.get('dr_tt_avg'), unit='dB')} | "
          f"LUFS={fmt_num(r.get('loudness_lufs'))} (backend: {r.get('loudness_backend','N/A')}) | "
          f"LRA={fmt_num(r.get('lra_est'), unit='LU')} | "
          f"NF={fmt_num(nf, pattern='.1f', unit=' dBFS')} (raw {fmt_num(nf_raw, pattern='.1f', unit=' dBFS')}, BB {fmt_num(nf_bb, pattern='.1f', unit=' dBFS')}) | "
          f"Spur={fmt_num(spur, pattern='.1f', unit=' dB')} | "
          f"ISP={fmt_num(isp, pattern='.2f', unit=' dB')} | "
          f"DC={fmt_dc(dc)} | "
          f"Clipping={'SÌ' if r.get('clipping_detected') else 'NO'}")

    nf_mode = r.get("nf_scoring_mode")
    if nf_mode is None:
        safe_env = os.environ.get("AQC_NF_SAFE", "").strip().lower()
        nf_safe = (safe_env not in ("0", "false", "off", "no", "n"))
        if (r.get('noise_consistency_diff_db') and float(r.get('noise_consistency_diff_db')) > 30.0) or (int(r.get('noise_windows_count') or 0) < 2 and float(r.get('noise_total_duration_sec') or 0.0) < 3.0):
            nf_mode = "EXCLUDED"
        else:
            nf_mode = "SAFE" if nf_safe else "PROFILE_24"
    print(f"> NF scoring mode: {nf_mode}")

def _policy_resolve(policy=None):
    """Ritorna la policy da usare."""
    return policy if isinstance(policy, dict) and "thresholds" in policy else get_quality_policy()



def _policy_metric_thresholds(metric_key, value, policy, r=None):
    pol = _policy_resolve(policy)
    th = pol["thresholds"]
    key = metric_key
    if key in ("true_peak_est_dbtp", "peak_dbfs_overall", "true_peak_dbtp"):
        key = "true_peak_dbtp"
    if key == "noise_floor_dbfs":
        nf_key = _policy_select_nf_profile(value, r, pol)
        base_thr = th.get(nf_key)
    elif key == "plr_effective_db":
        key = "plr_est"
        base_thr = th.get(key)
    else:
        base_thr = th.get(key)

    import copy, os
    prof = os.environ.get("AQC_PROFILE", "").strip().lower()
    if not base_thr or not isinstance(base_thr, dict) or not (prof.startswith("strict") or prof == "audiophile-strict"):
        return base_thr
    t = copy.deepcopy(base_thr)
    if key == "isp_margin_db":
        t["good"] = max(t.get("good", 1.0), 1.5)
        t["warn"] = max(t.get("warn", 0.5), 1.0)
    elif key == "spectral_balance_dev_db":
        t["good"] = min(t.get("good", 2.0), 1.5)
        t["warn"] = min(t.get("warn", 4.0), 3.0)
    elif key == "lra_est":
        t["good"] = max(t.get("good", 12.0), 14.0)
        t["warn"] = max(t.get("warn", 6.0), 10.0)
    return t

def policy_map_to_unit_score(value, thr):
    """
    Mappa un valore numerico nella fascia [0..1] in base alle soglie della policy.
    • good = 1, bad = 0, interpolazione lineare
    """
    if value is None or thr is None:
        return None
    try:
        v = float(value)
        if not np.isfinite(v):
            return None
    except (TypeError, ValueError):
        return None

    good = thr.get("good")
    bad = thr.get("bad")
    hib = thr.get("higher_is_better", True)
    if good is None or bad is None:
        return None

    good = float(good)
    bad = float(bad)

    if hib:
        if v >= good: return 1.0
        if v <= bad:  return 0.0
        return (v - bad) / (good - bad)
    else:
        if v <= good: return 1.0
        if v >= bad:  return 0.0
        return (bad - v) / (bad - good)

def get_metric_status(metric_key, value, policy=None, r=None):
    import os
    if metric_key == "clipping_detected":
        if value is None:
            return " N/A"
        try:
            return " ❌" if bool(value) else " ✅"
        except Exception:
            return " ???"
    if metric_key in ("true_peak_est_dbtp", "peak_dbfs_overall"):
        metric_key = "true_peak_dbtp"
    pol = _policy_resolve(policy)

    def _band_from_thr(v, thr):
        if thr is None or v is None:
            return " N/A"
        try:
            val = float(v)
        except Exception:
            return " N/A"
        if not np.isfinite(val):
            return " N/A"
        good = thr.get("good")
        warn = thr.get("warn")
        bad  = thr.get("bad")
        hib  = thr.get("higher_is_better", True)
        if good is None or warn is None or bad is None:
            return ""
        if hib:
            if val >= good:
                return " ✅"
            elif val <= bad:
                return " ❌"
            else:
                return " ⚠️ "
        else:
            if val <= good:
                return " ✅"
            elif val >= bad:
                return " ❌"
            else:
                return " ⚠️ "

    if metric_key == "noise_floor_dbfs":
        try:
            nwin = int((r or {}).get("noise_windows_count") or 0)
            ndur = float((r or {}).get("noise_total_duration_sec") or 0.0)
            diff = float((r or {}).get("noise_consistency_diff_db") or 0.0)
            conf_str = str((r or {}).get("noise_confidence") or "").strip().lower()
            extreme_uncertain = (diff > 30.0) or (nwin < 2 and ndur < 3.0) or ("escluso" in conf_str)
        except Exception:
            extreme_uncertain = False
        if extreme_uncertain:
            return " N/A"
        try:
            spur_label = str((r or {}).get("noise_spur_label") or "")
            spur_hc = int((r or {}).get("noise_spur_harmonics_count") or 0)
            hum_dense = spur_label.startswith("HUM") and spur_hc >= 10
        except Exception:
            hum_dense = False
        v_eff = None
        try:
            v_raw = None if value is None else float(value)
            if hum_dense:
                v_bb = (r or {}).get("nf_broadband_dbfs")
                v_eff = float(v_bb) if v_bb is not None and np.isfinite(float(v_bb)) else v_raw
            else:
                v_eff = v_raw
        except Exception:
            v_eff = None
        if v_eff is None or not np.isfinite(v_eff):
            return " N/A"
        nf_safe = (os.environ.get("AQC_NF_SAFE", "").strip().lower() not in ("0", "false", "off", "no", "n"))
        if nf_safe:
            thr16 = pol["thresholds"].get("noise_floor_dbfs_16")
            thr24 = pol["thresholds"].get("noise_floor_dbfs_24")
            b16 = _band_from_thr(v_eff, thr16) if thr16 else " N/A"
            b24 = _band_from_thr(v_eff, thr24) if thr24 else " N/A"
            def score_band(b):
                s = (b or "").strip()
                if s == "✅": return 3
                if s.startswith("⚠"): return 2
                if s == "❌": return 1
                return 0
            return b16 if score_band(b16) >= score_band(b24) else b24
        nf_key = _policy_select_nf_profile(v_eff, r, pol)
        thr = pol["thresholds"].get(nf_key)
        return _band_from_thr(v_eff, thr)

    if metric_key == "nf_broadband_dbfs":
        try:
            nwin = int((r or {}).get("noise_windows_count") or 0)
            ndur = float((r or {}).get("noise_total_duration_sec") or 0.0)
            diff = float((r or {}).get("noise_consistency_diff_db") or 0.0)
            conf_str = str((r or {}).get("noise_confidence") or "").strip().lower()
            extreme_uncertain = (diff > 30.0) or (nwin < 2 and ndur < 3.0) or ("escluso" in conf_str)
        except Exception:
            extreme_uncertain = False
        if extreme_uncertain:
            return " N/A"
        try:
            v = None if value is None else float(value)
        except Exception:
            v = None
        if v is None or not np.isfinite(v):
            return " N/A"
        nf_key = _policy_select_nf_profile(v, r, pol)
        thr = pol["thresholds"].get(nf_key)
        return _band_from_thr(v, thr)

    if metric_key == "dc_offset_dbfs":
        try:
            if value is None:
                return " N/A"
            v = float(value)
            if v == float("-inf") or v <= -180.0:
                return " ✅"
        except Exception:
            pass

    thr = _policy_metric_thresholds(metric_key, value, pol, r=r)
    if value is None:
        return " N/A"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return " ???"
    if not np.isfinite(v):
        return " N/A"
    return _band_from_thr(v, thr)

def prepare_analysis_views(L: np.ndarray, R: np.ndarray, mono: np.ndarray, sr: int, base_target_sr: int = 48000):
    if L is None or R is None or mono is None or sr is None or sr <= 0:
        return {"sr_base": sr, "mono_base": mono, "L_base": L, "R_base": R}
    try:
        L = np.asarray(L, dtype=np.float64, copy=False)
        R = np.asarray(R, dtype=np.float64, copy=False)
        mono = np.asarray(mono, dtype=np.float64, copy=False)
    except Exception:
        return {"sr_base": sr, "mono_base": mono, "L_base": L, "R_base": R}
    try:
        native = os.environ.get("AQC_NATIVE_SR", "").strip().lower() in ("1", "true", "on", "yes", "y")
    except Exception:
        native = False
    if native:
        return {"sr_base": int(sr), "mono_base": mono, "L_base": L, "R_base": R}
    base_sr = int(min(sr, base_target_sr if isinstance(base_target_sr, (int, float)) and base_target_sr > 0 else sr))
    if base_sr <= 0 or base_sr == sr:
        return {"sr_base": sr, "mono_base": mono, "L_base": L, "R_base": R}
    try:
        from scipy.signal import resample_poly
        g = math.gcd(sr, base_sr)
        up = base_sr // g
        down = sr // g
        if up <= 0 or down <= 0:
            return {"sr_base": sr, "mono_base": mono, "L_base": L, "R_base": R}
        Lb = resample_poly(L, up, down)
        Rb = resample_poly(R, up, down)
        Mb = resample_poly(mono, up, down)
        if Lb.size == 0 or Rb.size == 0 or Mb.size == 0:
            return {"sr_base": sr, "mono_base": mono, "L_base": L, "R_base": R}
        return {"sr_base": base_sr, "mono_base": Mb.astype(np.float64, copy=False), "L_base": Lb.astype(np.float64, copy=False), "R_base": Rb.astype(np.float64, copy=False)}
    except Exception:
        try:
            nL = int(round(L.size * (base_sr / float(sr))))
            nR = int(round(R.size * (base_sr / float(sr))))
            nM = int(round(mono.size * (base_sr / float(sr))))
            xL = np.linspace(0.0, L.size - 1.0, num=max(1, nL), endpoint=False, dtype=np.float64)
            xR = np.linspace(0.0, R.size - 1.0, num=max(1, nR), endpoint=False, dtype=np.float64)
            xM = np.linspace(0.0, mono.size - 1.0, num=max(1, nM), endpoint=False, dtype=np.float64)
            idxL = np.arange(L.size, dtype=np.float64)
            idxR = np.arange(R.size, dtype=np.float64)
            idxM = np.arange(mono.size, dtype=np.float64)
            Lb = np.interp(xL, idxL, L)
            Rb = np.interp(xR, idxR, R)
            Mb = np.interp(xM, idxM, mono)
            return {"sr_base": base_sr, "mono_base": Mb.astype(np.float64, copy=False), "L_base": Lb.astype(np.float64, copy=False), "R_base": Rb.astype(np.float64, copy=False)}
        except Exception:
            return {"sr_base": sr, "mono_base": mono, "L_base": L, "R_base": R}

def wait_for_user_exit():
    import os
    try:
        nonint = os.environ.get("AQC_NONINTERACTIVE", "").strip().lower() in ("1", "true", "on", "yes", "y")
    except Exception:
        nonint = False
    if nonint:
        return
    try:
        input("\nPremi Invio per uscire...")
    except (EOFError, KeyboardInterrupt):
        pass

def configure_threading(num_workers, per_worker_mb=None, mem_use_fraction=0.75):
    import os, platform, math
    max_cpu = os.cpu_count() or 1
    try:
        req = int(num_workers)
    except Exception:
        req = max_cpu
    try:
        import psutil
        avail = getattr(psutil.virtual_memory(), "available", None)
    except Exception:
        avail = None
    try:
        env_per_worker = os.environ.get("AQC_PER_WORKER_MB")
        if env_per_worker is not None:
            env_per_worker = float(env_per_worker)
            if not np.isfinite(env_per_worker) or env_per_worker <= 0:
                env_per_worker = None
    except Exception:
        env_per_worker = None
    pw_mb = None
    if isinstance(per_worker_mb, (int, float)) and per_worker_mb > 0:
        pw_mb = float(per_worker_mb)
    elif env_per_worker is not None:
        pw_mb = float(env_per_worker)
    else:
        pw_mb = 350.0
    try:
        env_frac = os.environ.get("AQC_MEM_USE_FRACTION")
        if env_frac is not None:
            mem_use_fraction = float(env_frac)
            if not (0.1 <= mem_use_fraction <= 0.95):
                mem_use_fraction = 0.75
    except Exception:
        mem_use_fraction = 0.75
    if avail is not None and isinstance(avail, (int, float)) and avail > 0:
        mem_cap = int((avail * float(mem_use_fraction)) // (pw_mb * 1024 * 1024))
        mem_cap = max(1, mem_cap)
    else:
        mem_cap = max(1, max_cpu // 2)
    hard_cap = max(1, min(mem_cap, max_cpu))
    workers = max(1, min(req, hard_cap, max_cpu))
    try:
        t_env = os.environ.get("AQC_BLAS_THREADS")
        threads_per_worker = int(t_env) if (t_env is not None and t_env.isdigit() and int(t_env) >= 1) else 1
    except Exception:
        threads_per_worker = 1
    env_threads = str(threads_per_worker)
    os.environ["MKL_NUM_THREADS"] = env_threads
    os.environ["NUMEXPR_NUM_THREADS"] = env_threads
    os.environ["OMP_NUM_THREADS"] = env_threads
    os.environ["OPENBLAS_NUM_THREADS"] = env_threads
    os.environ["VECLIB_MAXIMUM_THREADS"] = env_threads
    os.environ["ACCELERATE_NTHREADS"] = env_threads
    os.environ["NUMBA_NUM_THREADS"] = env_threads
    os.environ["MKL_DYNAMIC"] = "0"
    os.environ["OMP_DYNAMIC"] = "0"
    os.environ["OMP_PROC_BIND"] = "TRUE"
    os.environ["OMP_PLACES"] = "cores"
    os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
    os.environ["KMP_BLOCKTIME"] = "0"
    os.environ["FFMPEG_THREADS"] = "1"
    os.environ["NUMEXPR_MAX_THREADS"] = env_threads
    os.environ["SCIPY_FFT_THREADS"] = env_threads
    try:
        import psutil
        p = psutil.Process()
        logical = max_cpu
        try:
            physical = psutil.cpu_count(logical=False) or (logical // 2)
        except Exception:
            physical = logical // 2
        if physical and physical > 0 and logical >= physical:
            ht = max(1, logical // physical)
            groups = []
            for i in range(physical):
                grp = []
                for k in range(ht):
                    idx = i * ht + k
                    if idx < logical:
                        grp.append(idx)
                if grp:
                    groups.append(grp)
            order = []
            for k in range(ht):
                for i in range(len(groups)):
                    if k < len(groups[i]):
                        order.append(groups[i][k])
            if not order:
                order = list(range(logical))
        else:
            order = list(range(logical))
        unique_order = []
        seen = set()
        for idx in order:
            if idx not in seen:
                unique_order.append(idx)
                seen.add(idx)
        chosen = unique_order[:workers] if len(unique_order) >= workers else unique_order
        if not chosen:
            step = max(1, logical // workers) if workers > 0 else 1
            i = 0
            chosen = []
            while len(chosen) < workers and i < logical:
                chosen.append(i)
                i += step
            if not chosen:
                chosen = list(range(min(workers, logical)))
        try:
            p.cpu_affinity(chosen)
        except Exception:
            try:
                if platform.system().lower().startswith("linux"):
                    os.sched_setaffinity(0, set(chosen))
            except Exception:
                pass
        try:
            if platform.system().lower().startswith("win"):
                p.nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)
            else:
                p.nice(0)
        except Exception:
            pass
    except Exception:
        try:
            if platform.system().lower().startswith("linux"):
                os.sched_setaffinity(0, set(range(min(workers, max_cpu))))
        except Exception:
            pass
    return workers, threads_per_worker

def main_entry():
    import os, sys
    pol = get_quality_policy()
    md5_audio_groups = {}
    md5_representatives = {}
    temp_files_registry = {}
    all_temp_files_created = []
    temp_files_to_keep = set()
    file_paths_from_user = get_audio_files_from_directory()
    ffmpeg_available = check_ffmpeg()
    print("\n--- Fase 1: Calcolo MD5 Audio e Raggruppamento Duplicati ---")
    files_to_analyze_fully_info = []
    processed_md5s_for_selection = set()
    for original_path in file_paths_from_user:
        print(f"Pre-analisi MD5 per: {os.path.basename(original_path)}")
        md5_hash, path_to_use_for_analysis, temp_created_path = get_audio_md5(original_path, ffmpeg_available)
        if temp_created_path:
            temp_files_registry[original_path] = temp_created_path
            all_temp_files_created.append(temp_created_path)
        if md5_hash:
            if md5_hash not in md5_audio_groups:
                md5_audio_groups[md5_hash] = []
            md5_audio_groups[md5_hash].append(original_path)
            if md5_hash not in processed_md5s_for_selection:
                files_to_analyze_fully_info.append((path_to_use_for_analysis, original_path, bool(temp_created_path)))
                processed_md5s_for_selection.add(md5_hash)
                md5_representatives[md5_hash] = original_path
            else:
                rep = md5_representatives.get(md5_hash)
                rep_name = os.path.basename(rep) if rep else os.path.basename(md5_audio_groups[md5_hash][0])
                print(f"  -> Duplicato audio (MD5: {md5_hash[:8]}...) di '{rep_name}'. Sarà raggruppato.")
        else:
            print(f"  -> MD5 non calcolato/fallito per '{os.path.basename(original_path)}'. Sarà analizzato individualmente.")
            files_to_analyze_fully_info.append((path_to_use_for_analysis, original_path, bool(temp_created_path)))
    sorted_representative_results, winner_rep_result = perform_full_analysis_and_reporting(
        files_to_analyze_fully_info,
        md5_audio_groups,
        md5_representatives,
        temp_files_registry,
        all_temp_files_created,
        temp_files_to_keep,
        pol
    )
    return sorted_representative_results, winner_rep_result

if __name__ == "__main__":
    try:
        pol = get_quality_policy()
        md5_audio_groups = {}
        md5_representatives = {}
        temp_files_registry = {}
        all_temp_files_created = []
        temp_files_to_keep = set()

        file_paths_from_user = get_audio_files_from_directory()
        ffmpeg_available = check_ffmpeg()

        print("\n--- Fase 1: Calcolo MD5 Audio e Raggruppamento Duplicati ---")
        files_to_analyze_fully_info = []
        processed_md5s_for_selection = set()

        for original_path in file_paths_from_user:
            print(f"Pre-analisi MD5 per: {os.path.basename(original_path)}")
            md5_hash, path_to_use_for_analysis, temp_created_path = get_audio_md5(original_path, ffmpeg_available)
            if temp_created_path:
                temp_files_registry[original_path] = temp_created_path
                all_temp_files_created.append(temp_created_path)
            if md5_hash:
                if md5_hash not in md5_audio_groups:
                    md5_audio_groups[md5_hash] = []
                md5_audio_groups[md5_hash].append(original_path)
                if md5_hash not in processed_md5s_for_selection:
                    files_to_analyze_fully_info.append((path_to_use_for_analysis, original_path, bool(temp_created_path)))
                    processed_md5s_for_selection.add(md5_hash)
                    md5_representatives[md5_hash] = original_path
                else:
                    rep = md5_representatives.get(md5_hash)
                    rep_name = os.path.basename(rep) if rep else os.path.basename(md5_audio_groups[md5_hash][0])
                    print(f"  -> Duplicato audio (MD5: {md5_hash[:8]}...) di '{rep_name}'. Sarà raggruppato.")
            else:
                print(f"  -> MD5 non calcolato/fallito per '{os.path.basename(original_path)}'. Sarà analizzato individualmente.")
                files_to_analyze_fully_info.append((path_to_use_for_analysis, original_path, bool(temp_created_path)))

        print(f"\n--- Fase 2: Analisi Dettagliata ({len(files_to_analyze_fully_info)} file/gruppi unici) ---")
        representative_analysis_results = []

        num_workers_requested = prompt_num_workers()
        workers, threads_per_worker = configure_threading(num_workers_requested)
        max_cpu = os.cpu_count() or 1

        if workers > 1 and len(files_to_analyze_fully_info) > 1:
            try:
                import concurrent.futures
                print(f"Avvio analisi parallela con {workers} worker, {threads_per_worker} thread/worker (CPU: {max_cpu})...")
                with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
                    future_to_job = {}
                    for args in files_to_analyze_fully_info:
                        path_for_analysis, original_ref_path, is_temp_for_call = args
                        print(f"Avvio analisi: '{os.path.basename(path_for_analysis)}' (da originale: '{os.path.basename(original_ref_path)}')")
                        fut = ex.submit(_analyze_wrapper, args)
                        future_to_job[fut] = args
                    for fut in concurrent.futures.as_completed(future_to_job):
                        args = future_to_job[fut]
                        path_for_analysis, original_ref_path, is_temp_for_call = args
                        try:
                            res = fut.result()
                            if res:
                                representative_analysis_results.append(res)
                                print(f"Completata: '{os.path.basename(path_for_analysis)}' (Score: {res.get('score_float','N/A')}/100)")
                            else:
                                print(f"Completata: '{os.path.basename(path_for_analysis)}' (nessun risultato)")
                        except Exception as e:
                            print(f"Errore analisi '{os.path.basename(path_for_analysis)}': {type(e).__name__}: {e}")
            except Exception as e:
                print(f"Analisi parallela non disponibile/errore ({type(e).__name__}: {e}). Esecuzione in modalità sequenziale.")
                for path_for_analysis, original_ref_path, is_temp_for_call in files_to_analyze_fully_info:
                    print(f"Avvio analisi: '{os.path.basename(path_for_analysis)}' (da originale: '{os.path.basename(original_ref_path)}')")
                    analysis_result_dict = analyze_audio(path_for_analysis, is_temporary=is_temp_for_call)
                    if analysis_result_dict:
                        analysis_result_dict['_original_ref_path'] = original_ref_path
                        representative_analysis_results.append(analysis_result_dict)
                        print(f"Completata: '{os.path.basename(path_for_analysis)}' (Score: {analysis_result_dict.get('score_float','N/A')}/100)")
                    else:
                        print(f"Completata: '{os.path.basename(path_for_analysis)}' (nessun risultato)")
        else:
            print(f"Avvio analisi sequenziale con 1 worker (CPU: {max_cpu})...")
            for path_for_analysis, original_ref_path, is_temp_for_call in files_to_analyze_fully_info:
                print(f"Avvio analisi: '{os.path.basename(path_for_analysis)}' (da originale: '{os.path.basename(original_ref_path)}')")
                analysis_result_dict = analyze_audio(path_for_analysis, is_temporary=is_temp_for_call)
                if analysis_result_dict:
                    analysis_result_dict['_original_ref_path'] = original_ref_path
                    representative_analysis_results.append(analysis_result_dict)
                    print(f"Completata: '{os.path.basename(path_for_analysis)}' (Score: {analysis_result_dict.get('score_float','N/A')}/100)")
                else:
                    print(f"Completata: '{os.path.basename(path_for_analysis)}' (nessun risultato)")

        if not representative_analysis_results:
            print("\nNessun file valido analizzato con successo.")
            for temp_f_path in all_temp_files_created:
                if temp_f_path not in temp_files_to_keep and os.path.exists(temp_f_path):
                    try:
                        os.remove(temp_f_path)
                        print(f"Pulito temp: {os.path.basename(temp_f_path)}")
                    except OSError as e_clean:
                        print(f"Impossibile eliminare temp '{os.path.basename(temp_f_path)}': {e_clean}")
            sys.exit(0)

        technical_signature_groups = {}
        for res_dict in representative_analysis_results:
            try:
                tech_sig = make_signature(res_dict)
                if tech_sig not in technical_signature_groups:
                    technical_signature_groups[tech_sig] = []
                technical_signature_groups[tech_sig].append(res_dict)
            except Exception:
                continue

        for tech_group_list in technical_signature_groups.values():
            if not tech_group_list:
                continue
            try:
                shared_score_float = float(np.mean([g_res['score_float'] for g_res in tech_group_list]))
            except Exception:
                shared_score_float = None
            if shared_score_float is not None and np.isfinite(shared_score_float):
                for g_res in tech_group_list:
                    g_res['score_float'] = shared_score_float
                    g_res['score'] = int(round(shared_score_float))

        sorted_representative_results, winner_rep_result, ranking_mode = rank_results_hygiene_first(representative_analysis_results)

        print("\n--- Report Dettagliato ---")
        print("(✅ Ottimo, ⚠️ Attenzione, ❌ Critico)")
        print_console_report(sorted_representative_results, policy=pol)

        announce_winner(winner_rep_result, md5_audio_groups, temp_files_registry)
        move_winner_files(winner_rep_result, md5_audio_groups, temp_files_registry, temp_files_to_keep)

        log_choice = input("\nVuoi salvare un log tecnico dettagliato? [y/N]: ").strip().lower()
        if log_choice in ("y", "yes", "s", "si"):
            log_dir = os.path.dirname(__file__) if '__file__' in locals() else os.getcwd()
            excluded_clipping = [r for r in representative_analysis_results if r.get('clipping_class') in ('borderline', 'hard')]
            write_human_log(
                sorted_representative_results,
                path_dir=log_dir,
                excluded_clipping=excluded_clipping,
                md5_groups=md5_audio_groups,
                md5_representatives=md5_representatives,
                policy=pol
            )

        print("\n--- Pulizia file temporanei ---")
        deleted_final_count = 0
        kept_final_count = 0
        unique_temps_for_cleanup = set(all_temp_files_created)
        for temp_path_to_clean in unique_temps_for_cleanup:
            if temp_path_to_clean in temp_files_to_keep:
                print(f"Conservato (vincitore/duplicato spostato): {os.path.basename(temp_path_to_clean)}")
                kept_final_count += 1
            else:
                try:
                    if os.path.exists(temp_path_to_clean):
                        os.remove(temp_path_to_clean)
                        print(f"Eliminato: {os.path.basename(temp_path_to_clean)}")
                        deleted_final_count += 1
                except OSError as e_final_clean:
                    print(f"Impossibile eliminare temporaneo '{os.path.basename(temp_path_to_clean)}': {e_final_clean}")
        if not unique_temps_for_cleanup:
            print("Nessun file temporaneo creato o da pulire.")
        else:
            print(f"Pulizia completata: {deleted_final_count} file eliminati, {kept_final_count} conservati.")
        print("\nAnalisi completata.")
    finally:
        wait_for_user_exit()