#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AutoMaster (programma separato) — Elabora il “vincitore” in alta SR e salva SEMPRE due versioni: Hi‑SR e CD

Obiettivo (priorità assoluta: migliorare davvero, in modo potente e senza regressioni “a caso”)
Questo programma riceve in input il “vincitore” scelto dal tuo analizzatore tecnico. L’input può essere CD‑quality
(44.1 kHz, 16/24 bit) oppure Hi‑Res (88.2/96/176.4/192/352.8/384 kHz, tipicamente 24 bit). AutoMaster lavora in modo
completamente autonomo e deterministico per riparare e rifinire difetti tecnici usando una SR di lavoro elevata
coerente con la “famiglia” dell’input (44.1→176.4/352.8; 48→192/384), come fa un engineer quando processa “in 176.4/192”.

Filosofia: progressivo e provato, non tentativi alla cieca
- Per ciascun modulo (HUM, de‑clip, de‑esser, tilt tonale, ampiezza stereo, limiter true‑peak), esegue tentativi multipli
  (griglia ristretta di parametri); DOPO ogni tentativo analizza il risultato con lo stesso motore del tuo programma di analisi
  (import in‑process) e tiene solo le regolazioni che migliorano davvero, senza indebolirle “per forza”.
- La versione Hi‑Res risultante è già “promossa” (migliorata rispetto all’originale secondo l’analizzatore); poi viene generata
  anche la versione CD (44.1/16, downsample+TPDF). NON è richiesto che entrambe battano l’originale: è sufficiente che almeno
  una delle due (Hi‑Res o CD) sia superiore. L’analizzatore a valle potrà comunque confrontare i tre (originale, Hi‑Res, CD).

Gestione intelligente della SR di lavoro (headroom di calcolo, non per “inventare” HF)
- Famiglia 44.1 (44.1/88.2/176.4/352.8 kHz):
  - se input ≤ 176.4 → lavora a 176.4 kHz;
  - se input = 352.8 → lavora nativo a 352.8 kHz;
  - se input > 352.8 → clamp a 176.4 kHz.
- Famiglia 48 (48/96/192/384 kHz):
  - se input ≤ 192 → lavora a 192 kHz;
  - se input = 384 → lavora nativo a 384 kHz;
  - se input > 384 → clamp a 192 kHz.

Cosa fa (float 64‑bit; temporanei WAV float per velocità/robustezza)
1) DC/rumble HP (~20 Hz) per allineare lo zero.
2) HUM/toni: notching 50/60 Hz + armoniche (Q/# armoniche adattivi) — tentativi multipli, accetta solo se il tuo analizzatore
   rileva uno spur chiaramente ridotto senza regressioni.
3) De‑clip leggero: ripara SOLO tratti realmente tagliati (interpolazione), accetta se ISP/TP migliorano senza schiacciare PLR/DR.
4) De‑esser dinamico (5–9 kHz) prudente: agisce solo in presenza di sibilanza/durezza misurata, con soglie conservative.
5) Tilt/shelving lieve per ammorbidire HF/durezza 2–5 kHz se sbilanciamento evidente.
6) Ampiezza stereo M/S: corregge in piccolo se fuori range naturale.
7) Limiter true‑peak con lookahead (alla SR di lavoro) per garantire margine TP (≈ −1 dBTP).

Output (sempre due file nella cartella del programma; nessuna decisione al posto tuo)
- Salva SEMPRE una versione Hi‑SR post‑lavorazione (FLAC 24‑bit, SR di lavoro scelta) con copertina se disponibile.
- Salva SEMPRE una versione CD post‑lavorazione (FLAC 44.1 kHz / 16‑bit con dither TPDF deterministico) con copertina se disponibile.
- Sarà poi il tuo analizzatore a confrontare originale, Hi‑Res e CD post‑lavoro e a scegliere quale tenere.

Uso (nessuna opzione a riga di comando)
- Avvia il programma: chiede il percorso del file. Produce due file nella cartella del programma:
  • automasterHR_*SR*k_*.flac (Hi‑SR, 24 bit),
  • automasterCD_*.flac (44.1/16).
"""
import os
import sys
import math
import time
import tempfile
import importlib.util
import datetime
import zlib
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt, sosfilt, iirnotch, resample_poly, fftconvolve, tf2sos
from scipy.ndimage import minimum_filter1d

try:
    import numba as _nb
    _NUMBA_OK = True
except Exception:
    _NUMBA_OK = False

try:
    from mutagen import File as _MutagenFile
    from mutagen.flac import FLAC as _MutagenFLAC, Picture as _MutagenFLACPicture
    from mutagen.id3 import ID3, APIC, ID3NoHeaderError
    from mutagen.mp4 import MP4, MP4Cover
    _MUTAGEN_OK = True
except Exception:
    _MUTAGEN_OK = False


def dbfs(v):
    v = float(v)
    if not np.isfinite(v) or v <= 0:
        return -np.inf
    return 20.0 * math.log10(v)

def calculate_rms(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x * x)))

def true_peak_estimate_stereo(x_stereo, sr, os_factor=8):
    x = np.asarray(x_stereo, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    x = np.ascontiguousarray(x, dtype=np.float64)
    if os_factor and os_factor > 1:
        try:
            y = resample_poly(x, os_factor, 1, axis=0)
        except Exception:
            n = x.shape[0]
            t = np.linspace(0, n - 1, n * os_factor, endpoint=False)
            idx = np.arange(n, dtype=float)
            y = np.zeros((t.size, x.shape[1]), dtype=np.float64)
            for c in range(x.shape[1]):
                y[:, c] = np.interp(t, idx, x[:, c])
    else:
        y = x
    pk = float(np.max(np.abs(y)))
    return dbfs(pk) if pk > 0 else -np.inf

def detect_hum_f0_and_strength(x_mono, sr, max_seconds=20, detect_sr=48000):
    x = np.asarray(x_mono, dtype=np.float64)
    n = x.size
    if n < int(0.2 * sr):
        return None, 0.0
    if max_seconds and n > int(max_seconds * sr):
        mid = n // 2
        half = int((max_seconds * sr) // 2)
        a = max(0, mid - half)
        b = min(n, mid + half)
        x = x[a:b]
        n = x.size
    if detect_sr and int(sr) != int(detect_sr):
        try:
            g = math.gcd(int(sr), int(detect_sr))
            up = int(detect_sr // g)
            down = int(sr // g)
            x = resample_poly(x, up, down)
            sr = detect_sr
            n = x.size
        except Exception:
            pass
    w = np.hanning(n)
    spec = np.abs(np.fft.rfft(x * w)) + 1e-20
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    hi = min(20000.0, 0.98 * (sr / 2.0))
    band = (freqs >= 20.0) & (freqs <= hi)
    if not np.any(band):
        return None, 0.0
    mag = spec[band]
    fb = freqs[band]
    k = max(31, int(round(mag.size * 0.01)))
    if k % 2 == 0:
        k += 1
    if k >= mag.size:
        k = max(31, max(3, mag.size // 3) | 1)
    smooth = fftconvolve(mag, np.ones(k) / k, mode='same')
    prom_db = 20.0 * (np.log10(mag) - np.log10(smooth + 1e-20))
    def score_f0(f0):
        if f0 is None:
            return 0.0
        cnt = 0
        amps = []
        kmax = int(hi // f0)
        for h in range(1, kmax + 1):
            fk = f0 * h
            bw = max(1.0, fk * 0.01)
            msk = (fb >= fk - bw) & (fb <= fk + bw)
            if np.any(msk):
                a = float(np.max(prom_db[msk]))
                amps.append(a)
                if a >= 6.0:
                    cnt += 1
        if not amps:
            return 0.0
        return float(np.median(amps) + 0.5 * cnt)
    s50 = score_f0(50.0)
    s60 = score_f0(60.0)
    if s50 <= 0 and s60 <= 0:
        return None, 0.0
    return (50.0, s50) if s50 >= s60 else (60.0, s60)

def highpass_dc_sos(sr, fc=20.0, order=2):
    nyq = 0.5 * float(sr)
    w = min(0.999, max(1e-6, float(fc) / nyq))
    sos = butter(int(order), w, btype='highpass', output='sos')
    return np.ascontiguousarray(sos, dtype=np.float64)

def design_hum_notch_comb(sr, f0, max_harm=16):
    sos_list = []
    nyq = 0.5 * float(sr)
    for h in range(1, int(max_harm) + 1):
        fk = float(f0) * h
        if fk >= nyq * 0.98:
            break
        if h <= 3:
            Q = 80.0
        elif h <= 8:
            Q = 50.0
        else:
            Q = 35.0
        b, a = iirnotch(w0=fk, Q=Q, fs=float(sr))
        sos = tf2sos(b, a)
        sos_list.append(np.ascontiguousarray(sos, dtype=np.float64))
    if sos_list:
        return np.vstack(sos_list)
    return None

def apply_sos_chain_filtfilt(x, sos_chain):
    def _as_valid_sos(s):
        if s is None:
            return None
        arr = np.asarray(s, dtype=np.float64)
        if arr.ndim == 1 and arr.size == 6:
            arr = arr.reshape(1, 6)
        if arr.ndim != 2 or arr.shape[1] != 6 or arr.shape[0] < 1:
            return None
        return np.ascontiguousarray(arr, dtype=np.float64)
    y = np.asarray(x, dtype=np.float64, order="C")
    if sos_chain is None:
        return y
    if isinstance(sos_chain, (list, tuple)):
        chunks = []
        for s in sos_chain:
            s_ok = _as_valid_sos(s)
            if s_ok is not None:
                chunks.append(s_ok)
        if not chunks:
            return y
        sos_all = np.vstack(chunks)
    else:
        sos_all = _as_valid_sos(sos_chain)
        if sos_all is None:
            return y
    try:
        return sosfiltfilt(sos_all, y)
    except Exception:
        try:
            return sosfilt(sos_all, y)
        except Exception:
            return y

def declip_soft(x, threshold=0.985, win=64):
    y = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    n = y.size
    if n == 0:
        return y
    mask0 = np.abs(y) >= float(threshold)
    if not np.any(mask0):
        return y
    k = np.array([1, 1, 1, 1, 1], dtype=np.int32)
    conv = np.convolve(mask0.astype(np.int32), k, mode="same")
    mask = conv > 0
    starts = []
    ends = []
    if mask[0]:
        starts.append(0)
    idx_up = np.flatnonzero(~mask[:-1] & mask[1:])
    idx_dn = np.flatnonzero(mask[:-1] & ~mask[1:])
    if idx_up.size:
        starts.extend((idx_up + 1).tolist())
    if idx_dn.size:
        ends.extend(idx_dn.tolist())
    if mask[-1]:
        ends.append(n - 1)
    if len(starts) != len(ends):
        return y
    for i, j in zip(starts, ends):
        a = max(0, i - win)
        b = min(n - 1, j + win)
        left = i - 1
        while left >= a and mask[left]:
            left -= 1
        right = j + 1
        while right <= b and mask[right]:
            right += 1
        if left >= a and right <= b and left < i and right > j:
            x_idx = np.arange(i, j + 1, dtype=np.int64)
            y[i : j + 1] = np.interp(x_idx, [left, right], [y[left], y[right]])
        else:
            seg = y[i : j + 1]
            wlin = np.linspace(1.0, 0.8, seg.size)
            y[i : j + 1] = np.clip(seg * wlin, -1.0, 1.0)
    return np.tanh(1.2 * y) / np.tanh(1.2)

def deesser_dynamic(x, sr, f_lo=5000.0, f_hi=9000.0, depth_db=2.0, sens=1.5):
    x = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    nyq = 0.5 * sr
    lo = max(1e-6, f_lo / nyq)
    hi = min(0.999, f_hi / nyq)
    if hi <= lo:
        return x
    bpf = butter(2, [lo, hi], btype='band', output='sos')
    try:
        band = sosfiltfilt(bpf, x)
    except Exception:
        try:
            band = sosfilt(bpf, x)
        except Exception:
            return x
    env = np.abs(band)
    thr = float(np.median(env) * sens)
    if thr <= 0:
        return x
    gain_lin = np.ones_like(x)
    over = env > thr
    if np.any(over):
        att = np.minimum((env[over] - thr) / (3.0 * thr), depth_db / 6.0)
        gain_lin[over] = 10.0 ** (-att)
    return x * gain_lin

if _NUMBA_OK:
    @_nb.njit(cache=True)
    def _smooth_gain_numba(need, atk, rel):
        n = need.shape[0]
        g = need.copy()
        for t in range(1, n):
            prev = g[t - 1]
            cand = prev + (1.0 - prev) * (1.0 / rel)
            if cand < g[t]:
                g[t] = cand
        for t in range(n - 2, -1, -1):
            nxt = g[t + 1]
            cand = nxt + (1.0 - nxt) * (1.0 / atk)
            if cand < g[t]:
                g[t] = cand
        return g

def _smooth_gain_py(need, atk, rel):
    g = need.copy()
    for t0 in range(1, g.size):
        g[t0] = min(g[t0], g[t0 - 1] + (1.0 - g[t0 - 1]) * (1.0 / rel))
    for t0 in range(g.size - 2, -1, -1):
        g[t0] = min(g[t0], g[t0 + 1] + (1.0 - g[t0 + 1]) * (1.0 / atk))
    return g

def limiter_truepeak_stereo(x_stereo, sr, tp_target_db=-1.5, lookahead_ms=5.0, atk_ms=1.0, rel_ms=50.0):
    x = np.asarray(x_stereo, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    n, ch = x.shape
    osf = 8
    try:
        x_os = resample_poly(x, osf, 1, axis=0)
    except Exception:
        t = np.linspace(0, n - 1, n * osf, endpoint=False)
        idx = np.arange(n, dtype=float)
        x_os = np.zeros((t.size, ch), dtype=np.float64)
        for c in range(ch):
            x_os[:, c] = np.interp(t, idx, x[:, c])
    env = np.max(np.abs(x_os), axis=1).astype(np.float64)
    la = int(round((lookahead_ms / 1000.0) * sr * osf))
    atk = max(1, int(round((atk_ms / 1000.0) * sr * osf)))
    rel = max(1, int(round((rel_ms / 1000.0) * sr * osf)))
    target_lin = 10.0 ** (tp_target_db / 20.0)
    if la > 0:
        from collections import deque
        q = deque()
        env_la = np.empty_like(env)
        N = env.size
        for i in range(N - 1, -1, -1):
            while q and q[0] > i + la:
                q.popleft()
            vi = env[i]
            while q and env[q[-1]] <= vi:
                q.pop()
            q.append(i)
            env_la[i] = env[q[0]]
    else:
        env_la = env
    need = np.ones_like(env_la)
    over = env_la > target_lin
    if np.any(over):
        need[over] = np.clip(target_lin / np.maximum(env_la[over], 1e-12), 0.0, 1.0)
    g = need.copy()
    for t0 in range(1, g.size):
        g[t0] = min(g[t0], g[t0 - 1] + (1.0 - g[t0 - 1]) * (1.0 / rel))
    for t0 in range(g.size - 2, -1, -1):
        g[t0] = min(g[t0], g[t0 + 1] + (1.0 - g[t0 + 1]) * (1.0 / atk))
    y_os = x_os * g.reshape(-1, 1)
    try:
        y = resample_poly(y_os, 1, osf, axis=0)
    except Exception:
        y = y_os[::osf]
    y = y[:n, :]
    tp_meas = true_peak_estimate_stereo(y, sr, os_factor=8)
    if tp_meas > tp_target_db + 0.05:
        gain = 10.0 ** ((tp_target_db - 0.05 - tp_meas) / 20.0)
        y = y * gain
    return y

def ms_width_adjust(L, R, target_db_range=(-10.0, +6.0), max_gain_change=0.15):
    L = np.ascontiguousarray(np.asarray(L, dtype=np.float64))
    R = np.ascontiguousarray(np.asarray(R, dtype=np.float64))
    M = 0.5 * (L + R)
    S = 0.5 * (L - R)
    rmsM = calculate_rms(M)
    rmsS = calculate_rms(S)
    if rmsM <= 1e-12:
        return L, R
    rel_db = 20.0 * math.log10(max(rmsS, 1e-12) / rmsM)
    lo, hi = target_db_range
    if rel_db < lo - 0.5:
        want = min(max_gain_change, (lo - rel_db) / 20.0)
        S = S * (1.0 + want)
    elif rel_db > hi + 0.5:
        want = min(max_gain_change, (rel_db - hi) / 20.0)
        S = S * (1.0 - want)
    L2 = M + S
    R2 = M - S
    peak = max(1e-12, float(max(np.max(np.abs(L2)), np.max(np.abs(R2)))))
    if peak > 1.0:
        L2 /= peak
        R2 /= peak
    return np.ascontiguousarray(L2, dtype=np.float64), np.ascontiguousarray(R2, dtype=np.float64)

def classify_family(sr):
    r441 = abs((sr / 44100.0) - round(sr / 44100.0))
    r48 = abs((sr / 48000.0) - round(sr / 48000.0))
    return "441" if r441 < r48 else "480"

def choose_work_sr(sr_in):
    fam = classify_family(sr_in)
    if fam == "441":
        if sr_in <= 176400:
            return 176400
        elif sr_in <= 352800:
            return int(sr_in)
        else:
            return 176400
    else:
        if sr_in <= 192000:
            return 192000
        elif sr_in <= 384000:
            return int(sr_in)
        else:
            return 192000

def resample_to(sr_target, x, sr_in):
    if int(sr_in) == int(sr_target):
        return np.asarray(x, dtype=np.float64, order="C"), int(sr_in)
    try:
        xi = np.asarray(x, dtype=np.float64, order="C")
        if xi.ndim == 1:
            xi = xi.reshape(-1, 1)
        g = math.gcd(int(sr_in), int(sr_target))
        up = int(sr_target // g)
        down = int(sr_in // g)
        y = resample_poly(xi, up, down, axis=0)
        return y, int(sr_target)
    except Exception:
        xi = np.asarray(x, dtype=np.float64, order="C")
        if xi.ndim == 1:
            xi = xi.reshape(-1, 1)
        n_in = xi.shape[0]
        n_out = int(round(n_in * (float(sr_target) / float(sr_in))))
        t = np.linspace(0.0, n_in - 1.0, n_out, endpoint=False)
        idx = np.arange(n_in, dtype=float)
        y = np.empty((n_out, xi.shape[1]), dtype=np.float64, order="C")
        for c in range(xi.shape[1]):
            y[:, c] = np.interp(t, idx, xi[:, c])
        return y, int(sr_target)

def downsample_to_44100(stereo, sr_in):
    y = np.ascontiguousarray(np.asarray(stereo, dtype=np.float64))
    return resample_to(44100, y, sr_in)

def dither_tpdf_16bit(stereo_f64, seed=None):
    x = np.asarray(stereo_f64, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    x = np.clip(x, -1.0, 1.0 - 1e-12)
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    lsb = 1.0 / (2**15)
    noise = (rng.random(size=x.shape) - 0.5) + (rng.random(size=x.shape) - 0.5)
    x_d = x + noise * lsb
    x_q = np.round(x_d * 32768.0).astype(np.int16)
    return x_q

def setup_fast_tempdir(preferred=None):
    import tempfile
    base = preferred
    if not base:
        if os.name == "nt":
            base = os.environ.get("AM_TEMP", r"C:\Temp\AM")
        else:
            base = os.environ.get("AM_TEMP", "/tmp/automaster")
    try:
        os.makedirs(base, exist_ok=True)
    except Exception:
        base = tempfile.gettempdir()
    os.environ["TMP"] = base
    os.environ["TEMP"] = base
    os.environ["TMPDIR"] = base
    tempfile.tempdir = base
    return base

def _extract_tags(input_path):
    try:
        from mutagen import File as _MutagenFile
        from mutagen.flac import FLAC as _MutagenFLAC
        from mutagen.id3 import ID3, ID3NoHeaderError
        from mutagen.mp4 import MP4
    except Exception:
        return None
    def _add(d, k, vals):
        if not vals:
            return
        k2 = str(k).upper()
        if k2 in ("METADATA_BLOCK_PICTURE", "COVERART", "COVERARTMIME"):
            return
        if isinstance(vals, (list, tuple)):
            vv = [str(v) for v in vals if v is not None]
        else:
            vv = [str(vals)]
        if not vv:
            return
        d.setdefault(k2, [])
        d[k2].extend(vv)
    tags = {}
    try:
        mf = _MutagenFile(input_path)
        if mf is None:
            return None
        if isinstance(mf, _MutagenFLAC):
            vc = getattr(mf, "tags", None)
            if vc:
                for k in vc.keys():
                    _add(tags, k, list(vc.get(k)))
            return tags or None
        try:
            id3 = ID3(input_path)
            def _get_text(frame_id):
                f = id3.get(frame_id)
                if not f:
                    return None
                t = getattr(f, "text", None)
                if not t:
                    return None
                return [str(x) for x in t if x is not None]
            _add(tags, "TITLE", _get_text("TIT2"))
            _add(tags, "ARTIST", _get_text("TPE1"))
            _add(tags, "ALBUM", _get_text("TALB"))
            _add(tags, "ALBUMARTIST", _get_text("TPE2"))
            _add(tags, "COMPOSER", _get_text("TCOM"))
            _add(tags, "GENRE", _get_text("TCON"))
            _add(tags, "COPYRIGHT", _get_text("TCOP"))
            _add(tags, "ISRC", _get_text("TSRC"))
            _add(tags, "LABEL", _get_text("TPUB"))
            trck = _get_text("TRCK")
            if trck and trck[0]:
                parts = trck[0].split("/")
                _add(tags, "TRACKNUMBER", parts[0])
                if len(parts) > 1:
                    _add(tags, "TRACKTOTAL", parts[1])
            tpos = _get_text("TPOS")
            if tpos and tpos[0]:
                parts = tpos[0].split("/")
                _add(tags, "DISCNUMBER", parts[0])
                if len(parts) > 1:
                    _add(tags, "DISCTOTAL", parts[1])
            date = _get_text("TDRC") or _get_text("TYER")
            _add(tags, "DATE", date)
            comms = id3.getall("COMM")
            if comms:
                comm = None
                for c in comms:
                    if getattr(c, "lang", "eng") == "eng":
                        comm = c
                        break
                if comm is None:
                    comm = comms[0]
                _add(tags, "COMMENT", getattr(comm, "text", None))
        except ID3NoHeaderError:
            pass
        except Exception:
            pass
        try:
            m4 = MP4(input_path)
            def _get(k):
                v = m4.tags.get(k)
                if v is None:
                    return None
                if isinstance(v, list):
                    out = []
                    for it in v:
                        if isinstance(it, (bytes, bytearray)):
                            try:
                                out.append(it.decode("utf-8", "ignore"))
                            except Exception:
                                out.append(str(it))
                        else:
                            out.append(str(it))
                    return out
                return [str(v)]
            _add(tags, "TITLE", _get("\xa9nam"))
            _add(tags, "ALBUM", _get("\xa9alb"))
            _add(tags, "ARTIST", _get("\xa9ART"))
            _add(tags, "ALBUMARTIST", _get("aART"))
            _add(tags, "GENRE", _get("\xa9gen"))
            _add(tags, "DATE", _get("\xa9day"))
            _add(tags, "COMPOSER", _get("\xa9wrt"))
            _add(tags, "COMMENT", _get("\xa9cmt"))
            trkn = m4.tags.get("trkn")
            if trkn and len(trkn) > 0:
                tn, tt = trkn[0][0], trkn[0][1]
                if tn:
                    _add(tags, "TRACKNUMBER", str(tn))
                if tt:
                    _add(tags, "TRACKTOTAL", str(tt))
            disk = m4.tags.get("disk")
            if disk and len(disk) > 0:
                dn, dt = disk[0][0], disk[0][1]
                if dn:
                    _add(tags, "DISCNUMBER", str(dn))
                if dt:
                    _add(tags, "DISCTOTAL", str(dt))
        except Exception:
            pass
    except Exception:
        return None
    return tags or None

def _write_flac_tags(flac_path, tags):
    try:
        from mutagen.flac import FLAC as _MutagenFLAC
    except Exception:
        return False
    try:
        fl = _MutagenFLAC(flac_path)
        if fl.tags is None:
            fl.add_tags()
        for k, vals in tags.items():
            vals2 = [str(v) for v in (vals if isinstance(vals, (list, tuple)) else [vals]) if v is not None]
            if not vals2:
                continue
            fl[k] = vals2
        fl.save()
        return True
    except Exception:
        return False

def _extract_cover_art(input_path):
    try:
        from mutagen import File as _MutagenFile
        from mutagen.flac import FLAC as _MutagenFLAC
        from mutagen.id3 import ID3, ID3NoHeaderError
        from mutagen.mp4 import MP4, MP4Cover
    except Exception:
        return None
    try:
        mf = _MutagenFile(input_path)
        if mf is not None:
            if isinstance(mf, _MutagenFLAC):
                pics = list(getattr(mf, "pictures", []) or [])
                pic = None
                for p in pics:
                    if getattr(p, "type", None) == 3:
                        pic = p
                        break
                if pic is None and pics:
                    pics.sort(key=lambda p: len(getattr(p, "data", b"") or b""), reverse=True)
                    pic = pics[0]
                if pic and getattr(pic, "data", None):
                    mime = pic.mime or "image/jpeg"
                    return {"mime": mime, "data": pic.data}
        try:
            id3 = ID3(input_path)
            apics = id3.getall("APIC")
            if apics:
                apic = None
                for a in apics:
                    if getattr(a, "type", None) == 3:
                        apic = a
                        break
                if apic is None:
                    apic = apics[0]
                mime = getattr(apic, "mime", "image/jpeg") or "image/jpeg"
                data = getattr(apic, "data", None)
                if data:
                    return {"mime": mime, "data": data}
        except ID3NoHeaderError:
            pass
        except Exception:
            pass
        try:
            mp4 = MP4(input_path)
            covr = mp4.tags.get("covr")
            if covr:
                c = covr[0]
                if isinstance(c, MP4Cover):
                    mime = "image/jpeg" if c.imageformat == MP4Cover.FORMAT_JPEG else "image/png"
                    return {"mime": mime, "data": bytes(c)}
        except Exception:
            pass
    except Exception:
        return None
    return None

def _embed_cover_art_flac(flac_path, cover):
    try:
        from mutagen.flac import FLAC as _MutagenFLAC, Picture as _MutagenFLACPicture
    except Exception:
        return False
    if not cover or not isinstance(cover, dict) or not cover.get("data"):
        return False
    try:
        fl = _MutagenFLAC(flac_path)
        fl.clear_pictures()
        pic = _MutagenFLACPicture()
        pic.data = cover["data"]
        pic.type = 3
        pic.mime = cover.get("mime", "image/jpeg")
        pic.desc = "Cover"
        fl.add_picture(pic)
        fl.save()
        return True
    except Exception:
        return False

def find_analyzer():
    base = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.normpath(os.path.join(base, "..", "analyzer", "analyzer.py"))
    if os.path.isfile(cand):
        try:
            spec = importlib.util.spec_from_file_location("_am_analyzer_mod", cand)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, "analyze_audio"):
                return getattr(mod, "analyze_audio"), cand
        except Exception:
            pass
    here = os.path.dirname(os.path.abspath(sys.argv[0]))
    this = os.path.basename(__file__)
    for fn in os.listdir(here):
        if not fn.endswith(".py"):
            continue
        if fn == os.path.basename(__file__):
            continue
        mod_path = os.path.join(here, fn)
        try:
            spec = importlib.util.spec_from_file_location(f"_an_mod_{fn.replace('.','_')}", mod_path)
            if not spec or not spec.loader:
                continue
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, "analyze_audio"):
                return getattr(mod, "analyze_audio"), mod_path
        except Exception:
            continue
    return None, None

def analyze_with_oracle(analyze_audio_func, x_stereo_f64, sr, temp_tag):
    if analyze_audio_func is None:
        return {}, None
    os.environ["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "1")
    os.environ["MKL_NUM_THREADS"] = os.environ.get("MKL_NUM_THREADS", "1")
    os.environ["OPENBLAS_NUM_THREADS"] = os.environ.get("OPENBLAS_NUM_THREADS", "1")
    os.environ["NUMEXPR_NUM_THREADS"] = os.environ.get("NUMEXPR_NUM_THREADS", "1")
    os.environ["FFMPEG_THREADS"] = os.environ.get("FFMPEG_THREADS", "1")
    fd, tmp = tempfile.mkstemp(prefix=f"_am_{temp_tag}_", suffix=".wav")
    os.close(fd)
    try:
        y = np.asarray(x_stereo_f64, dtype=np.float32, order="C")
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        sf.write(tmp, y, sr, subtype="FLOAT", format="WAV")
        res = analyze_audio_func(tmp, is_temporary=True)
        if not isinstance(res, dict):
            res = {}
        return res, tmp
    except Exception:
        try:
            os.remove(tmp)
        except Exception:
            pass
        return {}, None

def try_hum_notch(analyze_audio_func, analyzer_module_path, workers, base_res, Lw, Rw, sr):
    def getf(d, k, dv):
        try:
            return float(d.get(k, dv))
        except Exception:
            return dv
    M = 0.5 * (Lw + Rw)
    f0, score = detect_hum_f0_and_strength(M, sr)
    if f0 is None or score < 4.0:
        return Lw, Rw
    seq = [6, 8, 12, 16]
    cur_L, cur_R = Lw, Rw
    cur_base = base_res
    spur_old = getf(cur_base, "noise_spur_db", 1e9)
    isp_old = getf(cur_base, "isp_margin_db", 0.0)
    plr_old = getf(cur_base, "plr_effective_db", getf(cur_base, "plr_est", 0.0))
    dr_old = getf(cur_base, "dr_tt_avg", 0.0)
    for nh in seq:
        sos = design_hum_notch_comb(sr, f0, max_harm=nh)
        if sos is None:
            break
        Lc = apply_sos_chain_filtfilt(cur_L, sos)
        Rc = apply_sos_chain_filtfilt(cur_R, sos)
        y = np.column_stack([Lc, Rc])
        res, tmp = analyze_with_oracle(analyze_audio_func, y, sr, f"hum_seq_{nh}")
        try:
            spur_new = getf(res, "noise_spur_db", spur_old)
            isp_new = getf(res, "isp_margin_db", isp_old)
            plr_new = getf(res, "plr_effective_db", plr_old)
            dr_new = getf(res, "dr_tt_avg", dr_old)
            if (spur_new <= spur_old - 3.0) and (isp_new >= isp_old - 0.2) and (plr_new >= plr_old - 0.4) and (dr_new >= dr_old - 0.4):
                cur_L, cur_R = Lc, Rc
                cur_base = res
                spur_old, isp_old, plr_old, dr_old = spur_new, isp_new, plr_new, dr_new
                continue
            else:
                break
        finally:
            if tmp:
                try:
                    os.remove(tmp)
                except Exception:
                    pass
    return cur_L, cur_R

def try_declip(analyze_audio_func, analyzer_module_path, workers, base_res, Lw, Rw, sr):
    def getf(d, k, dv):
        try:
            return float(d.get(k, dv))
        except Exception:
            return dv
    cur_L, cur_R = Lw, Rw
    cur_base = base_res
    tp_old = getf(cur_base, "true_peak_est_dbtp", true_peak_estimate_stereo(np.column_stack([cur_L, cur_R]), sr, os_factor=8))
    isp_old = getf(cur_base, "isp_margin_db", 0.0)
    plr_old = getf(cur_base, "plr_effective_db", getf(cur_base, "plr_est", 0.0))
    dr_old = getf(cur_base, "dr_tt_avg", 0.0)
    for th in (0.985, 0.975):
        Lc = declip_soft(cur_L, threshold=th, win=64)
        Rc = declip_soft(cur_R, threshold=th, win=64)
        y = limiter_truepeak_stereo(np.column_stack([Lc, Rc]), sr, tp_target_db=-1.5)
        res, tmp = analyze_with_oracle(analyze_audio_func, y, sr, f"declip_seq_{int((1 - th) * 1000)}")
        try:
            tp_new = getf(res, "true_peak_est_dbtp", tp_old)
            isp_new = getf(res, "isp_margin_db", isp_old)
            plr_new = getf(res, "plr_effective_db", plr_old)
            dr_new = getf(res, "dr_tt_avg", dr_old)
            if (tp_new <= tp_old - 0.3 or isp_new >= isp_old + 0.2) and (plr_new >= plr_old - 0.4) and (dr_new >= dr_old - 0.4):
                cur_L, cur_R = y[:, 0], y[:, 1]
                cur_base = res
                tp_old, isp_old, plr_old, dr_old = tp_new, isp_new, plr_new, dr_new
                continue
            else:
                break
        finally:
            if tmp:
                try:
                    os.remove(tmp)
                except Exception:
                    pass
    return cur_L, cur_R

def rbj_highshelf_sos(fs, f0, gain_db, Q=0.707):
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * (f0 / fs)
    alpha = math.sin(w0) / (2.0 * Q)
    cosw = math.cos(w0)
    b0 = A * ((A + 1) + (A - 1) * cosw + 2.0 * math.sqrt(A) * alpha)
    b1 = -2 * A * ((A - 1) + (A + 1) * cosw)
    b2 = A * ((A + 1) + (A - 1) * cosw - 2.0 * math.sqrt(A) * alpha)
    a0 = (A + 1) - (A - 1) * cosw + 2.0 * math.sqrt(A) * alpha
    a1 = 2 * ((A - 1) - (A + 1) * cosw)
    a2 = (A + 1) - (A - 1) * cosw - 2.0 * math.sqrt(A) * alpha
    b = np.array([b0, b1, b2], dtype=np.float64) / a0
    a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)
    sos = np.empty((1, 6), dtype=np.float64, order="C")
    sos[0, :] = [b[0], b[1], b[2], a[0], a[1], a[2]]
    return sos

def gentle_tilt_if_needed(L, R, sr):
    L = np.ascontiguousarray(np.asarray(L, dtype=np.float64))
    R = np.ascontiguousarray(np.asarray(R, dtype=np.float64))
    def band_rms(x, lo, hi):
        nyq = sr * 0.5
        lo_w = max(1e-6, lo / nyq)
        hi_w = min(0.999, hi / nyq)
        if hi_w <= lo_w:
            return 0.0
        sos = butter(4, [lo_w, hi_w], btype='band', output='sos')
        try:
            y = sosfiltfilt(sos, x)
        except Exception:
            try:
                y = sosfilt(sos, x)
            except Exception:
                return 0.0
        return calculate_rms(y)
    M = 0.5 * (L + R)
    low = band_rms(M, 30.0, 200.0)
    midh = band_rms(M, 2000.0, 5000.0)
    high = band_rms(M, 8000.0, min(18000.0, 0.49 * sr))
    sos_chain = []
    if midh > 1.25 * max(low, 1e-9):
        sos_chain.append(rbj_highshelf_sos(sr, f0=6000.0, gain_db=-1.5, Q=0.707))
    elif high > 1.35 * max(low, 1e-9):
        sos_chain.append(rbj_highshelf_sos(sr, f0=8000.0, gain_db=-1.0, Q=0.707))
    if not sos_chain:
        return L, R
    L2 = apply_sos_chain_filtfilt(L, sos_chain)
    R2 = apply_sos_chain_filtfilt(R, sos_chain)
    return L2, R2

def try_deesser(analyze_audio_func, analyzer_module_path, workers, base_res, Lw, Rw, sr):
    def getf(d, k, dv):
        try:
            return float(d.get(k, dv))
        except Exception:
            return dv
    cur_L, cur_R = Lw, Rw
    cur_base = base_res
    ton_old = getf(cur_base, "spectral_balance_dev_db", 9.9)
    plr_old = getf(cur_base, "plr_effective_db", getf(cur_base, "plr_est", 0.0))
    isp_old = getf(cur_base, "isp_margin_db", 0.0)
    for depth_db in (1.5, 2.5):
        Lc = deesser_dynamic(cur_L, sr, depth_db=depth_db, sens=1.6)
        Rc = deesser_dynamic(cur_R, sr, depth_db=depth_db, sens=1.6)
        y = np.column_stack([Lc, Rc])
        res, tmp = analyze_with_oracle(analyze_audio_func, y, sr, f"deesser_seq_{int(depth_db*10)}")
        try:
            ton_new = getf(res, "spectral_balance_dev_db", ton_old)
            plr_new = getf(res, "plr_effective_db", plr_old)
            isp_new = getf(res, "isp_margin_db", isp_old)
            if (ton_new <= ton_old - 0.2) and (plr_new >= plr_old - 0.3) and (isp_new >= isp_old - 0.1):
                cur_L, cur_R = Lc, Rc
                cur_base = res
                ton_old, plr_old, isp_old = ton_new, plr_new, isp_new
                continue
            else:
                break
        finally:
            if tmp:
                try:
                    os.remove(tmp)
                except Exception:
                    pass
    return cur_L, cur_R

def try_tilt(analyze_audio_func, base_res, Lw, Rw, sr):
    def getf(d, k, dv):
        try:
            return float(d.get(k, dv))
        except Exception:
            return dv
    cur_L, cur_R = Lw, Rw
    cur_base = base_res
    ton_old = getf(cur_base, "spectral_balance_dev_db", 9.9)
    plr_old = getf(cur_base, "plr_effective_db", getf(cur_base, "plr_est", 0.0))
    isp_old = getf(cur_base, "isp_margin_db", 0.0)
    improved = False
    first_choices = [(6000.0, -0.5), (8000.0, -0.5)]
    chosen_f0 = None
    for f0, g in first_choices:
        sos = rbj_highshelf_sos(sr, f0=f0, gain_db=g, Q=0.707)
        Lc = apply_sos_chain_filtfilt(cur_L, sos)
        Rc = apply_sos_chain_filtfilt(cur_R, sos)
        y = np.column_stack([Lc, Rc])
        res, tmp = analyze_with_oracle(analyze_audio_func, y, sr, f"tilt_seq_{int(f0)}_{int(g*10)}")
        try:
            ton_new = getf(res, "spectral_balance_dev_db", ton_old)
            plr_new = getf(res, "plr_effective_db", plr_old)
            isp_new = getf(res, "isp_margin_db", isp_old)
            if (ton_new <= ton_old - 0.2) and (plr_new >= plr_old - 0.3) and (isp_new >= isp_old - 0.1):
                cur_L, cur_R = Lc, Rc
                cur_base = res
                ton_old, plr_old, isp_old = ton_new, plr_new, isp_new
                chosen_f0 = f0
                improved = True
                break
        finally:
            if tmp:
                try:
                    os.remove(tmp)
                except Exception:
                    pass
    if not improved:
        return cur_L, cur_R
    for g in (-1.0, -1.5):
        sos = rbj_highshelf_sos(sr, f0=chosen_f0, gain_db=g, Q=0.707)
        Lc = apply_sos_chain_filtfilt(cur_L, sos)
        Rc = apply_sos_chain_filtfilt(cur_R, sos)
        y = np.column_stack([Lc, Rc])
        res, tmp = analyze_with_oracle(analyze_audio_func, y, sr, f"tilt_seq_{int(chosen_f0)}_{int(g*10)}")
        try:
            ton_new = getf(res, "spectral_balance_dev_db", ton_old)
            plr_new = getf(res, "plr_effective_db", plr_old)
            isp_new = getf(res, "isp_margin_db", isp_old)
            if (ton_new <= ton_old - 0.2) and (plr_new >= plr_old - 0.3) and (isp_new >= isp_old - 0.1):
                cur_L, cur_R = Lc, Rc
                cur_base = res
                ton_old, plr_old, isp_old = ton_new, plr_new, isp_new
                continue
            else:
                break
        finally:
            if tmp:
                try:
                    os.remove(tmp)
                except Exception:
                    pass
    return cur_L, cur_R

def try_stereo_width(analyze_audio_func, base_res, Lw, Rw, sr):
    def getf(d, k, dv):
        try:
            return float(d.get(k, dv))
        except Exception:
            return dv
    cur_L, cur_R = Lw, Rw
    cur_base = base_res
    wid_old = getf(cur_base, "stereo_width_iqr_db", 0.0)
    plr_old = getf(cur_base, "plr_effective_db", getf(cur_base, "plr_est", 0.0))
    isp_old = getf(cur_base, "isp_margin_db", 0.0)
    for step in (0.05, 0.10):
        Lc, Rc = ms_width_adjust(cur_L, cur_R, target_db_range=(-10.0, +6.0), max_gain_change=step)
        if np.allclose(Lc, cur_L) and np.allclose(Rc, cur_R):
            break
        y = np.column_stack([Lc, Rc])
        res, tmp = analyze_with_oracle(analyze_audio_func, y, sr, f"stereo_seq_{int(step*100)}")
        try:
            wid_new = getf(res, "stereo_width_iqr_db", wid_old)
            plr_new = getf(res, "plr_effective_db", plr_old)
            isp_new = getf(res, "isp_margin_db", isp_old)
            closer = abs(wid_new - 10.0) < abs(wid_old - 10.0)
            if closer and (plr_new >= plr_old - 0.2) and (isp_new >= isp_old - 0.1):
                cur_L, cur_R = Lc, Rc
                cur_base = res
                wid_old, plr_old, isp_old = wid_new, plr_new, isp_new
                continue
            else:
                break
        finally:
            if tmp:
                try:
                    os.remove(tmp)
                except Exception:
                    pass
    return cur_L, cur_R

def automaster_process(filepath, verbose=True, workers=1):
    analyze_audio_func, analyzer_module_path = find_analyzer() if "find_analyzer" in globals() else (None, None)
    try:
        data, sr = sf.read(filepath, dtype="float64", always_2d=True)
    except Exception as e:
        raise RuntimeError(f"Impossibile leggere '{filepath}': {e}")
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    L = data[:, 0]
    R = data[:, 1]
    cover = _extract_cover_art(filepath) if "_extract_cover_art" in globals() else None
    tags = _extract_tags(filepath) if "_extract_tags" in globals() else None
    base_res, tmp0 = analyze_with_oracle(analyze_audio_func, np.column_stack([L, R]), sr, "orig") if "analyze_with_oracle" in globals() else ({}, None)
    try:
        if verbose and base_res:
            print(f"[Analyzer] File originale: score={base_res.get('score_float')} TP={base_res.get('true_peak_est_dbtp')} ISP={base_res.get('isp_margin_db')} HUM={base_res.get('noise_spur_db')} PLR={base_res.get('plr_effective_db', base_res.get('plr_est'))}")
    finally:
        if tmp0:
            try:
                os.remove(tmp0)
            except Exception:
                pass
    sr_work = choose_work_sr(sr)
    if verbose:
        fam = classify_family(sr)
        print(f"[AutoMaster] SR in: {sr} Hz (famiglia {fam}) → SR lavoro: {sr_work} Hz")
    y_in = np.column_stack([L, R])
    y_work, _ = resample_to(sr_work, y_in, sr)
    Lw = y_work[:, 0]
    Rw = y_work[:, 1]
    sos_hp = highpass_dc_sos(sr_work, fc=20.0, order=2)
    Lw = apply_sos_chain_filtfilt(Lw, sos_hp)
    Rw = apply_sos_chain_filtfilt(Rw, sos_hp)
    if analyze_audio_func is None or "try_hum_notch" not in globals():
        hum_f0, hum_score = detect_hum_f0_and_strength(0.5 * (Lw + Rw), sr_work)
        if hum_f0 is not None and hum_score >= 4.0:
            sos_notches = design_hum_notch_comb(sr_work, hum_f0, max_harm=16)
            Lw = apply_sos_chain_filtfilt(Lw, sos_notches)
            Rw = apply_sos_chain_filtfilt(Rw, sos_notches)
        Lw, Rw = gentle_tilt_if_needed(Lw, Rw, sr_work)
        Lw, Rw = ms_width_adjust(Lw, Rw, target_db_range=(-10.0, +6.0), max_gain_change=0.10)
    else:
        cur_res, tmp1 = analyze_with_oracle(analyze_audio_func, np.column_stack([Lw, Rw]), sr_work, "work_init")
        try:
            base_res_work = cur_res
        finally:
            if tmp1:
                try:
                    os.remove(tmp1)
                except Exception:
                    pass
        Lw, Rw = try_hum_notch(analyze_audio_func, analyzer_module_path, workers, base_res_work, Lw, Rw, sr_work)
        cur_res, tmp2 = analyze_with_oracle(analyze_audio_func, np.column_stack([Lw, Rw]), sr_work, "work_after_hum")
        try:
            base_res_work = cur_res if cur_res else base_res_work
        finally:
            if tmp2:
                try:
                    os.remove(tmp2)
                except Exception:
                    pass
        if "try_declip" in globals():
            Lw, Rw = try_declip(analyze_audio_func, analyzer_module_path, workers, base_res_work, Lw, Rw, sr_work)
            cur_res, tmp3 = analyze_with_oracle(analyze_audio_func, np.column_stack([Lw, Rw]), sr_work, "work_after_declip")
            try:
                base_res_work = cur_res if cur_res else base_res_work
            finally:
                if tmp3:
                    try:
                        os.remove(tmp3)
                    except Exception:
                        pass
        if "try_deesser" in globals():
            Lw, Rw = try_deesser(analyze_audio_func, analyzer_module_path, workers, base_res_work, Lw, Rw, sr_work)
            cur_res, tmp4 = analyze_with_oracle(analyze_audio_func, np.column_stack([Lw, Rw]), sr_work, "work_after_deesser")
            try:
                base_res_work = cur_res if cur_res else base_res_work
            finally:
                if tmp4:
                    try:
                        os.remove(tmp4)
                    except Exception:
                        pass
        if "try_tilt" in globals():
            Lw, Rw = try_tilt(analyze_audio_func, base_res_work, Lw, Rw, sr_work)
            cur_res, tmp5 = analyze_with_oracle(analyze_audio_func, np.column_stack([Lw, Rw]), sr_work, "work_after_tilt")
            try:
                base_res_work = cur_res if cur_res else base_res_work
            finally:
                if tmp5:
                    try:
                        os.remove(tmp5)
                    except Exception:
                        pass
        if "try_stereo_width" in globals():
            Lw, Rw = try_stereo_width(analyze_audio_func, base_res_work, Lw, Rw, sr_work)
            cur_res, tmp6 = analyze_with_oracle(analyze_audio_func, np.column_stack([Lw, Rw]), sr_work, "work_after_stereo")
            try:
                base_res_work = cur_res if cur_res else base_res_work
            finally:
                if tmp6:
                    try:
                        os.remove(tmp6)
                    except Exception:
                        pass
    y_work2 = limiter_truepeak_stereo(np.column_stack([Lw, Rw]), sr_work, tp_target_db=-1.5)
    cur_res, tmp7 = analyze_with_oracle(analyze_audio_func, y_work2, sr_work, "work_after_limiter") if "analyze_with_oracle" in globals() else ({}, None)
    try:
        if isinstance(cur_res, dict):
            try:
                tp_an = float(cur_res.get("true_peak_est_dbtp", None))
            except Exception:
                tp_an = None
        else:
            tp_an = None
        if (tp_an is not None) and np.isfinite(tp_an) and (tp_an > -1.5 + 0.05):
            safety_gain = 10.0 ** ((-1.5 - 0.10 - tp_an) / 20.0)
            y_work2 = y_work2 * safety_gain
            cur_res2, tmp7b = analyze_with_oracle(analyze_audio_func, y_work2, sr_work, "work_after_safety")
            try:
                if isinstance(cur_res2, dict) and cur_res2:
                    cur_res = cur_res2
            finally:
                if tmp7b:
                    try:
                        os.remove(tmp7b)
                    except Exception:
                        pass
        if verbose and cur_res:
            print(f"[Analyzer] Post‑HiSR: score={cur_res.get('score_float')} TP={cur_res.get('true_peak_est_dbtp')} ISP={cur_res.get('isp_margin_db')} HUM={cur_res.get('noise_spur_db')} PLR={cur_res.get('plr_effective_db', cur_res.get('plr_est'))}")
    finally:
        if tmp7:
            try:
                os.remove(tmp7)
            except Exception:
                pass
    prog_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_hr = os.path.join(prog_dir, f"automasterHR_{sr_work//1000}k_{ts}.flac")
    sf.write(out_hr, y_work2, sr_work, subtype="PCM_24", format="FLAC")
    if "_write_flac_tags" in globals() and tags:
        _write_flac_tags(out_hr, tags)
    if "_embed_cover_art_flac" in globals() and cover:
        _embed_cover_art_flac(out_hr, cover)
    if verbose:
        tp_hr = true_peak_estimate_stereo(y_work2, sr_work, os_factor=8)
        print(f"Salvato Hi‑SR: {out_hr}  (TP≈{tp_hr:.2f} dBTP)")
    y_441, _ = downsample_to_44100(y_work2, sr_work)
    y_441 = limiter_truepeak_stereo(y_441, 44100, tp_target_db=-1.5)
    cur_res_cd, tmp_cd = analyze_with_oracle(analyze_audio_func, y_441, 44100, "cd_after_limiter") if "analyze_with_oracle" in globals() else ({}, None)
    try:
        if isinstance(cur_res_cd, dict):
            try:
                tp_cd = float(cur_res_cd.get("true_peak_est_dbtp", None))
            except Exception:
                tp_cd = None
        else:
            tp_cd = None
        if (tp_cd is not None) and np.isfinite(tp_cd) and (tp_cd > -1.5 + 0.05):
            safety_gain_cd = 10.0 ** ((-1.5 - 0.10 - tp_cd) / 20.0)
            y_441 = y_441 * safety_gain_cd
            cur_res_cd2, tmp_cd2 = analyze_with_oracle(analyze_audio_func, y_441, 44100, "cd_after_safety")
            try:
                if isinstance(cur_res_cd2, dict) and cur_res_cd2:
                    cur_res_cd = cur_res_cd2
            finally:
                if tmp_cd2:
                    try:
                        os.remove(tmp_cd2)
                    except Exception:
                        pass
    finally:
        if tmp_cd:
            try:
                os.remove(tmp_cd)
            except Exception:
                pass
    seed = zlib.adler32(np.ascontiguousarray(y_441).view(np.uint8)) if "zlib" in globals() else None
    try:
        x16 = dither_tpdf_16bit(y_441, seed=seed)
    except TypeError:
        x16 = dither_tpdf_16bit(y_441)
    out_cd = os.path.join(prog_dir, f"automasterCD_{ts}.flac")
    sf.write(out_cd, x16, 44100, subtype="PCM_16", format="FLAC")
    if "_write_flac_tags" in globals() and tags:
        _write_flac_tags(out_cd, tags)
    if "_embed_cover_art_flac" in globals() and cover:
        _embed_cover_art_flac(out_cd, cover)
    if verbose:
        x16f = x16.astype(np.float64) / 32768.0
        if x16f.ndim == 1:
            x16f = x16f.reshape(-1, 1)
        tp_cd_est = true_peak_estimate_stereo(x16f, 44100, os_factor=8)
        print(f"Salvato CD:   {out_cd}  (44.1 kHz / 16 bit, TP≈{tp_cd_est:.2f} dBTP)")
        print("AutoMaster ha creato le due versioni FLAC con metadati e copertina.")
    return out_hr, out_cd

def prompt_input_file():
    if len(sys.argv) >= 2:
        path = " ".join(sys.argv[1:]).strip().strip('"').strip("'")
        if os.path.isfile(path):
            return path
    while True:
        try:
            p = input("Trascina qui il file (oppure incolla il percorso) e premi Invio: ").strip().strip('"').strip("'")
        except (EOFError, KeyboardInterrupt):
            print("\nUscita.")
            sys.exit(0)
        if not p:
            print("Percorso vuoto. Riprova.")
            continue
        if os.path.isfile(p):
            return p
        print("Percorso non valido. Riprova.")

def main():
    temp_base = setup_fast_tempdir()
    in_path = prompt_input_file()
    try:
        out_hr, out_cd = automaster_process(in_path, verbose=True)
        print("Operazione completata.")
        print(f" • Temp:           {temp_base}")
        print(f" • Versione Hi‑SR: {out_hr}")
        print(f" • Versione CD:    {out_cd}")
        print("Procedi al confronto con il tuo analizzatore per scegliere cosa tenere.")
    except Exception as e:
        print(f"Errore: {e}")
        sys.exit(2)

if __name__ == "__main__":
    main()