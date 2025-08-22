# AutoMaster Suite

Remaster tecnico, conservativo e misurato. AutoMaster elabora in alta SR e accetta solo correzioni che migliorano davvero secondo un Analyzer esterno. Produce due FLAC lossless con metadati e copertina: Hi‑Res 24‑bit e CD 44.1/16 con dither TPDF deterministico.

- Linguaggio: Python
- Piattaforme: Windows, macOS, Linux
- Licenza: MIT

## Caratteristiche

- Processing ad alta SR (176.4/192 kHz o nativo) con filtri a fase lineare/zero‑latency dove appropriato
- Moduli conservativi: DC/rumble HP, HUM 50/60 Hz + armoniche, declip leggero, de‑esser prudente, tilt HF lieve, correzione width M/S, limiter true‑peak
- Valutazione oggettiva: dopo ogni tentativo, l’Analyzer decide se tenere la modifica
- Output:
  - Hi‑Res FLAC 24‑bit (SR di lavoro)
  - CD FLAC 44.1 kHz / 16‑bit con dither TPDF deterministico
- Metadati e copertina: copiati dal file sorgente nei due FLAC finali
- Parallelismo opzionale per velocizzare i tentativi (HUM/decl ip/de‑esser)
- Determinismo: stessa versione → stessi output (anche il CD grazie al dither con seed)

## Requisiti

- Python 3.12+ (consigliato 3.13)
- Pacchetti: numpy, scipy, soundfile, mutagen, pyloudnorm (vedi requirements.txt)
- FFmpeg opzionale (alcuni analyzer lo usano)
- Per file audio nel repo: Git LFS (consigliato)

Installazione rapida:
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt
```

## Struttura del repository

```
automaster-suite/
├─ automaster/
│  └─ automaster.py
├─ analyzer/
│  └─ analyzer.py           # Deve esporre: analyze_audio(path, is_temporary=True) -> dict
├─ scripts/
│  ├─ run_automaster.bat
│  └─ run_automaster.sh
├─ tests/
│  └─ test_smoke.py
├─ docs/
│  └─ CHANGELOG.md
├─ .github/workflows/ci.yml
├─ .gitattributes
├─ .gitignore
├─ .pre-commit-config.yaml
├─ requirements.txt
├─ LICENSE
└─ README.md
```

## Come si usa

1) Posiziona l’Analyzer in analyzer/analyzer.py con questa firma:
```python
def analyze_audio(path: str, is_temporary: bool = False) -> dict:
    ...
```

2) Esegui AutoMaster:
```bash
python automaster/automaster.py
```

3) Segui i prompt:
- trascina/incolla il file sorgente
- scegli quanti core usare (consigliato: fino a 4 o pari ai tentativi disponibili)

Output nella cartella del programma:
- automasterHR_<SR>k_<timestamp>.flac  (Hi‑Res 24‑bit)
- automasterCD_<timestamp>.flac        (44.1 kHz / 16‑bit con dither TPDF)

Entrambi includono metadati e cover del sorgente (se disponibili).

## Dettagli tecnici

- Scelta SR di lavoro:
  - famiglia 44.1 → 176.4 kHz (o nativo 352.8)
  - famiglia 48 → 192 kHz (o nativo 384)
- Limiter true‑peak: target di default −1.2 dBTP (robusto sugli ISP)
- CD: downsample HQ + dither TPDF deterministico (seed derivato dal segnale)
- Determinismo:
  - Hi‑Res: bit‑identico a parità di versione
  - CD: identico tra run grazie al dither con seed
- Parallelismo: i tentativi (HUM/de‑clip/de‑esser) possono essere valutati in parallelo dall’Analyzer; i worker vengono limitati al numero di candidati per evitare overhead

## Consigli d’uso

- Per confronti affidabili: allinea i volumi (LUFS) e usa ABX
- Evita di processare di nuovo il CD 16‑bit: aggiungere dither una seconda volta non porta benefici
- Tieni l’Hi‑Res come “master di lavoro” e il CD per compatibilità

## Contribuire

- Apri una issue per bug/feature
- PR benvenute: esegui lint e test prima di inviare
```bash
pip install pre-commit pytest
pre-commit install
pytest -q
```

## Licenza

MIT — vedi LICENSE.

## Note legali

- Non includere nel repo materiale coperto da copyright senza autorizzazione
- Usa frammenti audio liberi o sintetici per i test (Git LFS consigliato)
