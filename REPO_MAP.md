# GenSec — mappa dell'albero

Documento di orientamento per una sessione che riceve questo zip senza
altro contesto. Non sostituisce i recap in `handoff/`, che restano la
fonte autorevole sullo stato.

## Layout

```
src/gensec/          libreria
  solver/            fiber solver, timeline, losses, posttension,
                     section_state, check, prestress_ledger
  materials/         leggi costitutive, reologia, provider normativi
  geometry/          mesh, zone, elementi
  output/            plot, report
tests/               25 file, pytest
examples/            19 YAML (3 con construction_history)
tools/               numeric_fingerprint.py, datum_fingerprint.py
patchers/            patcher idempotenti + _patchkit.py
handoff/             49 documenti di consegna numerati (0-… → 12_10)
docs/                sorgenti Sphinx (.rst + conf.py)
.fingerprints/       fotografie numeriche progressive
```

## Come far girare le cose

L'albero usa il layout `src/`. In un ambiente senza installazione
editabile serve `PYTHONPATH`:

```bash
PYTHONPATH=src python -m pytest tests -q -p no:cacheprovider -o addopts=""
```

`-o addopts=""` disattiva le opzioni di coverage dichiarate in
`pyproject.toml`, che richiedono `pytest-cov`.

`tests/test_api.py` e `tests/test_server.py` non collezionano senza le
dipendenze del server; sono fuori dal perimetro del solutore.

La suite intera richiede alcuni minuti. Perimetro precompressione:

```bash
PYTHONPATH=src python -m pytest tests/test_phase12_ledger.py \
  tests/test_phase5_c5.py tests/test_posttension.py \
  tests/test_phase12_p3_red.py -q -p no:cacheprovider -o addopts=""
```

Atteso: **112 passed, 2 xfailed**. I due `xfailed` sono F4/F5 e devono
restare rossi finché P3 non atterra.

## Strumenti di misura

```bash
python tools/numeric_fingerprint.py capture out.txt      # 1969 valori
python tools/datum_fingerprint.py   capture out.txt      #   95 valori
python tools/datum_fingerprint.py   compare a.txt b.txt
```

Le due fotografie hanno scopi opposti. Dopo un refactoring che non deve
cambiare meccanica, `numeric_fingerprint` deve dire **INERT**. Dopo P3,
`datum_fingerprint` deve dire **MOVED** sulle chiavi `residue.*`: lì
INERT sarebbe silenzio, non prova.

Riferimenti pre-P3 già catturati in `.fingerprints/`:
`06_after_ledger.txt` (numerica) e `06_datum_PRE_P3.txt` (datum).

## Patcher

```bash
python patchers/p12_ledger.py --dry-run
python patchers/p12_ledger.py
python patchers/p12_ledger.py --selfcheck
```

I patcher sono idempotenti e preservano i fine-riga per file. **L'albero
in questo zip ha già `p12_ledger` applicato**: rieseguirlo riporta
`ALREADY` su ogni edit e `0 file(s) modified`.

Attenzione: i fine-riga sono **misti** e la cosa è voluta.
`timeline.py`, `posttension.py`, `tools/` e `patchers/` sono LF;
`check.py`, `losses.py` e diversi test sono CRLF. Un patcher che
normalizzasse tutto produrrebbe un diff enorme e falso.

## Da dove ripartire

`handoff/12_10-GENSEC_LEDGER_RECAP.md` §10 elenca l'aperto in ordine.
In sintesi: la storia di Boltzmann non registra i carichi applicati fra
due intervalli (misurato, non riparato); poi le finestre di azione; poi
P3.

## Convenzioni

- Prosa in italiano, codice e docstring in inglese (numpydoc, LaTeX per
  Sphinx).
- Normative: Eurocodici e NTC 2018 come provider di default dietro hook
  di override, mai cablati.
- Una casa per un fatto. Le condizioni di gate si misurano, non si
  dichiarano. Una fixture deve esibire il difetto prima della
  correzione.
