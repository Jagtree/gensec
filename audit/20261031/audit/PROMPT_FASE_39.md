# GenSec — Fase 39: input fail‑loud (Fase 1 della roadmap post‑audit)

Sei lo sviluppatore principale di GenSec. Questa fase apre la roadmap
`ROADMAP_POST_AUDIT_01.md`, nata da `AUDIT_ENG_01.md` e
`AUDIT_ENG_01_ADDENDUM_A.md`. Le decisioni di Fase 0 sono ratificate e non
si riaprono: se una misura le contraddice, **stop‑and‑report**.

Obiettivo: nessun dato d'ingresso entra in un calcolo se il documento non
lo ha dichiarato. Nessun default normativo, nessuna chiave ignorata,
nessuna combinazione fisicamente impossibile.

## 0. Principi vincolanti

1. **Fisica prima, norma dopo.** Una clausola calibra; non giustifica uno stato fisicamente impossibile.
2. **Agnosticismo normativo.** Nessun parametro normativo di default, nessun fallback silenzioso. Ogni rifiuto nomina il campo, il valore visto e il motivo, e **non suggerisce un valore** (elencare l'insieme ammesso sì, sceglierne uno no).
3. **Misura prima di affermare.** Ogni numero del ledger è riprodotto in sessione. Exit code letto dal processo (`cmd > log; echo EXIT=$?`), mai attraverso una pipe. Handoff e memoria sono ipotesi.
4. **Codice non riconosciuto** = lavoro di una precedente istanza: si misura, non si cancella.
5. **Isolamento delle norme**: un modulo EC2 non nomina altre norme, nemmeno nei commenti.

## 1. Provenienza e baseline

1. Verifica l'archivio fornito: zip con bundle git **interno**; SHA256; HEAD del bundle = HEAD del tree; tree pulito. `archive_currency` oggi non legge gli zip: fai il controllo a mano con gli stessi tre criteri (bundle dietro, bundle avanti, tree sporco) e registralo. L'estensione del tool è in Fase 4, non qui.
2. Ambiente: `uv`, Python 3.13, extras `[fast,gui]` + dev, `PYTHONPATH=src`, `MPLBACKEND=Agg`.
3. Gate PRE: `getattr_inventory`, `element_class_inventory`, `selfcheck_inventory`, `material_assumption_inventory`, `completion_marker`. Fingerprint PRE: `numeric`, `datum`, `domain`. I due gate rossi noti (`element_class` drift +6/−3, `completion_marker` 5 digest obsoleti) vanno registrati con stato misurato, non sistemati qui.
4. Suite pytest completa PRE: conteggio pass/fail/skip.

## 2. Perimetro

Ogni voce si chiude solo quando la sonda indicata, eseguita **prima** (deve fallire o mostrare il difetto) e **dopo** (deve cambiare esito), è diventata un test in `tests/` con il caso YAML in `tests/data/audit01/`.

| Voce | Intervento | Sonda / caso |
|---|---|---|
| A‑1 | `_refuse_unknown_keys` su `demands`, membri di `envelopes` e componenti inline delle combinazioni; ogni componente (`N_kN`, `Mx_kNm`, `My_kNm`) letta indipendentemente. Da decidere e motivare nel ledger: componente assente = 0 dichiarato oppure rifiuto | `pA_sign.py` + `A1_sign_right_rebar.yaml` (`My_only_key` → My=150); `E3_degenerate.yaml` (`Mx_kNM` → rifiuto) |
| B‑2 | `_declared(..., admissible=...)` verifica l'appartenenza; `fben2` rifiuta `ls` sconosciuto (oggi tutto ciò che non è F/A dà γ_c=1) | `B2_ls_ULS.yaml` → rifiuto; `B2_ls_F.yaml` invariato |
| B‑3 | Via i default di `Concrete` (`n_parabola`, `eps_c2`, `eps_cu2`, `fct`, `Ec`) e di `Steel` (`fyk`, `gamma_s`, `Es`, `k_hardening`, `eps_su`, `works_in_compression`). Il percorso via bridge EC2 resta intatto | `pB_loader.py` + `B3_steel_nodecl.yaml` → rifiuto con nome campo; `B5_c70_explicit.yaml` carica |
| F‑4a | Nessun default per `works_in_compression` su `prestressing_steel_ec2` | `F1_pretensioned_uls.yaml` → rifiuto finché non dichiarato |
| I‑1 | La legge costitutiva viene validata contro il `code` dichiarato: una legge o un insieme di costanti di un'altra norma (es. parabola‑rettangolo EC2 sotto `aci318-19`) viene rifiutata. Il meccanismo sta nel registro delle norme (`materials/codes.py`), non nel loader | `pI_aci.py` + `I1_example_aci318_copy.yaml` → rifiuto |
| E‑1 | Stato API/GUI derivato da `verified` e `inside`; rimuovere ogni lettura di `eta_3D` | `pE_api.py` + `E1_api_status_no2D.yaml` → `fail` |
| B‑4 | Coerenza nei `__post_init__`: ε_su > ε_yd; k ≥ 1; f_ct < f_cm (dove f_cm è noto); ε_cu2 ≤ ε_c2 < 0; n > 0 | `B4_steel_impossible.yaml` → rifiuto, un messaggio per difetto |
| B‑5 | `service_concrete_from_ec2` e `fben2`: argomenti normativi (`ls`, `loadtype`, `TypeConc`, `NA`, `time`) keyword‑only e senza default | firma ispezionata; chiamanti aggiornati |
| C‑3 | Barra senza `x`: rifiuto se la sezione è analizzata in flessione biassiale o se `n_bars` > 1 | `C1_example_input_lumped.yaml` → rifiuto |
| D0‑5 | `docs/user_guide/sign_conventions.rst` allineata al codice (+My comprime il bordo destro; ε = ε0 + χx·y − χy·x; My = −Σσ·A·x) | `pA_sign.py`: la doc descrive ciò che la sonda misura |
| Esempi | Correggere `example_aci318.yaml` (chiavi delle domande, legge, coefficienti), `example_input.yaml` (`x` delle barre), `example_prestress.yaml` (`gamma_s`, `works_in_compression`) e ogni esempio che oggi dipende da un default | tutti gli esempi caricano; elenco nel ledger |

**Fuori perimetro**: α_cc/k_t (B‑1), doppio conteggio della precompressione (F‑1), `grout_selfconsistent` (F‑2), filtro di ammissibilità (D‑1), convessità (D‑2/D‑4), output e grafici, strumenti. Se li incontri: registra lo stato misurato, non riaprirli.

## 3. Metodo

- Per ogni voce: sonda PRE → modifica minima → sonda POST → test. Una voce alla volta, ognuna con il suo commit.
- Il messaggio di rifiuto si testa sul **contenuto** (campo, valore, motivo), non solo sul tipo di eccezione.
- Fingerprint: dopo ogni voce che tocca esempi, cattura e confronta. **Ogni** riga di diff va spiegata nel ledger con la voce che la causa. Un diff non spiegato è stop‑and‑report.
- Se rimuovere un default rompe un percorso interno (solver, timeline, SLS, API) che costruiva materiali senza dichiararli: non reintrodurre il default, rendi esplicita la costruzione in quel punto e registralo.
- Uno strumento che non vede un difetto è a sua volta un rilievo: `material_assumption_inventory` oggi passa con i default in baseline (AUD‑I‑2). Registra come cambia il suo output dopo B‑3, senza modificare il tool (Fase 4).

## 4. Gate di chiusura (falsificabili)

| Gate | Criterio |
|---|---|
| G1 | Ogni voce del §2 ha sonda PRE (difetto), sonda POST (esito invertito) e test nuovo verde |
| G2 | Suite pytest completa: nessun test prima verde diventa rosso senza motivazione scritta; conteggi PRE/POST nel ledger |
| G3 | Gate tool POST: nessun peggioramento rispetto al PRE (i due rossi noti restano registrati) |
| G4 | Fingerprint: diff PRE/POST interamente spiegato voce per voce; nessun diff su esempi non toccati |
| G5 | `grep` sui default rimossi: zero occorrenze nei container `Concrete`/`Steel` e nel loader per `works_in_compression` |
| G6 | Consegna: zip con bundle interno; HEAD del bundle = HEAD del tree; tree pulito; SHA256 dello zip e del ledger |

## 5. Deliverable

- `handoff/39_0-GENSEC_LEDGER_39.md`: cosa è stato fatto, misure PRE/POST, diff dei fingerprint spiegati, decisioni prese (es. A‑1: componente assente).
- `handoff/39_1-GENSEC_PHASE39_RECAP.md`: sintesi e stato della roadmap (voci chiuse, voci aperte, prossima fase = Fase 2 della roadmap).
- Zip con bundle interno, SHA256 in chat.

In chat: italiano, solo misure, decisioni, blocchi di codice.

**Istruzione permanente:** quando scrivo «Continua», riprendi esattamente dal punto in cui ti eri fermato, senza riassunti né ripetizioni.
