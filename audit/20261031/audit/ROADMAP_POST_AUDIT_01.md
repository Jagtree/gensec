# ROADMAP post AUDIT_ENG_01 — versione definitiva

Fonti: `AUDIT_ENG_01.md` (SHA256 `f06f4852…24fd`) e `AUDIT_ENG_01_ADDENDUM_A.md`.
Tree di riferimento: `claude_remerge` @ `02aca33`, Fase 38.

Ogni voce si chiude quando la **sonda dell'audit che l'ha rilevata cambia
esito**. Sonde e casi YAML sono nello zip dell'audit (`probes/`, `cases/`) e
diventano test di regressione nella fase che chiude la voce.

---

## Decisioni ratificate (Fase 0, chiusa)

| ID | Decisione |
|---|---|
| D0‑1 | A6 viene riscritto sull'invariante fisico: subito dopo l'iniezione il cavo rilegge σ_p,eff, con `demand_after` nullo e con 1 N; ε_init continuo. Il confronto timeline↔driver resta come controllo secondario. |
| D0‑2 | Dominio ammissibile. (1) Nel nucleo, fisica: ogni gruppo di elementi entro i limiti della propria legge. (2) Il vincolo di forma in compressione (pivot C) è fornito dal modulo della norma; con `custom` va dichiarato. Le leggi custom (Kent‑Park, Mander, tabulate) portano il proprio ε_cu. |
| D0‑3 | (b) subito: la non convessità viene **misurata**, non dedotta dal tipo di legge; output "dominio non convesso: sovrastima fino a X% a N=…", con i punti nella zona colmata non verificati. (c) rappresentazione non convessa: **entra nello scope della precompressione**, perché l'addendum A misura concavità reali dell'1,8–5,6% di \|M\|max su sezioni con cavi aderenti. |
| D0‑4 | Tutti gli indici si chiamano η e vengono riportati tutti, con doc e output che dicono a quale domanda risponde ciascuno. Il calcolo esterno di eta_norm va corretto (distanza dall'inviluppo, non dal piano). η_2D non calcolabile se N ∉ [N_Rd,min, N_Rd,max] (si riporta η_N = N/N_Rd) oppure se **η_2D > 10·η_ray**. "Non calcolabile" conta sempre come **non verificato**. |
| D0‑5 | Si corregge la doc della convenzione di segno My (vale il codice: +My comprime il bordo destro) e la convenzione si stampa in ogni output. |
| F‑1 | Rifiuto ristretto: `prestress_actions` con `ref` a un cavo aderente e attivo che porta già la sua pre‑deformazione. La ritesatura di cavi non aderenti/esterni resta un'azione (ΔP). |
| Consegna | Zip con il git bundle dentro; `archive_currency` va esteso agli zip. |

---

## Fase 1 — Input fail‑loud  *(prossima fase: Fase 39)*

| Voce | Gravità | Intervento | Chiusura |
|---|---|---|---|
| A‑1 | S1 | `_refuse_unknown_keys` su demands, membri di envelope e componenti inline; ogni componente letta indipendentemente | `A1` `My_only_key` → My=150; `E3` `Mx_kNM` → rifiuto; `example_aci318` → rifiuto finché non corretto |
| B‑2 | S1 | `_declared` verifica l'appartenenza a `admissible`; `fben2` rifiuta un `ls` sconosciuto | `B2_ls_ULS.yaml` → rifiuto |
| B‑3 | S1 | Nessun default in `Concrete` (n, ε_c2, ε_cu2, f_ct, E_c) e in `Steel` (tutti i campi) | `B3_steel_nodecl.yaml` → rifiuto con nome campo |
| F‑4a | S1 | Nessun default per `works_in_compression` su `prestressing_steel_ec2` | `F1_pretensioned_uls.yaml` → rifiuto finché non dichiarato |
| I‑1 | S1 | La legge costitutiva e le sue costanti vengono validate contro il `code` dichiarato; mescolanze tra norme rifiutate | `I1_example_aci318_copy.yaml` → rifiuto |
| E‑1 | S1 | Stato API/GUI da `verified` e `inside`; via `eta_3D` | `E1_api_status_no2D.yaml` → `fail` |
| B‑4 | S2 | Coerenza: ε_su > ε_yd, k ≥ 1, f_ct < f_cm, ε_cu2 ≤ ε_c2 < 0 | `B4_steel_impossible.yaml` → rifiuto |
| B‑5 | S2 | Argomenti normativi di `service_concrete_from_ec2` e `fben2` keyword‑only, senza default | firma senza `=` |
| C‑3 | S2 | Barre senza `x` rifiutate in analisi biassiale o con `n_bars` > 1 | `C1_example_input_lumped.yaml` → rifiuto |
| D0‑5 | S2 | Correzione doc `sign_conventions.rst` | testo coerente con `strain_field` |
| Esempi | — | Correggere `example_aci318`, `example_input`, `example_prestress` e tutti gli esempi che dipendevano da default | ogni esempio carica senza default |

E‑1 è stato anticipato dalla Fase 3: è S1, si risolve in una riga e non tocca i fingerprint.

**Fingerprint.** Cambiano solo gli esempi corretti. Ogni riga di diff va dichiarata nel ledger.

## Fase 2 — Fisica

| Voce | Gravità | Intervento | Chiusura |
|---|---|---|---|
| B‑1 | S1 | `fben2`: α_cc (valore dell'annesso) separato da k_t (solo se la resistenza è determinata a t > 28 d); struttura multi‑annesso, ogni annesso dichiarato con la sua fonte | `pB_concrete.py`: f_cd continuo a 28 d |
| F‑1 | S1 | Rifiuto ristretto (decisione F‑1) | `F1_pretensioned_uls.yaml` → rifiuto |
| F‑2 | S1 | Eliminare la scorciatoia `eps0=chi=0` in `grout_selfconsistent`; riscrivere A6 (D0‑1) | `pF_grout.py`: read‑back 1400,0000 a 0 N e a 1 N |
| D‑1 + F‑4b | S1 | Filtro di ammissibilità per gruppo prima dell'inviluppo (bulk per zona, barre, cavi a ε+ε_init); pivot C dal modulo norma; bordo di schiacciamento per zona (vedi residuo R‑6) | `pD_admiss.py`: 0 vertici inammissibili; `pF_hullvertex.py`: nessun vertice con calcestruzzo < ε_cu2 o cavo compresso |
| D‑2 (b) | S1 | Misura della concavità per fetta di N (metodo dell'addendum A) e segnalazione con entità | `D2_brittle_tension.yaml` → segnalato, punti nella zona colmata non verificati |

**Fingerprint.** D‑1 sposta i domini in forte compressione e quelli precompressi. F‑2 cambia `datum_fingerprint` sugli esempi stress → grout senza carico. Tutto va dichiarato.

## Fase 2bis — Dominio non convesso (opzione c), nello scope della precompressione

- Rappresentazione non convessa del dominio 3D a partire dagli stati ammissibili: inviluppo per fette di N e contorni Mx‑My non convessi.
- η_ray, η_2D, η_path e il test `inside` ridefiniti su quella rappresentazione; per eta_norm/beta va deciso come estendere distanze e raggio di Chebyshev al caso non convesso.
- Chiusura: concavità di AUD‑D‑4 (`example_external_tendon`, `example_staged_injection`, F1) e di D‑2 rappresentate senza sovrastima; nessun cambiamento sui casi convessi (C2) oltre la tolleranza dichiarata.

## Fase 3 — η e output come documento di calcolo

| Voce | Intervento |
|---|---|
| E‑2 | Regole D0‑4 (η_N, η_2D non calcolabile, soglia 10·η_ray); eta_norm esterno corretto; doc per ciascun η |
| E‑3 | Stato "non calcolabile: motivo", contato come non verificato; base dello stage fuori dominio → non calcolabile; JSON senza `Infinity` (`null` + `reason`) |
| G‑1 | `kind: characteristic` senza `fyk`/`fpk` → rifiuto, oppure check marcato "disabilitato per dichiarazione" |
| J‑1 | Colore = metrica, tratteggio = superamento, marker "n.c."/"N>N_Rd", nessun arrotondamento a 1.000 per valori > 1 |
| J‑2 | Contorno, colore e legenda per ogni zona bulk |
| J‑3 | Immagini sezione + asse neutro anche per gli stage delle combinazioni |
| J‑4 / K‑1 | Intestazione di tracciabilità in ogni summary: versione, HEAD, norma, leggi e costanti, risoluzione, convenzioni, elenco "non verificato da GenSec" |
| J‑5 | Intestazione `My_kNm` in `n_my_domain.csv` |
| C‑2 | Risoluzione dichiarata nell'input e stampata; via il `n_angles=36` fisso |

## Fase 4 — Strumenti ciechi e provenienza

- **I‑2** `material_assumption_inventory`: vede i default dei campi dataclass; un default in baseline non passa il gate.
- **I‑3** `StressLimits` separato da `ec2_stress_limits`; nessuna menzione di ACI nei moduli EC2.
- **A6** già riscritto in Fase 2; qui si estende il principio: ogni test di equivalenza ha anche un invariante fisico.
- **Provenienza:** consegna in zip con bundle interno; `archive_currency --gate` esteso agli zip; bundle rigenerato dalla storia corrente; `element_class_inventory` e `completion_marker` riportati a verde.

## Fase 5 — Residui, poi benchmark, poi AUDIT_ENG_02

Residui non misurati, in quest'ordine:

| ID | Residuo |
|---|---|
| R‑1 | Tensioni SLS in sezione fessurata contro analisi fessurata a mano |
| R‑2 | Perdite AAEM su un caso in forma chiusa |
| R‑3 | `PreexistingState`, i due orologi, retrofit come evento |
| R‑4 | Percorso `ntc18` (α_cc, coefficienti, citazioni) |
| R‑5 | Dichiarazioni di generalità sui materiali misti (acciaio, legno, CFRP, interfaccia) |
| R‑6 | Bordo di schiacciamento con più zone bulk: oggi `NMDiagram` usa solo il `bulk_material` primario (letto nel codice, non misurato); essenziale per Mander |

Poi i benchmark esterni (chiusura D3): esempi svolti EN 1992 (classi > C50 incluse), sezione precompressa da letteratura, software indipendente su sezione asimmetrica biassiale.

Poi **AUDIT_ENG_02**: stesse sonde dell'audit 01 e dell'addendum A. Gate: ogni S1 invertito; fingerprint cambiati solo dove dichiarato.

---

## Dipendenze

```
Fase 1 ──► Fase 2 ──► Fase 2bis ──► Fase 3 ──► Fase 4 ──► Fase 5
            │  D-1 (filtro) è prerequisito di 2bis (la rappresentazione
            │  non convessa si costruisce sugli stati ammissibili)
            └─ F-2/A6 indipendenti: possono partire subito dopo la Fase 1
```
