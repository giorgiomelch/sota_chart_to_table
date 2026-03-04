# Benchmark Chart-to-Table: DePlot, Gemini

Questo repository propone un framework di benchmark per il task **Chart-to-Table**, progettato per valutare e mettere a confronto le prestazioni di un modello task-oriented (**DePlot**) e di un LLM multimodale (**Gemini 2.5 Flash**).

---

## Struttura e Preparazione dei Dati

Per testare i modelli su dataset personalizzati (ad esempio provenienti PubMedCentral), devi organizzare i file rispettando una specifica gerarchia basata sul livello di difficoltà.

Distribuisci i file in questo modo:
* **Immagini (Input):** `data/pmc/<livello_difficoltà>/`
* **Tabelle (Ground Truth):** `data_groundtruth/pmc/<livello_difficoltà>/`

Le cartelle per i livelli di difficoltà devono essere nominate rigorosamente come:
* `easy`
* `medium`
* `hard`

---

## Pipeline di Esecuzione

Segui questi passaggi nell'ordine indicato per generare il set di dati, avviare le inferenze sui due modelli e raccogliere i risultati.

**1. Generazione del dataset sintetico**
Esegui lo script bash per generare i grafici sintetici e salvare automaticamente le rispettive tabelle di ground truth:
```bash
./chart_factory/_create_all_charts.sh
```

**2. Inferenza con Gemini**
Avvia lo script per generare le tabelle predette partendo dalle immagini nella cartella `data` utilizzando il modello `gemini-2.5-flash` (modello cambiabile tramite apposito parametro in ask_gemini.py):

```bash
python ./gemini/ask_gemini.py
```

**3. Inferenza con DePlot**
Avvia lo script dedicato a DePlot per ottenere le predizioni del modello sulle stesse immagini di input:

```bash
python ./deplot/inference.py
```

---

## Valutazione e Metriche

L'analisi dei risultati, sia qualitativa che quantitativa, avviene tramite Jupyter Notebook dedicati. La valutazione si basa sulla **Relative Mapping Similarity**.

* **Esplorazione visiva e metriche puntuali:**
  Usa `visualize_predictions.ipynb`. Questo notebook permette di visualizzare i grafici affiancati alle predizioni dei modelli e alla ground truth, calcolando istantaneamente **Precision**, **Recall** e **F1-Score** per ogni singolo grafico.

* **Report aggregato delle metriche:**
  Usa `metric/results_metric.ipynb`. Questo notebook calcola e mostra i risultati quantitativi, fornendo le medie delle metriche a livello globale, per classe di grafico e per livello di difficoltà.