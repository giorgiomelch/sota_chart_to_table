PROMPT_AreaLineBarHistogram = """
Sei un analista dati esperto nell'estrazione di informazioni visive. 
Il tuo compito è analizzare l'immagine del grafico fornita e ricostruire la tabella dati sottostante estraendo i valori esatti o, 
se non esplicitati, compiendo la stima più accurata possibile in base agli assi.

Restituisci l'output ESCLUSIVAMENTE in formato JSON valido, senza testo aggiuntivo o formattazione markdown esterna al JSON.
Il JSON deve rispettare rigorosamente la seguente struttura:

{
    "chart_title": "Titolo principale del grafico (null se assente)",
    "x_axis_label": "Etichetta dell'asse X (null se assente)",
    "y_axis_label": "Etichetta dell'asse Y (null se assente)",
    "categorical_axis": "Specifica quale asse rappresenta le categorie (variabile indipendente). Rispondi ESCLUSIVAMENTE con la stringa 'x' oppure 'y'. Se il grafico non ha un asse categorico (es. scatter plot con due assi numerici), restituisci null.",
    "data_points": [
        {
        "series_name": "Nome della serie (es. voce della legenda). Usa 'Main' se c'è una sola serie senza legenda.",
        "x_value": "Categoria o valore numerico sull'asse X.",
        "y_value": "Valore numerico sull'asse Y."
        }
    ],
    "comment": "Eventuale nota tecnica sull'estrazione o null se non necessaria."
}
"""
PROMPT_Scatter = """
Sei un analista dati esperto nell'estrazione di informazioni visive. 
Il tuo compito è analizzare l'immagine del grafico scatter fornita e ricostruire la tabella dati sottostante estraendo i valori esatti o, 
se non esplicitati, compiendo la stima più accurata possibile in base agli assi.

Restituisci l'output ESCLUSIVAMENTE in formato JSON valido, senza testo aggiuntivo o formattazione markdown esterna al JSON.
Il JSON deve rispettare rigorosamente la seguente struttura:

{
    "chart_title": "Titolo principale del grafico (null se assente)",
    "x_axis_label": "Etichetta dell'asse X (null se assente)",
    "y_axis_label": "Etichetta dell'asse Y (null se assente)",
    "data_points": [
        {
        "series_name": "Nome della serie (es. voce della legenda). Usa 'Main' se c'è una sola serie senza legenda.",
        "x_value": "Categoria o valore numerico sull'asse X.",
        "y_value": "Valore numerico sull'asse Y."
        }
    ],
    "comment": "Eventuale nota tecnica sull'estrazione o null se non necessaria."
}
"""
PROMPT_Radar = """
Sei un analista dati esperto nell'estrazione di informazioni visive. 
Il tuo compito è analizzare l'immagine del grafico fornita e ricostruire la tabella dati sottostante estraendo i valori esatti o, 
se non esplicitati, compiendo la stima più accurata possibile in base agli assi.

Restituisci l'output ESCLUSIVAMENTE in formato JSON valido, senza testo aggiuntivo o formattazione markdown esterna al JSON.
Il JSON deve rispettare rigorosamente la seguente struttura:

{
    "chart_title": "Titolo principale del grafico (null se assente)",
    "data_points": [
        {
            "series_name": "Nome della serie (es. voce della legenda). Usa 'Main' se c'è una sola serie.",
            "x_value": "Usa il nome del vertice/variabile.",
            "y_value": "Valore numerico corrispondente.",
        }
    ],
    "comment": "Eventuale nota tecnica sull'estrazione o null se non necessaria."
}
"""
PROMPT_Pie = """
Sei un analista dati esperto nell'estrazione di informazioni visive. 
Il tuo compito è analizzare l'immagine del grafico fornita e ricostruire la tabella dati sottostante estraendo i valori esatti o, 
se non esplicitati, compiendo la stima più accurata possibile in base agli assi.

Restituisci l'output ESCLUSIVAMENTE in formato JSON valido, senza testo aggiuntivo o formattazione markdown esterna al JSON.
Il JSON deve rispettare rigorosamente la seguente struttura:

{
    "chart_title": "Titolo principale del grafico (null se assente)",
    "data_points": [
        {
            "series_name": "Nome della serie (es. voce della legenda). Usa 'Main' se c'è una sola serie.",
            "x_value": "Usa il nome della fetta.",
            "y_value": "Valore numerico corrispondente (percentuale, conteggio o punteggio).",
        }
    ],
    "comment": "Eventuale nota tecnica sull'estrazione o null se non necessaria."
}
"""
PROMPT_Venn = """
Sei un analista dati esperto nell'estrazione di informazioni visive. 
Il tuo compito è analizzare l'immagine del grafico fornita e ricostruire la tabella dati sottostante estraendo i valori esatti o, 
se non esplicitati, compiendo la stima più accurata possibile in base agli assi.

Restituisci l'output ESCLUSIVAMENTE in formato JSON valido, senza testo aggiuntivo o formattazione markdown esterna al JSON.
Il JSON deve rispettare rigorosamente la seguente struttura:

{
    "chart_title": "Titolo principale del grafico (null se assente)",
    "data_points": [
        {
            "series_name": "Nome della serie (es. voce della legenda). Usa 'Main' se c'è una sola serie. Per il diagramma di Venn indica il nome dell'insieme o l'intersezione (es. 'A', 'B', 'A interseca B').",
            "x_value": "Usa il nome del vertice/variabile.",
            "y_value": "Valore numerico corrispondente.",
        }
    ],
    "comment": "Eventuale nota tecnica sull'estrazione o null se non necessaria."
}
"""

PROMPT_Box = """
Sei un analista dati esperto nell'estrazione di informazioni visive. 
Il tuo compito è analizzare l'immagine del grafico fornita e ricostruire la tabella dati sottostante estraendo i valori esatti o, 
se non esplicitati, compiendo la stima più accurata possibile in base agli assi.

Restituisci l'output ESCLUSIVAMENTE in formato JSON valido, senza testo aggiuntivo, preamboli o commenti se non nel campo apposito "comment".
Il JSON deve rispettare rigorosamente la seguente struttura standardizzata, che si adatta a diverse tipologie di grafici:

{
    "chart_title": "Titolo principale del grafico (null se assente)",
    "x_axis_label": "Etichetta dell'asse X (null se assente)",
    "y_axis_label": "Etichetta dell'asse Y (null se assente)",
    "categorical_axis": "Specifica quale asse rappresenta le categorie (variabile indipendente). Rispondi ESCLUSIVAMENTE con la stringa 'x' oppure 'y'. Se il grafico non ha un asse categorico (es. scatter plot con due assi numerici), restituisci null.",
    "data_points": [
        {
            "series_name": "Nome della serie o gruppo. Usa 'Main' se c'è una sola serie.",
            "x_value": "Categoria o valore numerico sull'asse X.",
            "y_value": {
                "min": "valore minimo",
                "q1": "primo quartile (se applicabile, altrimenti null)",
                "median": "mediana o valore centrale",
                "q3": "terzo quartile (se applicabile, altrimenti null)",
                "max": "valore massimo"
            }
        }
    ],
    "comment": "Eventuale nota tecnica sull'estrazione o null se non necessaria."
}
"""
PROMPT_Errorpoint = """
Sei un analista dati esperto nell'estrazione di informazioni visive. 
Il tuo compito è analizzare l'immagine del grafico fornita e ricostruire la tabella dati sottostante estraendo i valori esatti o, 
se non esplicitati, compiendo la stima più accurata possibile in base agli assi.

Restituisci l'output ESCLUSIVAMENTE in formato JSON valido, senza testo aggiuntivo, preamboli o commenti se non nel campo apposito "comment".
Il JSON deve rispettare rigorosamente la seguente struttura standardizzata, che si adatta a diverse tipologie di grafici:

{
    "chart_title": "Titolo principale del grafico (null se assente)",
    "x_axis_label": "Etichetta dell'asse X (null se assente)",
    "y_axis_label": "Etichetta dell'asse Y (null se assente)",
    "categorical_axis": "Specifica quale asse rappresenta le categorie (variabile indipendente). Rispondi ESCLUSIVAMENTE con la stringa 'x' oppure 'y'. Se il grafico non ha un asse categorico (es. scatter plot con due assi numerici), restituisci null.",
    "data_points": [
        {
            "series_name": "Nome della serie o gruppo. Usa 'Main' se c'è una sola serie.",
            "x_value": "Categoria o valore numerico sull'asse X.",
            "y_value": {
                "min": "valore minimo",
                "median": "mediana o valore centrale",
                "max": "valore massimo"
            }
        }
    ],
    "comment": "Eventuale nota tecnica sull'estrazione o null se non necessaria."
}
"""

PROMPT_Bubble = """
Sei un analista dati esperto nell'estrazione di informazioni visive. 
Il tuo compito è analizzare l'immagine del grafico fornita e ricostruire la tabella dati sottostante estraendo i valori esatti o, 
se non esplicitati, compiendo la stima più accurata possibile in base agli assi.

Restituisci l'output ESCLUSIVAMENTE in formato JSON valido, senza testo aggiuntivo, preamboli o commenti se non nel campo apposito "comment".
Il JSON deve rispettare rigorosamente la seguente struttura standardizzata, che si adatta a diverse tipologie di grafici:

{
    "chart_title": "Titolo principale del grafico (null se assente)",
    "x_axis_label": "Etichetta dell'asse X (null se assente)",
    "y_axis_label": "Etichetta dell'asse Y (null se assente)",
    "data_points": [
        {
            "series_name": "Nome della serie o voce della legenda. Usa 'Main' se c'è una sola serie.",
            "x_value": "Valore numerico o categoria sull'asse X.",
            "y_value": "Valore numerico sull'asse Y.",
            "z_value": "Valore numerico rappresentato dalla dimensione della bolla."
        }
    ],
    "comment": "Eventuale nota tecnica sull'estrazione o null se non necessaria."
}
"""

PROMPT_Violin = """
Sei un analista dati esperto. Il tuo compito è analizzare l'immagine del grafico a violino (Violin plot) fornito e ricostruire le informazioni sulla distribuzione.
Ricorda che la larghezza della figura rappresenta la densità dei dati: le sezioni più larghe indicano un'alta frequenza, mentre le sezioni strette indicano i valori estremi. I quartili non sono calcolabili a vista a meno che non ci sia un indicatore interno.

Restituisci l'output ESCLUSIVAMENTE in formato JSON valido, senza testo aggiuntivo o formattazione markdown esterna al JSON.
Il JSON deve rispettare rigorosamente la seguente struttura:

{
    "chart_title": "Titolo principale del grafico (null se assente)",
    "x_axis_label": "Etichetta dell'asse X (null se assente)",
    "y_axis_label": "Etichetta dell'asse Y (null se assente)",
    "data_points": [
        {
            "series_name": "Nome della categoria o gruppo sull'asse X.",
            "x_value": "Valore numerico o categoria sull'asse X.",
            "y_value": {
                "min": "Valore sull'asse Y in cui termina la coda inferiore del violino.",
                "max": "Valore sull'asse Y in cui termina la coda superiore del violino.",
                "mode": "Valore sull'asse Y in corrispondenza della sezione orizzontalmente più larga del violino (il picco di densità).",
                "median": "Valore della mediana. Estrailo SOLO se visivamente indicato (es. da un punto bianco o una linea orizzontale interna), altrimenti scrivi tassativamente null."
            },
            "z_value": null
        }
    ],
    "comment": "Specifica se la mediana era visibile o se hai potuto estrarre solo la moda e i limiti estremi."
}
"""