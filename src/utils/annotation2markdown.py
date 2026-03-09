import json
import os

def json_to_markdown(file_path):
    """
    Legge un file JSON contenente i dati di un grafico e genera due rappresentazioni 
    tabellari testuali (formato originale e trasposto).

    La funzione estrae dinamicamente il titolo, le etichette degli assi 
    (x_axis_label, y_axis_label) e i punti dati. Implementa una logica specifica 
    per la gestione delle serie:
    - Se esiste una singola serie predefinita ("Main"), utilizza l'etichetta 
      indicata dal capo categorical_axis per nominare la riga/colonna dei valori
      se questa non esiste in base all'asse Y.
    - Se esistono serie multiple, utilizza i nomi effettivi delle serie (series_name) 
      per differenziare le righe/colonne.

    I dati vengono incrociati tramite una mappa (dizionario lookup) basata su 
    chiavi (serie, valore X). Questo garantisce che la tabella venga costruita 
    correttamente anche se i punti nel JSON non sono in ordine o se mancano 
    alcuni valori per determinate serie.

    Argomenti:
        file_path (str): Il percorso del file JSON da analizzare.

    Ritorna:
        tuple: Una tupla di due elementi (original_table, transposed_table).
            - original_table (str): Etichette X sulle colonne, serie/valori sulle righe.
            - transposed_table (str): Etichette X sulle righe, serie/valori sulle colonne.
            Se il file non esiste o non è valido, ritorna stringhe di errore.
    """
    file_path = str(file_path)
    if not os.path.exists(file_path):
        return f"Errore: Il file '{file_path}' non esiste.", ""
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        return "Errore: Il file non contiene un JSON valido.", ""
    except Exception as e:
        return f"Errore durante la lettura del file: {e}", ""
        
    title = data.get("chart_title", "Unknown Entity")
    raw_x_label = data.get("x_axis_label") or "X-Axis"
    raw_y_label = data.get("y_axis_label") or "Value"
    cat_axis = str(data.get("categorical_axis", "x")).strip().lower()
    swap_axes = (cat_axis == "y")
    if swap_axes:
        x_label = raw_y_label
        y_label = raw_x_label
    else:
        x_label = raw_x_label
        y_label = raw_y_label
    data_points = data.get("data_points", [])
    
    x_values = []
    series_names = []
    lookup = {}
    
    for point in data_points:
        if swap_axes:
            x = str(point.get("y_value", ""))
            y = str(point.get("x_value", ""))
        else:
            x = str(point.get("x_value", ""))
            y = str(point.get("y_value", ""))
            
        s = str(point.get("series_name", "Main"))
        
        if x not in x_values:
            x_values.append(x)
        if s not in series_names:
            series_names.append(s)
            
        lookup[(s, x)] = y

    is_single_main = len(series_names) == 1 or series_names[0] == "Main"
    
    lookup = {}
    for point in data_points:
        x = str(point.get("x_value", ""))
        s = str(point.get("series_name", "Main"))
        y = str(point.get("y_value", ""))
        lookup[(s, x)] = y

    # --- COSTRUZIONE TABELLA ORIGINALE ---
    if title == None : title=""
    title_row = f"TITLE | {title}"
    header_row = f" {x_label} | " + " | ".join(x_values)
    
    original_rows = [title_row, header_row]
    for s in series_names:
        row_label = y_label if is_single_main else s
        row_data = [f" {row_label}"]
        for x in x_values:
            row_data.append(lookup.get((s, x), "0")) 
        original_rows.append(" | ".join(row_data))
        
    original_table = "\n".join(original_rows)
    
    # --- COSTRUZIONE TABELLA TRASPOSTA ---
    transposed_headers = [f" {x_label}"]
    for s in series_names:
        transposed_headers.append(y_label if is_single_main else s)
        
    transposed_header_str = " | ".join(transposed_headers)
    transposed_rows_list = [title_row, transposed_header_str]
    
    for x in x_values:
        row_data = [f" {x}"]
        for s in series_names:
            row_data.append(lookup.get((s, x), "0"))
        transposed_rows_list.append(" | ".join(row_data))
        
    transposed_table = "\n".join(transposed_rows_list)
    
    return (original_table, transposed_table)

def json_to_markdown_scatter(file_path):
    file_path = str(file_path)
    if not os.path.exists(file_path):
        return f"Errore: Il file '{file_path}' non esiste.", ""
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        return "Errore: Il file non contiene un JSON valido.", ""
    except Exception as e:
        return f"Errore durante la lettura: {e}", ""

    title = data.get("chart_title", "")
    x_label = data.get("x_axis_label") or "X-Axis"
    y_label = data.get("y_axis_label") or "Y-Axis"
    data_points = data.get("data_points", [])

    if not data_points:
        return "Nessun dato presente nel grafico.", ""

    # Verifica se esiste solo la serie di default "Main"
    series_names = {str(point.get("series_name", "Main")) for point in data_points}
    is_single_main = len(series_names) == 1 

    # Costruzione della Tabella Long (Ideale per Scatter)
    rows = []
    if title:
        rows.append(f"TITLE | {title}")

    if is_single_main:
        rows.append(f"| {x_label} | {y_label} ")
        for point in data_points:
            x = str(point.get("x_value", ""))
            y = str(point.get("y_value", ""))
            rows.append(f" Main | {x} | {y} ")
    else:
        rows.append(f" Serie | {x_label} | {y_label} ")
        for point in data_points:
            s = str(point.get("series_name", "Main"))
            x = str(point.get("x_value", ""))
            y = str(point.get("y_value", ""))
            rows.append(f" {s} | {x} | {y} ")

    original_table = "\n".join(rows)

    return original_table, original_table


def json_to_markdown_errorpoint(file_path):
    file_path = str(file_path)
    if not os.path.exists(file_path):
        return f"Errore: Il file '{file_path}' non esiste.", ""
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        return "Errore: Il file non contiene un JSON valido.", ""
    except Exception as e:
        return f"Errore durante la lettura del file: {e}", ""
        
    title = data.get("chart_title", "Unknown Entity")
    raw_x_label = data.get("x_axis_label") or "X-Axis"
    raw_y_label = data.get("y_axis_label") or "Value"
    
    cat_axis = str(data.get("categorical_axis", "x")).strip().lower()
    swap_axes = (cat_axis == "y")
    
    if swap_axes:
        x_label = raw_y_label
        y_label = raw_x_label
    else:
        x_label = raw_x_label
        y_label = raw_y_label
        
    data_points = data.get("data_points", [])
    
    x_values = []
    series_names = []
    lookup = {}
    
    # Unico ciclo ottimizzato per l'estrazione e il controllo di swap_axes
    for point in data_points:
        if swap_axes:
            cat_val = str(point.get("y_value", ""))
            val_data = point.get("x_value", {})
        else:
            cat_val = str(point.get("x_value", ""))
            val_data = point.get("y_value", {})
            
        s = str(point.get("series_name", "Main"))
        
        if cat_val not in x_values:
            x_values.append(cat_val)
        if s not in series_names:
            series_names.append(s)
        
        # Estrazione delle tre metriche se il valore è un dizionario
        if isinstance(val_data, dict):
            v_min = str(val_data.get("min", "0"))
            v_med = str(val_data.get("median", "0"))
            v_max = str(val_data.get("max", "0"))
        else:
            # Fallback nel caso in cui il JSON presenti ancora valori scalari misti a dizionari
            v_min = v_med = v_max = str(val_data)
            
        lookup[(s, cat_val)] = {"min": v_min, "median": v_med, "max": v_max}

    is_single_main = len(series_names) == 1 or series_names[0] == "Main"
    
    # --- COSTRUZIONE TABELLA ORIGINALE ---
    if title == None : title=""
    title_row = f"TITLE | {title}"
    header_row = f" {x_label} | " + " | ".join(x_values)
    
    original_rows = [title_row, header_row]
    for s in series_names:
        base_label = y_label if is_single_main else s
        
        row_min = [f" min {base_label}"]
        row_med = [f" median {base_label}"]
        row_max = [f" max {base_label}"]
        
        for x in x_values:
            data_dict = lookup.get((s, x), {"min": "0", "median": "0", "max": "0"})
            row_min.append(data_dict["min"])
            row_med.append(data_dict["median"])
            row_max.append(data_dict["max"])
            
        original_rows.append(" | ".join(row_min))
        original_rows.append(" | ".join(row_med))
        original_rows.append(" | ".join(row_max))
        
    original_table = "\n".join(original_rows)
    
    # --- COSTRUZIONE TABELLA TRASPOSTA ---
    transposed_headers = [f" {x_label}"]
    for s in series_names:
        base_label = y_label if is_single_main else s
        # Aggiunta delle tre colonne per ogni elemento/serie
        transposed_headers.extend([f"min {base_label}", f"median {base_label}", f"max {base_label}"])
        
    transposed_header_str = " | ".join(transposed_headers)
    transposed_rows_list = [title_row, transposed_header_str]
    
    for x in x_values:
        row_data = [f" {x}"]
        for s in series_names:
            data_dict = lookup.get((s, x), {"min": "0", "median": "0", "max": "0"})
            row_data.extend([data_dict["min"], data_dict["median"], data_dict["max"]])
        transposed_rows_list.append(" | ".join(row_data))
        
    transposed_table = "\n".join(transposed_rows_list)
    
    return (original_table, transposed_table)


def json_to_markdown_box(file_path):
    file_path = str(file_path)
    if not os.path.exists(file_path):
        return f"Errore: Il file '{file_path}' non esiste.", ""
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        return "Errore: Il file non contiene un JSON valido.", ""
    except Exception as e:
        return f"Errore durante la lettura del file: {e}", ""
        
    title = data.get("chart_title", "Unknown Entity")
    raw_x_label = data.get("x_axis_label") or "X-Axis"
    raw_y_label = data.get("y_axis_label") or "Value"
    
    cat_axis = str(data.get("categorical_axis", "x")).strip().lower()
    swap_axes = (cat_axis == "y")
    
    if swap_axes:
        x_label = raw_y_label
        y_label = raw_x_label
    else:
        x_label = raw_x_label
        y_label = raw_y_label
        
    data_points = data.get("data_points", [])
    
    x_values = []
    series_names = []
    lookup = {}
    
    for point in data_points:
        if swap_axes:
            cat_val = str(point.get("y_value", ""))
            val_data = point.get("x_value", {})
        else:
            cat_val = str(point.get("x_value", ""))
            val_data = point.get("y_value", {})
            
        s = str(point.get("series_name", "Main"))
        
        if cat_val not in x_values:
            x_values.append(cat_val)
        if s not in series_names:
            series_names.append(s)
        
        if isinstance(val_data, dict):
            # Gestione esplicita dei valori None/null dal JSON
            v_min = str(val_data.get("min", "null")) if val_data.get("min") is not None else "null"
            v_q1  = str(val_data.get("q1", "null")) if val_data.get("q1") is not None else "null"
            v_med = str(val_data.get("median", "null")) if val_data.get("median") is not None else "null"
            v_q3  = str(val_data.get("q3", "null")) if val_data.get("q3") is not None else "null"
            v_max = str(val_data.get("max", "null")) if val_data.get("max") is not None else "null"
        else:
            # Fallback se il dato non è un dizionario strutturato
            v_min = v_q1 = v_med = v_q3 = v_max = str(val_data)
            
        lookup[(s, cat_val)] = {"min": v_min, "q1": v_q1, "median": v_med, "q3": v_q3, "max": v_max}

    is_single_main = len(series_names) == 1 or series_names[0] == "Main"
    
    # --- COSTRUZIONE TABELLA ORIGINALE ---
    if title == None : title=""
    title_row = f"TITLE | {title}"
    header_row = f" {x_label} | " + " | ".join(x_values)
    
    original_rows = [title_row, header_row]
    for s in series_names:
        base_label = y_label if is_single_main else s
        
        row_min = [f" min {base_label}"]
        row_q1  = [f" q1 {base_label}"]
        row_med = [f" median {base_label}"]
        row_q3  = [f" q3 {base_label}"]
        row_max = [f" max {base_label}"]
        
        for x in x_values:
            data_dict = lookup.get((s, x), {"min": "null", "q1": "null", "median": "null", "q3": "null", "max": "null"})
            row_min.append(data_dict["min"])
            row_q1.append(data_dict["q1"])
            row_med.append(data_dict["median"])
            row_q3.append(data_dict["q3"])
            row_max.append(data_dict["max"])
            
        original_rows.append(" | ".join(row_min))
        original_rows.append(" | ".join(row_q1))
        original_rows.append(" | ".join(row_med))
        original_rows.append(" | ".join(row_q3))
        original_rows.append(" | ".join(row_max))
        
    original_table = "\n".join(original_rows)
    
    # --- COSTRUZIONE TABELLA TRASPOSTA ---
    transposed_headers = [f" {x_label}"]
    for s in series_names:
        base_label = y_label if is_single_main else s
        transposed_headers.extend([
            f"min {base_label}", 
            f"q1 {base_label}", 
            f"median {base_label}", 
            f"q3 {base_label}", 
            f"max {base_label}"
        ])
        
    transposed_header_str = " | ".join(transposed_headers)
    transposed_rows_list = [title_row, transposed_header_str]
    
    for x in x_values:
        row_data = [f" {x}"]
        for s in series_names:
            data_dict = lookup.get((s, x), {"min": "null", "q1": "null", "median": "null", "q3": "null", "max": "null"})
            row_data.extend([
                data_dict["min"], 
                data_dict["q1"], 
                data_dict["median"], 
                data_dict["q3"], 
                data_dict["max"]
            ])
        transposed_rows_list.append(" | ".join(row_data))
        
    transposed_table = "\n".join(transposed_rows_list)
    
    return (original_table, transposed_table)

if __name__ == "__main__":
    #print(json_to_markdown("predictions/Gemini/pmc/bar/hard/PMC12681856_fig_5_crop_46.json")[0])
    #print(json_to_markdown("predictions/Gemini/pmc/bar/hard/PMC12681856_fig_5_crop_46.json")[1])

    res = json_to_markdown_errorpoint("predictions/Gemini/synthetic/errorpoint/chart_001_points_capsize_heavy.json")
    print("--- TABELLA ORIGINALE ---")
    print(res[0])
    print("\n--- TABELLA TRASPOSTA ---")
    print(res[1])

    res = json_to_markdown_box("predictions/Gemini/synthetic/box/chart_002_colored_per_series.json")
    print("--- TABELLA ORIGINALE ---")
    print(res[0])
    print("\n--- TABELLA TRASPOSTA ---")
    print(res[1])

    res = json_to_markdown_scatter("predictions/Gemini/synthetic/scatter/scatter_001_simple_scatter.json")
    print("--- TABELLA ORIGINALE ---")
    print(res[0])
    print("\n--- TABELLA TRASPOSTA ---")
    print(res[1])

    res = json_to_markdown_scatter("predictions/Gemini/synthetic/scatter/scatter_010_multi_class_scatter.json")
    print("--- TABELLA ORIGINALE ---")
    print(res[0])
    print("\n--- TABELLA TRASPOSTA ---")
    print(res[1])
