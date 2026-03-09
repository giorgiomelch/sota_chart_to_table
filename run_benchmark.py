from src.models.deplot import DePlot_predict
from src.models.ask_gemini import gemini_predict
from src.evaluation.generate_reports import generate_reports
from src.evaluation.evaluate import run_evaluation

if __name__ == "__main__":
    # esegui predizioni per ciascun modello
    DePlot_predict()
    gemini_predict()
    # valuta e salva i grafici delle metriche
    run_evaluation()
    # scrivi i report
    generate_reports()