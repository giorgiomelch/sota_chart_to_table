#!/bin/bash

NUM_CHARTS=50

echo "Avvio generazione di $NUM_CHARTS grafici a barre..."
echo "$NUM_CHARTS" | python chart_factory/bar_chart_generator.py
echo "----------------------------------------"

echo "Avvio generazione di $NUM_CHARTS grafici a torta..."
echo "$NUM_CHARTS" | python chart_factory/pie_chart_generator.py
echo "----------------------------------------"

echo "Avvio generazione di $NUM_CHARTS grafici a linee..."
echo "$NUM_CHARTS" | python chart_factory/line_chart_generator.py
echo "----------------------------------------"

echo "Tutte le generazioni sono state completate."