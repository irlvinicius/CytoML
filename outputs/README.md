# Saídas locais

Figuras e tabelas geradas pelos notebooks (`outputs/figures/`), **não versionadas**: são recriadas
ao reexecutar os notebooks, e os resultados já aparecem embutidos nos notebooks executados.

- `figures/classifier/` — `models/classifier/notebooks/03_eval.ipynb` (teste: IC 95%, métricas por
  classe, curvas ROC, matriz de erros, análise de falhas) e `07_cross_validation.ipynb` (validação
  cruzada).
- `figures/detector/` — `models/yolo/yolo26m/notebooks/03_eval.ipynb` (comparação com o baseline,
  curvas PR e F1 × confiança, validação cruzada, análise de falhas).
- `figures/clahe_calibration/` — `src/preprocessing/notebooks/02_filters_before_after.ipynb`.

Os pesos e artefatos de treino ficam no Hugging Face (ver "Pesos dos modelos" no README principal).
