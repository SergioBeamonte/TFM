cd "C:/Users/sergi/OneDrive/MSc Data Science/TFM/TFM"
count_grid() { python -c "import pandas as pd,glob; print(sum(len(pd.read_csv(f)) for f in glob.glob('example/*/grid_full/grid_search_results_*.csv')))" 2>/dev/null || echo 0; }
count_scale() { python -c "import pandas as pd,os; print(pd.read_csv('example/explore_scale_nhlv1.csv').groupby(['trigger','optimizer','rep']).ngroups if os.path.exists('example/explore_scale_nhlv1.csv') else 0)" 2>/dev/null || echo 0; }
echo "CHAIN START $(date '+%Y-%m-%d %H:%M:%S')"
echo "CHAIN: esperando a que el grid llegue a 320 combos..."
while [ "$(count_grid)" -lt 320 ]; do sleep 180; done
echo "CHAIN: grid COMPLETO a $(date '+%H:%M:%S'). Lanzando explore_scale_nhlv1."
for i in $(seq 1 200); do
  c=$(count_scale)
  echo "[scale try $i] hechas=$c/30 $(date '+%H:%M:%S')"
  [ "$c" -ge 30 ] && { echo "SCALE COMPLETE en try $i"; break; }
  python -u explore_scale_nhlv1.py > example/_scale_run.log 2>&1
done
echo "CHAIN END $(date '+%Y-%m-%d %H:%M:%S') scale=$(count_scale)/30"
