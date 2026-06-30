cd "C:/Users/sergi/OneDrive/MSc Data Science/TFM/TFM"
count_scale() { python -c "import pandas as pd,os; print(pd.read_csv('example/explore_scale_nhlv1.csv').groupby(['trigger','optimizer','rep']).ngroups if os.path.exists('example/explore_scale_nhlv1.csv') else 0)" 2>/dev/null || echo 0; }
echo "UNIFIED START $(date '+%Y-%m-%d %H:%M:%S')"
for i in $(seq 1 300); do
  echo "[grid try $i] $(date '+%H:%M:%S')"
  python -u run_full_grid.py > example/_fullgrid.log 2>&1 && { echo "GRID DONE try $i $(date '+%H:%M:%S')"; break; }
  echo "  (grid murio, reintento; resume retoma)"
done
for i in $(seq 1 200); do
  c=$(count_scale)
  echo "[scale try $i] hechas=$c/30 $(date '+%H:%M:%S')"
  [ "$c" -ge 30 ] && { echo "SCALE DONE en try $i"; break; }
  python -u explore_scale_nhlv1.py > example/_scale_run.log 2>&1
done
echo "UNIFIED END $(date '+%Y-%m-%d %H:%M:%S') scale=$(count_scale)/30"
