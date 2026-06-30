cd "C:/Users/sergi/OneDrive/MSc Data Science/TFM/TFM"
count() { python -c "import pandas as pd,os; print(pd.read_csv('example/explore_noise_subset.csv').groupby(['optimizer','rules_pct','noise','rep']).ngroups if os.path.exists('example/explore_noise_subset.csv') else 0)" 2>/dev/null || echo 0; }
echo "NOISE-SUBSET START $(date '+%H:%M:%S')"
for i in $(seq 1 100); do
  c=$(count)
  echo "[try $i] hechos=$c/360 $(date '+%H:%M:%S')"
  [ "$c" -ge 360 ] && { echo "NOISE-SUBSET DONE"; break; }
  python -u explore_noise_subset.py > example/_noisesub_run.log 2>&1
done
echo "NOISE-SUBSET END $(date '+%H:%M:%S') $(count)/360"
