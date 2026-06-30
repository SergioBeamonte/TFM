cd "C:/Users/sergi/OneDrive/MSc Data Science/TFM/TFM"
count() { python -c "import pandas as pd,os; print(pd.read_csv('example/explore_noise.csv').groupby(['optimizer','fitness_type','noise','rep']).ngroups if os.path.exists('example/explore_noise.csv') else 0)" 2>/dev/null || echo 0; }
echo "NOISE START $(date '+%H:%M:%S')"
for i in $(seq 1 100); do
  c=$(count)
  echo "[noise try $i] hechos=$c/360 $(date '+%H:%M:%S')"
  [ "$c" -ge 360 ] && { echo "NOISE DONE"; break; }
  python -u explore_noise.py > example/_noise_run.log 2>&1
done
echo "NOISE END $(date '+%H:%M:%S') $(count)/360"
