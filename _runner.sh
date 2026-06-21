cd "C:/Users/sergi/OneDrive/MSc Data Science/TFM/TFM"
count() { python -c "import pandas as pd;print(pd.read_csv(r'$1').groupby(['optimizer','fitness_type','rep']).ngroups)" 2>/dev/null || echo 0; }
echo "RUNNER START $(date '+%H:%M:%S')"
for i in $(seq 1 80); do
  c=$(count example/explore_incremental.csv)
  echo "  [inc try $i] hechas=$c/200"
  [ "$c" -ge 200 ] && break
  python -u explore_incremental.py > example/_inc_run.log 2>&1
done
for i in $(seq 1 80); do
  c=$(count example/explore_capacity.csv)
  echo "  [cap try $i] hechas=$c/200"
  [ "$c" -ge 200 ] && break
  python -u explore_capacity.py > example/_cap_run.log 2>&1
done
echo "RUNNER DONE $(date '+%H:%M:%S') inc=$(count example/explore_incremental.csv) cap=$(count example/explore_capacity.csv)"
