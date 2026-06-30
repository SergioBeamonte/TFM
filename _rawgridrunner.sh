cd "C:/Users/sergi/OneDrive/MSc Data Science/TFM/TFM"
echo "RAWGRID START $(date '+%H:%M:%S')"
for i in $(seq 1 100); do
  echo "[try $i] $(date '+%H:%M:%S')"
  python -u run_full_grid.py > example/_rawgrid.log 2>&1 && { echo "RAWGRID DONE try $i $(date '+%H:%M:%S')"; break; }
  echo "  (murio, reintento)"
done
echo "RAWGRID END $(date '+%H:%M:%S')"
