cd "C:/Users/sergi/OneDrive/MSc Data Science/TFM/TFM"
echo "FULLGRID START $(date '+%Y-%m-%d %H:%M:%S')"
for i in $(seq 1 400); do
  echo "[fullgrid try $i] $(date '+%H:%M:%S')"
  python -u run_full_grid.py > example/_fullgrid.log 2>&1 && { echo "FULLGRID COMPLETE en try $i $(date '+%H:%M:%S')"; break; }
  echo "  (proceso murio en try $i, reintento; el resume retoma)"
done
echo "FULLGRID END $(date '+%Y-%m-%d %H:%M:%S')"
