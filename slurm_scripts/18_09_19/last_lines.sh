for i in slurm*; do head -2 "$i" | tail -1; grep -A2 "EPOCH" "$i" | tail -n 2;  done >> results.txt
