for i in slurm*; do grep -A2 "EPOCH" "$i" | tail -n 2 | head -n 1 > auc_"$i" ;  done 
