for file in  /home/kacper/doktorat/FAIS/Higgs_CP_state/prometheus_out/slurm*
do
  destination=$(head -2 "$file"  | tail -1 | sed -e 's/ //g')
  python plot_auc.py -i "$file" -o "$destination".eps -t "b" -v "r" -l "" -x "upper left"
done

