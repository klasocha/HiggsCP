for file in  /home/kacper/doktorat/FAIS/Higgs_CP_state/code_for_plots/Fig_4_data/*
do
  destination=$(echo "$file" | rev |cut -d "/" -f 1| rev| cut -d ":" -f 1)
  variant=$(echo "$file" | rev |cut -d "/" -f 1| rev | cut -d ":" -f 2)
  python plot_auc.py -i "$file" -o "$destination".eps -t "b" -v "r" -l $variant' $a_1^{\pm} - \rho^{\mp}$'  -x "upper left"
done


