export RHORHO_DATA=/home/janczarknurek/higgs/HiggsCP/examples/th-www.if.uj.edu.pl/erichter/forHiggsCP/HiggsCP_data/rhorho
# python main.py -e 5 -t nn_rhorho -i $RHORHO_DATA -f Variant-All --num_classes 15
parallel -j 6 python main.py -e 5 -t nn_rhorho -i $RHORHO_DATA -f Variant-All --num_classes ::: $(seq 7 50)
