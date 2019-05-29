#!/bin/bash
#ERW
# how to make step, only 2,4,6,8,10,..
for i in {2..20}
do
   python main.py -e 5 --num_classes $i
done
