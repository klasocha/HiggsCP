#!/bin/bash
for i in {2..50}
do
   python main.py --num_classes $i
done