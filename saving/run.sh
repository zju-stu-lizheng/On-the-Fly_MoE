### profile the activation for each layer
# python activation.py
# ### profile the average silu(gate) per channel
# python average.py
# ### profile the threshold for each layer
# python threshold.py

python mystatistics.py
python lmevaluate.py  > 1.out
python lmevaluate.py --use_average > average.out
