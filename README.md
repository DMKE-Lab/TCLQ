# TCLQ #
## Installation ##

The dependencies can be installed via either conda or pip. TCLQ is compatible
with Python >= 3.7 and PyTorch >= 1.8.0.

### From Conda ###

```bash
conda install torchdrug pytorch cudatoolkit -c milagraph -c pytorch -c pyg
conda install easydict pyyaml -c conda-forge
```

### From Pip ###

```bash
pip install torchdrug torch
pip install easydict pyyaml
```

Note that PyTorch > = 1.8. 0, and the torch library needs to correspond to the torch-scatter, 
torch-spare, and torch-geometric versions. If the environment version does not correspond, 
environment configuration errors (such as OSError errors, and so on) will occur.

## Generate a dataset ##

This part of the code is all under the Dataset_generation folder, and there are data sets 
that have been generated and processed in the code. If you need to generate the data set 
of temporal complex logical query, you can generate the data set for temporal complex 
logical query according to ICEWS18 data set through create_queries. py file. You need to generate 
each query type separately, and specify which query type of data to generate by setting the 
value of `gen_id`. 

You just need to modify its default value, and then right-click to run.


## Usage ##

To run TCLQ, first of all, you need cd to the code file, that is, TCLQ file, 
and then you can choose a single GPU to run or multiple GUP to run at the same time. 
Please note that this code needs larger GPU memory.

The code for a single GPU to run is as follows:
```bash
python script/run.py -c config/icews18.yaml --gpus [0]
```

The code for multiple GPUs to run is as follows:

```bash
python -m torch.distributed.launch --nproc_per_node=4 script/run.py -c config/icews18.yaml --gpus [0,1,2,3]
```