# GRAN-NAS-Pipeline

The repository is derived from the original GRAN github repository: https://github.com/lrjconan/GRAN.

# GRAN

Efficient Graph Generation with Graph Recurrent Attention Networks(https://arxiv.org/abs/1910.00760) as described in the following NeurIPS 2019 paper:

```
@inproceedings{liao2019gran,
  title={Efficient Graph Generation with Graph Recurrent Attention Networks},
  author={Liao, Renjie and Li, Yujia and Song, Yang and Wang, Shenlong and Nash, Charlie and Hamilton, William L. and Duvenaud, David and Urtasun, Raquel and Zemel, Richard},
  booktitle={NeurIPS},
  year={2019}
}
```

## Visualization

### Generation of GRAN per step:
![](http://www.cs.toronto.edu/~rjliao/imgs/gran_model.gif)


### Overall generation process:
<img src="http://www.cs.toronto.edu/~rjliao/imgs/gran_generation.gif" height="400px" width="550px" />


## Dependencies
Python 3, PyTorch(1.2.0)

Other dependencies can be installed via

  ```pip install -r requirements.txt```

## Conda installation

In general,

  ```conda env create -f mackenzie.yml```

For RTX 3090

  ```conda env create -f gran_nas_rtx_3090.yml```


## Run Demos

### Train
* To run the training of experiment ```X``` where ```X``` is one of {```gran_nas_evaluation.yaml```, ```mcgran.yaml```}:

  ```python run_exp.py -c config/X -t```


**Note**:

* Please check the folder ```config``` for a full list of configuration yaml files.
* Most hyperparameters in the configuration yaml file are self-explanatory.

### Test

* After training, you can specify the ```test_model``` field of the configuration yaml file with the path of your best model snapshot, e.g.,

  ```test_model: exp/gran_grid/xxx/model_snapshot_best.pth```

* To run the test of experiments ```X``` where ```X``` is one of {```gran_nas_evaluation.yaml```, ```mcgran.yaml```}:

  ```python run_exp.py -c config/X -e```

### Search 

* To search neural network architectures ```X``` where ```X``` is one of {```gran_nas_higher_order_search.yaml```, ```targeted_search.yaml```}::

  ```python run_exp.py -c config/X -s```

**Note**:

* Please check the [evaluation](https://github.com/JiaxuanYou/graph-generation) to set up.

We also compute the orbit counts for each graph, represented as a high-dimensional data point. We then compute the MMD between the two sets of sampled points using ORCA (see http://www.biolab.si/supp/orca/orca.html) at eval/orca. One first needs to compile ORCA by

    ```g++ -O2 -std=c++11 -o orca orca.cpp```
    
in directory eval/orca. (the binary file already in repo works in Ubuntu).


## Sampled Graphs from MCGRAN

* Valid neural networks:
![](/samples/rq2_valid_nn.PNG)

* Invalid neural networks:
![](/samples/rq2_invalid_nn.PNG)


## Cite
Please cite our paper if you use this code in your research work.

## Questions/Bugs
Please submit a Github issue or contact sathish.corley@gmail.com if you have any questions or find any bugs.
