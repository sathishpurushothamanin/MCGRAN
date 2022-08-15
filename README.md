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


## Run Demos

### Train
* To run the training of experiment ```X``` where ```X``` is one of {```gran_grid```, ```gran_DD```, ```gran_DB```, ```gran_lobster```}:

  ```python run_exp.py -c config/gran_nas_evaluation.yaml.yaml```


**Note**:

* Please check the folder ```config``` for a full list of configuration yaml files.
* Most hyperparameters in the configuration yaml file are self-explanatory.

### Test

* After training, you can specify the ```test_model``` field of the configuration yaml file with the path of your best model snapshot, e.g.,

  ```test_model: exp/gran_grid/xxx/model_snapshot_best.pth```

* To run the test of experiments ```X```:

  ```python run_exp.py -c config/gran_nas_evaluation.yaml -t```

**Note**:

* Please check the [evaluation](https://github.com/JiaxuanYou/graph-generation) to set up.

We also compute the orbit counts for each graph, represented as a high-dimensional data point. We then compute the MMD between the two sets of sampled points using ORCA (see http://www.biolab.si/supp/orca/orca.html) at eval/orca. One first needs to compile ORCA by

    ```g++ -O2 -std=c++11 -o orca orca.cpp```
    
in directory eval/orca. (the binary file already in repo works in Ubuntu).

### Trained Models
* You could use our trained model for comparisons. Please make sure you are using the same split of the dataset. Running the following script will download the trained model:

	```./download_model.sh```

## Sampled Graphs from GRAN

* Proteins Graphs from Training Set:
![](http://www.cs.toronto.edu/~rjliao/imgs/protein_train.png)

* Proteins Graphs Sampled from GRAN:
![](http://www.cs.toronto.edu/~rjliao/imgs/protein_sample.png)

## Cite
Please cite our paper if you use this code in your research work.

## Questions/Bugs
Please submit a Github issue or contact sathish.corley@gmail.com if you have any questions or find any bugs.
