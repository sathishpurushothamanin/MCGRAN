# GRAN-NAS-Pipeline

The repository is derived from the original GRAN github repository: https://github.com/lrjconan/GRAN.

## Dependencies
Python 3, PyTorch(1.2.0)

Other dependencies can be installed via

  ```pip install -r requirements.txt```

## Conda installation

In general,

  ```conda env create -f sur-mcgran-gpu.yml```

# install nasbench instructions
```
git clone https://github.com/google-research/nasbench
cd nasbench
pip install -e .
```

# Tensorflow nasbench updates for 2.0
```
tf.estimator.SessionRunHook instead of tf.train.SessionRunHook in nashbench\lib\training_time.py
tf.estimator.NanLossDuringTrainingError instead of tensorflow._api.v2.train.NanLossDuringTrainingError in nashbench\lib\evaluate.py
tf.compat.v1.python_io.tf_record_iterator instead of tf.python_io.tf_record_iterator in nasbench\api.py
```

# dataset download instructions
```
cd data
mkdir nas-101
cd nas-101
curl -O https://storage.googleapis.com/nasbench/nasbench_only108.tfrecord
```

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


## Sampled Graphs from MCGRAN

* Valid neural networks:
![](/samples/rq2_valid_nn.PNG)

* Invalid neural networks:
![](/samples/rq2_invalid_nn.PNG)


## Targeted Search
Our iterative algorithm along with MCGRAN continuously samples novel architecture with higher (targeted) test accuracy.

![](/samples/targeted_search.png)

## Cite
Please cite our paper if you use this code in your research work.

## Questions/Bugs
Please submit a Github issue or contact sathish.corley@gmail.com if you have any questions or find any bugs.
