from nasbench import api
from nasbench.lib import model_metrics_pb2
from nasbench.api import OutOfDomainError
from nasbench.lib import config
import os
# Use nasbench_full.tfrecord for full dataset (run download command above).
filepath = os.path.join('data/nas-101', 'nasbench_only108.tfrecord')
nasbench = api.NASBench(filepath, seed = 1234)
nasbench_config = config.build_config()