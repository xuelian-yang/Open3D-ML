# -*- coding: utf-8 -*-

"""
References
----------
    Open3D-ML » Getting started » Reading a dataset
        https://github.com/isl-org/Open3D-ML#reading-a-dataset
"""

import datetime
import logging
import open3d as o3d
import open3d.ml.torch as ml3d  # or open3d.ml.tf as ml3d
import os.path as osp
import sys
import time
from termcolor import colored


logger = logging.getLogger(__name__)


def setup_log():
    medium_format = (
        '[%(asctime)s] %(levelname)s : %(filename)s[%(lineno)d] %(funcName)s'
        ' >>> %(message)s'
    )

    dt_now = datetime.datetime.now()
    log_name = osp.basename(__file__).replace('.py', '.log')
    get_log_file = osp.join(osp.dirname(osp.abspath(__file__)), log_name)
    logging.basicConfig(
        filename=get_log_file,
        filemode='w',
        level=logging.INFO,
        format=medium_format
    )
    logging.info('@{} created at {}'.format(get_log_file, dt_now))
    print(colored('@{} created at {}'.format(get_log_file, dt_now), 'magenta'))


def main():
    path = '/home/sigma/Downloads/SemanticKITTI/data_odometry_velodyne/'
    logging.warning(f'load {path}')

    # construct a dataset by specifying dataset_path
    dataset = ml3d.datasets.SemanticKITTI(dataset_path=path)

    # get the 'all' split that combines training, validation and test set
    all_split = dataset.get_split('all')

    # print the attributes of the first datum
    print(all_split.get_attr(0))

    # print the shape of the first point cloud
    print(all_split.get_data(0)['point'].shape)

    # show the first 100 frames using the visualizer
    vis = ml3d.vis.Visualizer()
    vis.visualize_dataset(dataset, 'all', indices=range(100))


if __name__ == '__main__':
    print(colored(f'sys.version:        {sys.version}', 'yellow'))
    print(colored(f'open3d.__version__: {o3d.__version__}\n', 'yellow'))
    time_beg_test = time.time()
    setup_log()

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    main()
    time_end_test = time.time()
    logger.warning(f'tests/test_semantickitti.py elapsed {time_end_test - time_beg_test:.3f} seconds')
    print(colored(f'tests/test_semantickitti.py elapsed {time_end_test - time_beg_test:.3f} seconds', 'yellow'))
