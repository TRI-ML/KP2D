Neural Outlier Rejection for Self-Supervised Keypoint Learning
==============================================================

Overview
--------

**IO-Net:** A novel proxy task for the self-supervision of
keypoint description.

**KeyPointNet:** An improved keypoint-network
architecture that is especially amenable to robust keypoint detection
and description.

`**[Full paper]** <https://openreview.net/pdf?id=Skx82ySYPH>`__

Inference and evaluation
------------------------

Setting up your environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~

You need a machine with recent Nvidia drivers and a GPU. We recommend
using docker (see
`nvidia-docker2 <https://github.com/NVIDIA/nvidia-docker>`__
instructions) to have a reproducible environment. To setup your
environment, type in a terminal (only tested in Ubuntu 18.04):

.. code:: bash

    git clone https://github.com/TRI-ML/KP2D.git
    cd KP2D
    # if you want to use docker (recommended)
    make docker-build

We will list below all commands as if run directly inside our container.
To run any of the commands in a container, you can either start the
container in interactive mode with ``make docker-start`` to land in a
shell where you can type those commands, or you can do it in one step:

.. code:: bash

    make docker-run COMMAND="some-command"

Data
~~~~

Download HPatches data:

.. code:: bash

    cd /data/datasets/kp2d/
    wget http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz
    tar -xvf hpatches-sequences-release.tar.gz
    mv hpatches-release HPatches

Pre-trained models:
~~~~~~~~~~~~~~~~~~~

Download the pre-trained models from
`here <https://tri-ml-public.s3.amazonaws.com/github/kp2d/models/pretrained_models.tar.gz>`__
and place them in ``/data/models/kp2d/``

To evaluate any of the models, simply run:

.. code:: bash

    make docker-run COMMAND="python scripts/eval_keypoint_net.py --pretrained_model /data/models/kp2d/v4.ckpt --input /data/datasets/kp2d/HPatches/"

Evaluation for **(320, 240)**:

+-------+---------------+--------------+-------+-------+-------+--------+
| Model | Repeatability | Localization |   C1  |   C3  |   C5  | MScore |
+=======+===============+==============+=======+=======+=======+========+
|  V0*  |     0.644     |     1.087    | 0.459 | 0.816 | 0.888 | 0.518  | 
+-------+---------------+--------------+-------+-------+-------+--------+
|  V1*  |     0.678 	|     0.980    | 0.453 | 0.828 | 0.905 | 0.552  | 
+-------+---------------+--------------+-------+-------+-------+--------+
|  V2*  |     0.679 	|     0.942    | 0.534 | 0.860 | 0.914 | 0.573  | 
+-------+---------------+--------------+-------+-------+-------+--------+
|  V3   |     0.685 	|     0.885    | 0.602 | 0.836 | 0.886 | 0.520  |
+-------+---------------+--------------+-------+-------+-------+--------+
|  V4   |     0.687 	|     0.892    | 0.593 | 0.867 | 0.910 | 0.546  |
+-------+---------------+--------------+-------+-------+-------+--------+

Evaluation for **(640, 480)**:

+-------+---------------+--------------+-------+-------+-------+--------+
| Model | Repeatability | Localization |   C1  |   C3  |   C5  | MScore |
+=======+===============+==============+=======+=======+=======+========+
|  V0*  |     0.633     |     1.157    | 0.45  | 0.810 | 0.890 | 0.486  | 
+-------+---------------+--------------+-------+-------+-------+--------+
|  V1*  |     0.673     |     1.049    | 0.464 | 0.817 | 0.895 | 0.519  | 
+-------+---------------+--------------+-------+-------+-------+--------+
|  V2*  |     0.68      |     1.008    | 0.510 | 0.855 | 0.921 | 0.544  | 
+-------+---------------+--------------+-------+-------+-------+--------+
|  V3   |     0.682     |     0.972    | 0.550 | 0.812 | 0.883 | 0.486  |
+-------+---------------+--------------+-------+-------+-------+--------+
|  V4   |     0.684     |     0.972    | 0.566 | 0.84  | 0.900 | 0.511  |
+-------+---------------+--------------+-------+-------+-------+--------+

\* - these models were trained again after submission - the numbers
deviate slightly from the paper, however the same trends can be
observed.

License
-------

The source code is released under the `MIT license <LICENSE.md>`__.

Citation
--------

Please use the following citation when referencing our work:

::

    @inproceedings{
    tang2020neural,
    title={Neural Outlier Rejection for Self-Supervised Keypoint Learning},
    author={Jiexiong Tang and Hanme Kim and Vitor Guizilini and Sudeep Pillai and Rares Ambrus},
    booktitle={International Conference on Learning Representations},
    year={2020}
    }

.. |image0| image:: media/imgs/diagram_architecture.png
