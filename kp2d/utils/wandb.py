# Copyright 2020 Toyota Research Institute.  All rights reserved.

import numbers
import os
import socket

import numpy as np
from PIL import Image

import wandb
from wandb.git_repo import GitRepo


class WandBLogger:
    """Dedicated logger class for WandB. Creates a group name based on the description.
    Training/validation sub-runs can be viewed within that group. Timestamp is required
    to create unique description name in the UI.
    Parameters
    ----------
    params: dict
        Dictionary containing all configuration parameters. If `groupd_id`
        is set in params, then the wandb_logger will associate the run with
        that group
    description: str
        Name for experiment in WandB UI.
    unique_id: str
        Unique id for the run
    project: str, default: wandb-debug
        Name of the project in the WandB UI
    entity: str, default: tri
        Team name or username. CMI uses `tri`
    mode: str, default: run
        Mode for WandB logging. Set to `dryrun` to not sync to cloud.
    job_type: str, default: debug
        Used for filtering UI in WandB
    """

    def __init__(self,
                 params,
                 description=None,
                 project='debug_wandb',
                 entity='tri',
                 mode='run',
                 job_type='train'):

        super().__init__()

        # Set up environment variables to work with wandb

        os.environ['WANDB_PROJECT'] = project
        os.environ['WANDB_ENTITY'] = entity
        os.environ['WANDB_MODE'] = mode

        pwd = os.getenv('PWD')
        params.update({'run_dir': pwd,
                       'job_type': job_type,
                       'description': description,
                       'name': ''})

        # Update wandb config params with git repo details
        try:
            self._repo = wandb.git_repo.GitRepo(root=pwd)
            params.update({'ip': socket.gethostbyname(socket.gethostname()),
                           'last_commit': self._repo.last_commit,
                           'branch': self._repo.branch})
        except:
            print('Failed to fetch git repo details.')
        self._wandb_logger = wandb.init(config=params, allow_val_change=True)
        wandb.run.save()

        self.description = description
        self.run_id = wandb.run.id
        self.run_name = wandb.run.name

        print('-'*50)
        print(self._wandb_logger, self.run_id, self.run_name, self.run_url)

    @property    
    def run_url(self) -> str:
        """Returns run URL."""
        return 'https://app.wandb.ai/{}/{}/runs/{}'.format(
            wandb.run.entity, wandb.run.project, wandb.run.id) if self._wandb_logger else None

    def log_values(self, key, values, now=True):
        """Add metrics to the logging buffer
        Parameters
        ----------
        key: str
            Name of the value to log.
        values: number or dictionary
            Value or values to be logged. If dictionary and
            key is provided, key will be applied as prefix
        now: bool, default: True
            Flag to log immediately to cloud
        """
        temp = self._parse_values_for_logging(key, values)
        self._wandb_logger.history.row.update(temp)
        if now:
            self.commit_log()

    def log_summary(self, key, values, now=True):
        """Add metrics to the summary statistics
        Parameters
        ----------
        key: str
            Name of the value to log
        values: number or dictionary
            Value or values to be logged. If dictionary and
            key is provided, key will be applied as prefix
        now: bool, default: True
            Flag to log immediately to cloud
        """
        temp = self._parse_values_for_logging(key, values)
        self._wandb_logger.summary.update(temp)
        if now:
            self.commit_log()

    def log_dictionary_subset(self, keys, dictionary, now=True):
        """"Log a subset of a dictionary based on keys
        Parameters
        ----------
        keys: list of str, default: None
            List of keys to log
        dictionary: dict
            Dictionary containing values to log
        now: bool, default: True
            Flag to log immediately to cloud
        """
        log_populated = False
        for _k in keys:
            if _k in dictionary:
                log_populated = True
                self.log_values(_k, dictionary[_k])

        if log_populated and now:
            self.commit_log()

    def commit_log(self):
        """Send buffer to wandb, and create a new row.
        Initialize the next point in history.
        """
        self._wandb_logger.history.add()

    def log_tensor_image(self, image, key, caption, size=(400, 400), now=True):
        """Log image to wandb
        Parameters
        ----------
        image: torch.FloatTensor (RGB float32 0-255 range) (C, H, W)
            Image to log
        key: str
            Key associated with image
        caption: str
            Caption for the image
        size: tuple (int, int), default: (400, 400)
            Size of image to upload
        now: bool, default: True
            Flag to log immediately to cloud
        """
        image = image.numpy()
        image = np.array(image, dtype=np.uint8).transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
        self.log_numpy_image(image, key, caption, size=size, now=now)

    def log_numpy_image(self, image, key, caption, size=(400, 400), now=True):
        """Log numpy image to WandB
        Parameters
        ----------
        image: (H, W, C) numpy.ndarray (RGB float32 or uint8 0-255 range)
            Image to log
        key: str
            Key associated with image
        caption: str
            Caption for the image
        size: tuple (int, int), default: (400, 400)
            Size of image to upload
        now: bool, default: False
            Flag to log immediately to cloud
        """
        height, width, _n_channels = image.shape
        assert _n_channels == 3
        image_to_log = Image.fromarray(image)
        image_to_log = image_to_log.resize(size, Image.BILINEAR)
        self._wandb_logger.history.row.update({key: [wandb.Image(image_to_log, caption=caption)]})
        if now:
            self.commit_log()

    def _parse_values_for_logging(self, key, values):
        """Utility to prep dictionary of values for logging"""
        if isinstance(values, dict):
            temp = {}
            for _k, _v in values.items():
                if key:
                    temp[key + '_' + _k] = _v
                else:
                    temp[_k] = _v
            # WandB has a length limit of 256 on keys so this fixes that error
            for _k in temp.copy():
                new_key = _k[:76] + ' ...' if len(_k) > 80 else _k
                temp[new_key] = temp.pop(_k)
        elif isinstance(values, numbers.Number):
            temp = {key: values}
        elif values is None:
            temp = {}
        else:
            raise TypeError("{} is of of type {}, cannot log it.".format(values, type(values)))
        return temp

    def end_log(self):
        """Uses global process to kill subprocesses. Changes coming in future"""
        try:
            wandb.join()
        except TypeError:
            pass
