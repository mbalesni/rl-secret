from setuptools import setup

setup(name='secret',
      version='0.1',
      install_requires=[
          'torch',
          'numpy',
          'gym',
          'click',
          'wandb',
          'matplotlib',
          'scikit-image',
          'xvfbwrapper',
          'tqdm',
      ])
