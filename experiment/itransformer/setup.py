from setuptools import setup

setup(
    name='itransformer-mia',
    version='0.1.0',
    description='iTransformer + MIA pipeline for PTB-XL (TV3)',
    packages=['itransformer'],
    install_requires=[
        'numpy>=1.26',
        'scipy>=1.11',
        'scikit-learn>=1.4',
        'torch>=2.2',
        'pandas>=2.0',
        'wfdb>=4.1',
        'tqdm',
        'matplotlib',
    ],
)
