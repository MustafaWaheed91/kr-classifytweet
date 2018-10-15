from setuptools import setup, find_packages

setup(
    name='classifytweet',
    version='0.0.1rc0',

    # Package data
    packages=find_packages(),
    include_package_data=True,

    # Insert dependencies list here
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'nltk',
        'Keras',
        'tensorflow',
        'matplotlib'
    ],

    entry_points={
       "classifytweet.training": [
           "train=classifytweet.train:entry_point",
       ]
    }
)