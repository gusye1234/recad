from setuptools import setup

with open('readme.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='reck',
    version='0.0.1',
    author='Jianbai Ye',
    packages=['reck'],
    description='A unified framework for recommender system attacking',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='GPLv3',
    install_requires=['torch', 'pandas', 'scipy', 'tabulate', 'tqdm', 'coloredlogs'],
    python_requires='>=3.6',
)
