from setuptools import setup


vars2find = ['__author__', '__version__', '__url__']
varstfp = {}
with open('./recad/__init__.py') as f:
    for line in f.readlines():
        for v in vars2find:
            if line.startswith(v):
                line = line.replace(' ', '').replace("\"", '').replace("\'", '').strip()
                varstfp[v] = line.split('=')[1]

with open('readme.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='recad',
    url=varstfp['__url__'],
    version=varstfp['__version__'],
    author=varstfp['__author__'],
    packages=['recad'],
    description='A unified framework for recommender system attacking',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='GPLv3',
    install_requires=['torch', 'pandas', 'scipy', 'tabulate', 'tqdm', 'coloredlogs'],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'recad_runner = recad:main',
        ]
    },
)
