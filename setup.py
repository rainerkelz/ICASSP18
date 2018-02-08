from setuptools import setup
setup(
    name='pipedream',
    version='0.1',
    description='experimenting with DL',
    author='Rainer Kelz',
    author_email='rainer.kelz@jku.at',
    platforms=['any'],
    license='MIT',
    url='https://gitlab.cp.jku.at/rainer/pipedream',
    packages=['pipedream'],
    install_requires=[
        'numpy',
        'scipy',
        'theano',
        'lasagne',
        'madmom',
        'scikit-learn',
        'matplotlib'
    ]
)
