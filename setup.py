from setuptools import setup, find_packages

setup(
    name='simpledl',
    version='0.1.1',
    url='https://github.com/thomaswilley/pysimpledl',
    author='@thomaswilley',
    description=('A simple deep learning package with minimal dependencies and limited interoperability with ONNX.'),
    license='BSD',
    keywords='AI ML DL machinelearning',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    install_requires=['numpy'],
)
