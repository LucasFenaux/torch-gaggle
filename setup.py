from setuptools import find_packages, setup
setup(
    name='torch-gaggle',
    packages=find_packages(include=['gaggle*']),
    version='0.0.2',
    description='Gaggle: Genetic Algorithms on the GPU using PyTorch',
    author='Lucas Fenaux',
    license='MIT',
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "transformers",
        "matplotlib",
        "gym"
    ]
)
