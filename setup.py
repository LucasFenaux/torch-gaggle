from setuptools import find_packages, setup
setup(
    name='torch-gaggle',
    packages=find_packages(include=['gaggle*']),
    version='0.0.1',
    description='Testing gaggle pip packaging',
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
