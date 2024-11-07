from setuptools import setup, find_packages

setup(
    name='sprout',
    version='0.1.0',
    packages=find_packages(include=['sprout', 'sprout.*']),
    install_requires=[
        "transformers",
        "torch",
        "datasets",
        "numpy<2.0.0",
        "ipykernel",
        "ipywidgets",
        "jsonlines", 
    ],
)