from setuptools import find_packages
from setuptools import setup

setup(
    name="decarb-research",
    version="0.0.1",
    description="Parse results from LEAP and generate graphics",
    packages=find_packages(),
    python_requires=">=3.8",
)
