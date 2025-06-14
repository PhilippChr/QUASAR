from setuptools import find_packages, setup

with open("requirements.txt", "r") as f:
    requirements = list(f.read().splitlines())

setup(
    name="quasar",
    version="1.0",
    description="Code for the QUASAR project (published in IEEE Data Engineering Bulletin 2024).",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Philipp Christmann",
    author_email="pchristm@mpi-inf.mpg.de",
    url="https://qa.mpi-inf.mpg.de/quasar",
    packages=find_packages(),
    include_package_data=False,
    keywords=[
        "qa",
        "question answering",
        "heterogeneous QA",
        "knowledge bases",
        "heterogeneous sources",
        "graph neural networks",
        "GNNs",
        "iterative GNNs",
    ],
    classifiers=["Programming Language :: Python :: 3.8"],
    install_requires=requirements,
)
