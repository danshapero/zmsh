from setuptools import setup, find_packages

setup(
    name="zmsh",
    version="0.0.1",
    license="MPL 2.0",
    description="mesh topology transformation via SMT",
    author="Daniel Shapero",
    author_email="shapero@uw.edu",
    packages=find_packages(exclude=["doc", "test"]),
    install_requires=["numpy", "scipy", "z3-solver"],
)
