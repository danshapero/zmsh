from setuptools import setup, find_packages

setup(
    name="zmsh",
    version="0.0.1",
    license="proprietary",
    description="mesh topology transformation via SMT",
    author="Daniel Shapero",
    packages=find_packages(exclude=["doc", "test"]),
    install_requires=["numpy", "scipy", "z3-solver"],
)
