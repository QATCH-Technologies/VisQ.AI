# setup.py
from setuptools import setup, find_packages

setup(
    name="visqai",
    version="1.0.0",
    author="Paul MacNichol",
    author_email="paul.macnichol@qatchtech.com",
    description="VisQ.AI core package",
    # <-- look for packages in visQAI/src, not src
    packages=find_packages(where="visQAI/src"),
    package_dir={"": "visQAI/src"},
    install_requires=[
        "joblib>=1.2.0",
        "pandas>=1.5.0",
        "PyQt5>=5.15.0",
    ],
    python_requires=">=3.8",
)
