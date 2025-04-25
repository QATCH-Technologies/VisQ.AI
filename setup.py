from setuptools import setup, find_packages

setup(
    name="visqai",
    version="1.0.0",
    author="Paul MacNichol",
    author_email="paul.macnichol@qatchtech.com",
    description="VisQ.AI core package",
    packages=find_packages(where="visQAI/src"),
    package_dir={"": "visQAI/src"},
    python_requires=">=3.8",
    install_requires=[
        # Core
        "joblib>=1.2.0",
        "pandas>=1.5.0",
        "PyQt5>=5.15.0",
        # ML & plotting
        "tensorflow>=2.10.0",
        "optuna>=3.0.0",
        "matplotlib>=3.6.0",
        # Additional model libs
        "xgboost>=1.7.0",
        "scikit-learn>=1.0.0",
    ],
)
