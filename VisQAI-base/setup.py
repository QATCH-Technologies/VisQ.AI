from setuptools import setup, find_packages

setup(
    name="package",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "package": [
            "model/**/*",
            "transformer.pkl",
            "__pycache__/*.pyc"
        ]
    },
    zip_safe=False,
    install_requires=['scikit-learn>=0.24', 'tensorflow>=2.0'],
    author="QATCH Technologies",
    description="Base VisQ.AI Regressor",
)
