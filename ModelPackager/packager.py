#!/usr/bin/env python3
"""
modular_build_package.py

A fully‐modular refactoring that now supports a pluggable `Trainer` class,
which is responsible for doing the model fitting and exporting a SavedModel.

Usage example:
    python modular_build_package.py \
        --bundle-root MyBundle \
        --package-name mypkg \
        --train-csv data/train.csv \
        --dataprocessor-module mymodule.dataprocessor \
        --dataprocessor-class MyDataProcessor \
        --transformer-module mymodule.transformer \
        --transformer-class MyTransformerPipeline \
        --trainer-module mymodule.trainer \
        --trainer-class MyTrainer \
        --trainer-kwargs '{"learning_rate": 1e-3, "layers": [128, 64]}' \
        --package-version 0.2.1 \
        --install-requires "scikit-learn>=0.24" "tensorflow>=2.3"
"""

import os
import shutil
import compileall
import textwrap
import importlib
import argparse
import cloudpickle
import json
from dataclasses import dataclass, field
from typing import List, Dict
import numpy as np
# mymodule/trainer.py

import os
from keras import layers, models
from keras.callbacks import EarlyStopping
import pandas as pd

# ——————————————————————————————————————————————————————————————————————————————
# 1) Define a simple config dataclass to hold everything you might want to swap out
# ——————————————————————————————————————————————————————————————————————————————


class CategoricalEncoder:
    """
    Find every column ending in '_type' and map its unique values
    to integers 1..n (missing → 0). Mappings are stored so you can
    transform future data the same way.
    """

    def __init__(self):
        self.mapping: dict[str, dict[str, int]] = {}
        self.categorical_columns: list[str] = []

    def fit(self, X: pd.DataFrame):
        # detect all _type columns
        self.categorical_columns = [
            c for c in X.columns if c.endswith('_type')]
        for col in self.categorical_columns:
            # factorize picks up all unique non-null values
            codes, uniques = pd.factorize(X[col].astype(str), sort=False)
            # codes are 0..n-1 for uniques, -1 for any NaN; shift +1 → 1..n, 0 for NaN
            self.mapping[col] = {val: i+1 for i, val in enumerate(uniques)}

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.categorical_columns:
            # map unseen values to 0
            X[col] = X[col].astype(str).map(
                self.mapping[col]).fillna(0).astype(int)
        return X

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.fit(X)
        return self.transform(X)


@dataclass
class BuilderConfig:
    # Filesystem‐level names
    bundle_root: str = "VisQAI_base"
    package_name: str = "package"
    source_py_files: List[str] = field(
        default_factory=lambda: ["dataprocessor.py", "predictor.py"]
    )
    train_csv: str = os.path.join("ModelPackager", "train_features.csv")

    # Python‐level components for data processing + transformation
    dataprocessor_module: str = "dataprocessor"
    dataprocessor_class: str = "DataProcessor"
    transformer_module: str = "transformer"
    transformer_class: str = "TransformerPipeline"

    trainer_module: str = "train"
    trainer_class: str = "Trainer"
    trainer_kwargs: Dict = field(default_factory=dict)

    dataprocessor_kwargs: Dict = field(default_factory=lambda: {
        "drop_columns": [],
        "feature_columns": None,
        "target_prefix": "Viscosity_"
    })

    # Model‐training parameters (these will be passed into TrainerClass via trainer_kwargs if desired)
    # (We no longer build the Keras model here; the Trainer is in charge.)
    # You can pass anything you like in `trainer_kwargs` JSON from CLI

    # Package metadata
    package_version: str = "0.1.0"
    install_requires: List[str] = field(
        default_factory=lambda: ["scikit-learn>=0.24", "tensorflow>=2.0"]
    )
    author: str = "QATCH Technologies"
    description: str = "Base VisQ.AI Regressor"


# ——————————————————————————————————————————————————————————————————————————————
# 2) The PackageBuilder class implements each step as a method
# ——————————————————————————————————————————————————————————————————————————————

class PackageBuilder:
    def __init__(self, config: BuilderConfig):
        self.cfg = config

        # Fully‐qualified paths
        self.bundle_root = self.cfg.bundle_root
        self.package_dir = os.path.join(
            self.bundle_root, self.cfg.package_name)

        # Dynamically import:
        #  - DataProcessor class
        #  - Transformer class
        #  - Trainer class
        self.DataProcessorClass = self._dynamic_import(
            self.cfg.dataprocessor_module,
            self.cfg.dataprocessor_class
        )
        self.TransformerClass = self._dynamic_import(
            self.cfg.transformer_module,
            self.cfg.transformer_class
        )
        self.TrainerClass = self._dynamic_import(
            self.cfg.trainer_module,
            self.cfg.trainer_class
        )

    def _dynamic_import(self, module_path: str, attr_name: str):
        """
        Dynamically import `attr_name` from `module_path`.
        """
        module = importlib.import_module(module_path)
        try:
            return getattr(module, attr_name)
        except AttributeError:
            raise ImportError(
                f"Cannot find '{attr_name}' in module '{module_path}'"
            )

    def create_pkg_dirs(self):
        """
        1) Remove existing bundle_root (if present).
        2) Create bundle_root/package_name.
        3) Copy source_py_files into that package dir.
        4) Touch an empty __init__.py.
        """
        # derive your project root from the location of this file:
        project_root = os.path.dirname(os.path.abspath(__file__))

        # 1) Remove any existing bundle
        if os.path.isdir(self.bundle_root):
            shutil.rmtree(self.bundle_root)

        # 2) Recreate package dir
        os.makedirs(self.package_dir, exist_ok=True)

        # 3) Copy each source file, resolving its path under project_root
        for fname in self.cfg.source_py_files:
            src = os.path.join(project_root, fname)
            if not os.path.isfile(src):
                raise FileNotFoundError(
                    f"Expected source file '{fname}' at '{src}'."
                )
            shutil.copy(src, self.package_dir)

        # 4) Touch __init__.py
        init_path = os.path.join(self.package_dir, "__init__.py")
        with open(init_path, "w", encoding="utf-8"):
            pass

    def compile_and_remove_py(self):
        """
        1) Compile all .py → .pyc under __pycache__.
        2) Delete the raw .py files, leaving only __pycache__/*.pyc + __init__.py.
        """
        compileall.compile_dir(self.package_dir, force=True, quiet=1)

        for fname in self.cfg.source_py_files:
            full_py = os.path.join(self.package_dir, fname)
            if os.path.isfile(full_py):
                os.remove(full_py)

    def train_and_dump_objects(self):
        """
        1) Instantiate & fit DataProcessor.
        2) Fit Transformer on processed features.
        3) Instantiate Trainer(input_dim, output_dim, **trainer_kwargs) and call train(X_scaled, y, save_dir).
        4) Serialize the fitted transformer via cloudpickle.
        """
        os.makedirs(self.package_dir, exist_ok=True)

        DP = self.DataProcessorClass(self.cfg.dataprocessor_kwargs)
        X_df, y_df = DP.process_train(self.cfg.train_csv)

        cat_enc = CategoricalEncoder()
        X_df = cat_enc.fit_transform(X_df)
        print(X_df)
        transformer = self.TransformerClass()
        X_scaled = transformer.fit_transform(X_df)

        input_dim = X_scaled.shape[1]
        output_dim = y_df.shape[1]
        trainer = self.TrainerClass(
            input_dim=input_dim,
            output_dim=output_dim,
            **self.cfg.trainer_kwargs
        )

        # Let the Trainer do the fitting and exporting. We expect it to write a SavedModel under package_dir/model/
        saved_model_dir = os.path.join(self.package_dir, "model")
        if os.path.isdir(saved_model_dir):
            shutil.rmtree(saved_model_dir)

        # The Trainer must create `saved_model_dir` (including assets/, variables/, saved_model.pb)
        trainer.train(
            X_scaled.values,
            y_df.values,
            save_dir=saved_model_dir
        )

        # STEP D: Serialize the fitted transformer
        transformer_path = os.path.join(self.package_dir, "transformer.pkl")
        with open(transformer_path, "wb") as f:
            # We assume TransformerPipeline has an attribute `.scaler.pipeline`
            cloudpickle.dump(transformer.scaler.pipeline, f)

        print("Finished training & dumping:")
        print(f"  - SavedModel directory: {saved_model_dir}")
        print(f"  - transformer.pkl at: {transformer_path}")

    def write_setup_py(self):
        """
        Create setup.py in bundle_root that includes:
          - The SavedModel folder under package_dir/model/
          - transformer.pkl
          - __pycache__/*.pyc
        """
        setup_py = os.path.join(self.bundle_root, "setup.py")
        content = textwrap.dedent(f"""
            from setuptools import setup, find_packages

            setup(
                name="{self.cfg.package_name}",
                version="{self.cfg.package_version}",
                packages=find_packages(),
                include_package_data=True,
                package_data={{
                    "{self.cfg.package_name}": [
                        "model/**/*",
                        "transformer.pkl",
                        "__pycache__/*.pyc"
                    ]
                }},
                zip_safe=False,
                install_requires={self.cfg.install_requires},
                author="{self.cfg.author}",
                description="{self.cfg.description}",
            )
        """).lstrip()

        with open(setup_py, "w", encoding="utf-8") as f:
            f.write(content)

    def zip_bundle(self):
        """
        Create a ZIP archive of the entire bundle_root, named bundle_root.zip
        """
        zip_name = f"{self.bundle_root}.zip"
        if os.path.isfile(zip_name):
            os.remove(zip_name)

        archive_path = shutil.make_archive(
            self.bundle_root, "zip", self.bundle_root
        )
        print(f"Created zip archive: {archive_path}")

    def run_all(self):
        """
        Sequentially run all steps:
          1) create_pkg_dirs
          2) compile_and_remove_py
          3) train_and_dump_objects
          4) write_setup_py
          5) zip_bundle
        """
        print("Creating package directories…")
        self.create_pkg_dirs()

        print("Compiling .py → .pyc and removing raw .py…")
        self.compile_and_remove_py()

        print("Training & dumping objects via Trainer…")
        self.train_and_dump_objects()

        print("Writing setup.py…")
        self.write_setup_py()

        print("Zipping bundle…")
        self.zip_bundle()

        print("\n==> All done!  Bundle is at:",
              self.bundle_root, "and", f"{self.bundle_root}.zip")
        print(
            f"To build a wheel, run:\n  cd {self.bundle_root}\n  python -m pip install --upgrade wheel setuptools\n  python setup.py bdist_wheel\n"
        )
        print("\nThe resulting .whl will contain only:")
        print(f"  {self.cfg.package_name}/__init__.py")
        print(f"  {self.cfg.package_name}/__pycache__/*.pyc")
        print(f"  {self.cfg.package_name}/model/**/*")
        print(f"  {self.cfg.package_name}/transformer.pkl")
        print("\nNo raw .py sources or model.h5 are included in the final artifact.")


# ——————————————————————————————————————————————————————————————————————————————
# 3) CLI glue: parse overrides, instantiate BuilderConfig, and run PackageBuilder
# ——————————————————————————————————————————————————————————————————————————————

def main():
    parser = argparse.ArgumentParser(
        description="Modular build script for packaging a VisQ.AI‐style bundle (with a pluggable Trainer)."
    )

    # Bundle/package names
    parser.add_argument("--bundle-root", default="VisQAI-base",
                        help="Root directory for the bundle (default: VisQAI-base)")
    parser.add_argument("--package-name", default="package",
                        help="Subfolder name under bundle_root (default: package)")

    # Source .py files
    parser.add_argument("--source-files", nargs="+",
                        default=["dataprocessor.py", "predictor.py"],
                        help="List of .py source files to include/compile")

    # Training CSV path
    parser.add_argument("--train-csv", default="ModelPackager/train_features.csv",
                        help="Path to the CSV used for training")

    # DataProcessor class
    parser.add_argument("--dataprocessor-module", default="dataprocessor",
                        help="Python module path for the DataProcessor (default: dataprocessor)")
    parser.add_argument("--dataprocessor-class", default="DataProcessor",
                        help="Class name inside dataprocessor_module (default: DataProcessor)")

    # Transformer class
    parser.add_argument("--transformer-module", default="transformer",
                        help="Python module path for the TransformerPipeline (default: transformer)")
    parser.add_argument("--transformer-class", default="TransformerPipeline",
                        help="Class name inside transformer_module (default: TransformerPipeline)")

    # Trainer class (new)
    parser.add_argument("--trainer-module", default="train",
                        help="Python module path for the Trainer class (default: train)")
    parser.add_argument("--trainer-class", default="Trainer",
                        help="Class name inside trainer_module (default: Trainer)")

    # Trainer kwargs, passed as a JSON string
    parser.add_argument(
        "--trainer-kwargs",
        type=str,
        default="{}",
        help="JSON‐encoded dict of kwargs to pass into TrainerClass, e.g. '{\"learning_rate\":0.001, \"layers\":[128,64]}'"
    )

    # Package metadata overrides
    parser.add_argument("--package-version", default="0.1.0",
                        help="Version string for setup.py (default: 0.1.0)")
    parser.add_argument("--install-requires", nargs="*", default=["scikit-learn>=0.24", "tensorflow>=2.0"],
                        help="List of pip requirements (e.g. --install-requires sklearn>=0.24 tensorflow>=2.0)")

    parser.add_argument("--author", default="QATCH Technologies",
                        help="Author string for setup.py (default: QATCH Technologies)")
    parser.add_argument("--description", default="Base VisQ.AI Regressor",
                        help="Description string for setup.py")

    args = parser.parse_args()

    # Parse the JSON‐encoded trainer kwargs
    try:
        trainer_kwargs = json.loads(args.trainer_kwargs)
        if not isinstance(trainer_kwargs, dict):
            raise ValueError(
                "trainer-kwargs must be a JSON object (i.e. a dict).")
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse trainer-kwargs: {e}")

    # Build the dataprocessor kwargs dict (you can also expose these as CLI args if desired)
    dataprocessor_kwargs = {
        "drop_columns": [],
        "feature_columns": None,
        "target_prefix": "Viscosity_"
    }

    config = BuilderConfig(
        bundle_root=args.bundle_root,
        package_name=args.package_name,
        source_py_files=args.source_files,
        train_csv=args.train_csv,
        dataprocessor_module=args.dataprocessor_module,
        dataprocessor_class=args.dataprocessor_class,
        transformer_module=args.transformer_module,
        transformer_class=args.transformer_class,
        trainer_module=args.trainer_module,
        trainer_class=args.trainer_class,
        trainer_kwargs=trainer_kwargs,
        dataprocessor_kwargs=dataprocessor_kwargs,
        package_version=args.package_version,
        install_requires=args.install_requires,
        author=args.author,
        description=args.description
    )

    builder = PackageBuilder(config)
    builder.run_all()


if __name__ == "__main__":
    main()
