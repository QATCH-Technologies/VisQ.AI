"""
package_model.py
=================
Build a signed deployment package from the latest (or given) checkpoint(s).

Argparse'd replacement for scripts/packager.py's hardcoded main().
"""

from __future__ import annotations

import argparse

from visqai.logging_config import configure_logging
from visqai.packaging.packager import SecurePredictorPackager, get_latest_checkpoints


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Package a trained visqai model into a signed deployment zip.")
    p.add_argument("--experiments-dir", default="models/experiments", help="Used to auto-locate the latest checkpoint(s) if --checkpoints isn't given.")
    p.add_argument("--checkpoints", nargs="+", default=None, help="Explicit checkpoint path(s). Overrides --experiments-dir auto-detection.")
    p.add_argument("--output-dir", default=r"models\production")
    p.add_argument("--source-dir", default="src/visqai")
    p.add_argument("--package-name", default=None, help="Auto-generated (visqai_{single,ensemble}_{timestamp}) if omitted.")
    p.add_argument("--version", default="1.0")
    p.add_argument("--author", default="Unknown")
    p.add_argument("--client", default="Unknown")
    p.add_argument("--notes", default="")
    p.add_argument("--model-created-date", default=None)
    p.add_argument("--ensemble", action="store_true", default=None, help="Force ensemble packaging. Auto-detected from checkpoint count if omitted.")
    p.add_argument("--no-signing", dest="enable_signing", action="store_false", default=True)
    p.add_argument("--save-private-key", default=None, help="Path to also save the generated private key to.")
    return p.parse_args(argv)


def main(argv=None):
    configure_logging()
    args = parse_args(argv)

    if args.checkpoints:
        checkpoints = args.checkpoints
    else:
        print("Locating latest checkpoints...")
        checkpoints = get_latest_checkpoints(args.experiments_dir)

    packager = SecurePredictorPackager(
        output_dir=args.output_dir,
        source_dir=args.source_dir,
        enable_signing=args.enable_signing,
    )
    packager.package(
        model_paths=checkpoints,
        package_name=args.package_name,
        model_created_date=args.model_created_date,
        client=args.client,
        author=args.author,
        version=args.version,
        notes=args.notes,
        is_ensemble=args.ensemble,
        save_private_key=args.save_private_key,
    )


if __name__ == "__main__":
    main()
