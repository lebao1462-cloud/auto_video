import argparse
import os
from pathlib import Path


def default_packages_dir() -> Path:
    configured = os.getenv("ARGOS_PACKAGES_DIR")
    if configured:
        return Path(configured)
    if os.name == "nt" and Path("D:/").exists():
        return Path("D:/ArgosTranslate/packages")
    return Path.home() / ".local" / "share" / "argos-translate" / "packages"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Explicitly install Argos packages required by Buzz localization."
    )
    parser.add_argument(
        "--packages-dir",
        default=str(default_packages_dir()),
        help="Argos package directory (default follows ARGOS_PACKAGES_DIR / drive D policy).",
    )
    args = parser.parse_args()

    packages_dir = Path(args.packages_dir).expanduser().resolve()
    packages_dir.mkdir(parents=True, exist_ok=True)
    os.environ["ARGOS_PACKAGES_DIR"] = str(packages_dir)

    from argostranslate import package

    print(f"Argos packages directory: {packages_dir}")
    print("Refreshing Argos package index (explicit user action)...")
    package.update_package_index()
    available = package.get_available_packages()

    required = [("en", "vi"), ("zh", "en")]
    for source, target in required:
        candidate = next(
            (item for item in available if item.from_code == source and item.to_code == target),
            None,
        )
        if candidate is None:
            raise RuntimeError(f"Argos package index does not contain {source}->{target}.")
        print(f"Installing {source}->{target} package {candidate.package_version}...")
        candidate.install()
        print(f"Installed {source}->{target}.")

    print("Argos offline translation packages are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
