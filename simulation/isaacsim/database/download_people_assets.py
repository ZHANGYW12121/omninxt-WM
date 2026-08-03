#!/usr/bin/env python3
"""Download the Warehouse crowd character assets for fully offline use."""

from pathlib import Path
import argparse
import hashlib
import os
import sys

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import omni.client
from pxr import Usd, UsdUtils


REMOTE_ROOT = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/5.1/Isaac/People/Characters"
)
REMOTE_PEOPLE_ROOT = REMOTE_ROOT.rsplit("/", 1)[0]
DEFAULT_DESTINATION = Path(
    os.environ.get(
        "PEGASUS_PEOPLE_ASSET_ROOT",
        Path(__file__).resolve().parents[3] / ".local" / "assets" / "people" / "Characters",
    )
)
CHARACTERS = (
    "original_male_adult_construction_05",
    "original_female_adult_business_02",
)
ANIMATION_FILES = (
    "LookAround.skelanim.usd",
    "Sit.skelanim.usd",
    "stand_idle_loop.skelanim.usd",
    "stand_idle_wave_loop.skelanim.usd",
    "stand_walk_1.skelanim.usd",
    "stand_walk_1_mirror.skelanim.usd",
    "stand_walk_2.skelanim.usd",
    "stand_walk_2_mirror.skelanim.usd",
    "stand_walk_3.skelanim.usd",
    "stand_walk_3_mirror.skelanim.usd",
    "stand_walk_4.skelanim.usd",
    "stand_walk_4_mirror.skelanim.usd",
    "stand_walk_5.skelanim.usd",
    "stand_walk_5_mirror.skelanim.usd",
    "stand_walk_7.skelanim.usd",
    "stand_walk_7_mirror.skelanim.usd",
)


def _check(result, action):
    if result != omni.client.Result.OK:
        raise RuntimeError(f"{action} failed: {result}")


def _download_file(remote_url, local_path):
    result, _, content = omni.client.read_file(remote_url)
    _check(result, f"download {remote_url}")
    payload = bytes(content)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = local_path.with_name(local_path.name + ".part")
    temporary_path.write_bytes(payload)
    temporary_path.replace(local_path)
    digest = hashlib.sha256(payload).hexdigest()[:12]
    print(f"downloaded {local_path} ({len(payload)} bytes, sha256={digest})")
    return len(payload)


def _download_tree(remote_dir, local_dir):
    result, entries = omni.client.list(remote_dir)
    _check(result, f"list {remote_dir}")
    file_count = 0
    byte_count = 0
    for entry in entries:
        name = entry.relative_path
        if name.startswith("."):
            continue
        remote_path = f"{remote_dir.rstrip('/')}/{name}"
        local_path = local_dir / name
        if entry.flags & omni.client.ItemFlags.CAN_HAVE_CHILDREN:
            child_files, child_bytes = _download_tree(remote_path, local_path)
            file_count += child_files
            byte_count += child_bytes
        else:
            byte_count += _download_file(remote_path, local_path)
            file_count += 1
    return file_count, byte_count


def _verify(destination):
    required = [destination / "Biped_Setup.usd"]
    for character in CHARACTERS:
        character_dir = destination / character
        required.extend(character_dir.glob("*.usd"))
        if not any(character_dir.glob("*.usd")):
            raise RuntimeError(f"No USD file downloaded in {character_dir}")
    required.extend(destination.parent / "Animations" / name for name in ANIMATION_FILES)
    biped_demo_usds = list((destination / "biped_demo").glob("*.usd"))
    if not biped_demo_usds:
        raise RuntimeError(f"No USD file downloaded in {destination / 'biped_demo'}")
    required.extend(biped_demo_usds)
    missing = [path for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError("Missing or empty files: " + ", ".join(map(str, missing)))

    for usd_path in required:
        stage = Usd.Stage.Open(str(usd_path), load=Usd.Stage.LoadAll)
        if stage is None:
            raise RuntimeError(f"USD cannot be opened: {usd_path}")
        _, dependencies, unresolved = UsdUtils.ComputeAllDependencies(str(usd_path))
        if unresolved:
            raise RuntimeError(
                f"Unresolved dependencies in {usd_path}: "
                + ", ".join(map(str, unresolved))
            )
        remote_dependencies = [
            str(path)
            for path in dependencies
            if str(path).startswith(("http://", "https://", "omniverse://"))
        ]
        if remote_dependencies:
            raise RuntimeError(
                f"Remote dependencies remain in {usd_path}: "
                + ", ".join(remote_dependencies)
            )
        print(f"verified offline USD: {usd_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument(
        "--verify-only", action="store_true", help="validate the existing local mirror"
    )
    args = parser.parse_args()
    destination = args.destination.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    if args.verify_only:
        _verify(destination)
        print(f"Offline people assets verified: {destination}")
        return 0

    total_files = 0
    total_bytes = _download_file(
        f"{REMOTE_ROOT}/Biped_Setup.usd", destination / "Biped_Setup.usd"
    )
    total_files += 1
    for character in CHARACTERS:
        files, size = _download_tree(
            f"{REMOTE_ROOT}/{character}", destination / character
        )
        total_files += files
        total_bytes += size
    files, size = _download_tree(
        f"{REMOTE_ROOT}/biped_demo", destination / "biped_demo"
    )
    total_files += files
    total_bytes += size
    animations_dir = destination.parent / "Animations"
    for animation_name in ANIMATION_FILES:
        total_bytes += _download_file(
            f"{REMOTE_PEOPLE_ROOT}/Animations/{animation_name}",
            animations_dir / animation_name,
        )
        total_files += 1

    _verify(destination)
    print(
        f"Offline people assets ready: {destination} "
        f"({total_files} files, {total_bytes / (1024 * 1024):.1f} MiB)"
    )
    print(f"export PEGASUS_PEOPLE_ASSET_ROOT={destination}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
    finally:
        simulation_app.close()
