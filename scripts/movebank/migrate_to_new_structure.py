#!/usr/bin/env python3
"""
Migrate data from legacy directory structure to new structure.

Legacy structure:
    chips_out/{study_id}/point_{id}_{big/medium/small}.tif
    chips_out/{study_id}_manifest.json
    dem_out/{study_id}/{study_id}_dem_chunk_{i}.tif
    dem_out/{study_id}/dem_manifest.json

New structure:
    studies/{study_id}/
        manifest.json
        trajectories.csv
        chips/train/{point_id:05d}_{layer_group}.tif
        chips/test/{point_id:05d}_{layer_group}.tif
        tiles/dem/tile_{i:03d}.tif
        tiles/dem/index.json

Uses symlinks to avoid duplicating data.
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import pandas as pd


# Mapping from old resolution names to new layer group names
RESOLUTION_TO_LAYER = {
    "big": "era5",  # ERA5-Land + ERA5
    "medium": "modis",  # MODIS spectral products
    # "small": "landcover",  # Dynamic World (disabled)
}


def find_trajectory_csv(study_id: str, csv_dir: Path) -> Path | None:
    """Find the trajectory CSV for a study."""
    for f in csv_dir.glob(f"{study_id}_*_traintest.csv"):
        return f
    return None


def migrate_study(
    study_id: str,
    chips_dir: Path,
    dem_dir: Path,
    csv_dir: Path,
    output_dir: Path,
    use_symlinks: bool = True,
    dry_run: bool = False,
) -> dict:
    """
    Migrate a single study to the new structure.

    Returns dict with migration status and stats.
    """
    result = {
        "study_id": study_id,
        "status": "success",
        "chips_migrated": 0,
        "dem_tiles_migrated": 0,
        "errors": [],
    }

    study_chips_dir = chips_dir / study_id
    study_dem_dir = dem_dir / study_id
    study_output_dir = output_dir / "studies" / study_id

    # Check if source directories exist
    if not study_chips_dir.exists():
        result["status"] = "skipped"
        result["errors"].append(f"Chips directory not found: {study_chips_dir}")
        return result

    # Find trajectory CSV
    csv_path = find_trajectory_csv(study_id, csv_dir)
    if csv_path is None:
        result["status"] = "error"
        result["errors"].append(f"Trajectory CSV not found for {study_id}")
        return result

    print(f"Migrating {study_id}...")

    if dry_run:
        print(f"  [DRY RUN] Would create: {study_output_dir}")
    else:
        # Create output directories
        (study_output_dir / "chips" / "train").mkdir(parents=True, exist_ok=True)
        (study_output_dir / "chips" / "test").mkdir(parents=True, exist_ok=True)
        (study_output_dir / "tiles" / "dem").mkdir(parents=True, exist_ok=True)

    # Load trajectory CSV to get train/test split
    df = pd.read_csv(csv_path)
    df = df.sort_values(["individual-local-identifier", "timestamp"]).reset_index(drop=True)
    df["point_id"] = df.index

    # Save trajectories.csv with point_id
    if not dry_run:
        df.to_csv(study_output_dir / "trajectories.csv", index=False)

    # Build point_id -> split mapping
    point_to_split = dict(zip(df["point_id"], df["set"]))

    # Migrate chips
    for old_name, new_layer in RESOLUTION_TO_LAYER.items():
        for chip_file in study_chips_dir.glob(f"point_*_{old_name}.tif"):
            # Parse point ID from filename
            try:
                point_id = int(chip_file.stem.split("_")[1])
            except (IndexError, ValueError):
                result["errors"].append(f"Could not parse point ID from {chip_file.name}")
                continue

            # Get split for this point
            split = point_to_split.get(point_id)
            if split not in ("train", "test"):
                # Point not in CSV or invalid split
                continue

            # New filename
            new_filename = f"{point_id:05d}_{new_layer}.tif"
            new_path = study_output_dir / "chips" / split / new_filename

            if dry_run:
                print(f"  [DRY RUN] {chip_file} -> {new_path}")
            else:
                if new_path.exists():
                    new_path.unlink()

                if use_symlinks:
                    # Create relative symlink
                    rel_path = os.path.relpath(chip_file.resolve(), new_path.parent)
                    new_path.symlink_to(rel_path)
                else:
                    shutil.copy2(chip_file, new_path)

            result["chips_migrated"] += 1

    # Migrate DEM tiles
    dem_manifest_path = study_dem_dir / "dem_manifest.json"
    if not dem_manifest_path.exists():
        # Try the _new variant
        dem_manifest_path = study_dem_dir / "dem_manifest_new.json"

    if dem_manifest_path.exists():
        with open(dem_manifest_path) as f:
            dem_manifest = json.load(f)

        new_dem_index = {"tiles": []}

        for chunk in dem_manifest.get("chunk_info", []):
            old_filename = chunk["filename"]
            old_path = study_dem_dir / old_filename
            chunk_id = chunk.get("chunk_id", 0)

            if not old_path.exists():
                result["errors"].append(f"DEM tile not found: {old_path}")
                continue

            new_filename = f"tile_{chunk_id:03d}.tif"
            new_path = study_output_dir / "tiles" / "dem" / new_filename

            if dry_run:
                print(f"  [DRY RUN] {old_path} -> {new_path}")
            else:
                if new_path.exists():
                    new_path.unlink()

                if use_symlinks:
                    rel_path = os.path.relpath(old_path.resolve(), new_path.parent)
                    new_path.symlink_to(rel_path)
                else:
                    shutil.copy2(old_path, new_path)

            # Add to new index
            new_dem_index["tiles"].append({
                "file": new_filename,
                "bounds": chunk["bounds"],
            })

            result["dem_tiles_migrated"] += 1

        # Write new DEM index
        if not dry_run:
            with open(study_output_dir / "tiles" / "dem" / "index.json", "w") as f:
                json.dump(new_dem_index, f, indent=2)
    else:
        result["errors"].append(f"DEM manifest not found: {dem_manifest_path}")

    # Create study manifest
    train_count = len(df[df["set"] == "train"])
    test_count = len(df[df["set"] == "test"])

    manifest = {
        "study_id": study_id,
        "version": "1.0.0",
        "trajectories": {
            "file": "trajectories.csv",
            "n_points": len(df),
            "n_individuals": df["individual-local-identifier"].nunique(),
        },
        "splits": {
            "train": {"n_points": train_count},
            "test": {"n_points": test_count},
        },
        "chips": {
            "layer_groups": list(RESOLUTION_TO_LAYER.values()),
            "train_count": train_count,
            "test_count": test_count,
        },
        "tiles": {
            "dem": {
                "index": "tiles/dem/index.json",
                "count": result["dem_tiles_migrated"],
            }
        },
    }

    if not dry_run:
        with open(study_output_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

    print(f"  Migrated {result['chips_migrated']} chips, {result['dem_tiles_migrated']} DEM tiles")
    if result["errors"]:
        print(f"  Warnings: {len(result['errors'])}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Migrate MoveBench data to new structure")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/data/vision/beery/scratch/justin/movebench"),
        help="Root data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: data-root)",
    )
    parser.add_argument(
        "--study-id",
        type=str,
        default=None,
        help="Migrate only this study (default: all)",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of creating symlinks",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    args = parser.parse_args()

    data_root = args.data_root
    output_dir = args.output_dir or data_root

    chips_dir = data_root / "chips_out"
    dem_dir = data_root / "dem_out"
    csv_dir = data_root / "mbr_studies_2year_filtered_traintest"

    # Find studies to migrate
    if args.study_id:
        study_ids = [args.study_id]
    else:
        # Find all studies with chips
        study_ids = [
            d.name for d in chips_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]

    print(f"Found {len(study_ids)} studies to migrate")
    print(f"Output directory: {output_dir}")
    print(f"Using {'copies' if args.copy else 'symlinks'}")
    if args.dry_run:
        print("DRY RUN - no changes will be made")
    print()

    results = []
    for study_id in sorted(study_ids):
        result = migrate_study(
            study_id=study_id,
            chips_dir=chips_dir,
            dem_dir=dem_dir,
            csv_dir=csv_dir,
            output_dir=output_dir,
            use_symlinks=not args.copy,
            dry_run=args.dry_run,
        )
        results.append(result)

    # Summary
    print()
    print("=" * 60)
    print("Migration Summary")
    print("=" * 60)

    success = sum(1 for r in results if r["status"] == "success")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    errors = sum(1 for r in results if r["status"] == "error")

    print(f"  Success: {success}")
    print(f"  Skipped: {skipped}")
    print(f"  Errors:  {errors}")

    total_chips = sum(r["chips_migrated"] for r in results)
    total_dem = sum(r["dem_tiles_migrated"] for r in results)
    print(f"  Total chips migrated: {total_chips}")
    print(f"  Total DEM tiles migrated: {total_dem}")

    # Write summary
    if not args.dry_run:
        summary_path = output_dir / "studies" / "migration_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSummary written to: {summary_path}")


if __name__ == "__main__":
    main()
