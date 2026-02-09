#!/usr/bin/env python3
"""Add a broad taxonomic group column to Movebank metadata."""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

GROUP_AVIAN = "avian"
GROUP_TERRESTRIAL = "terrestrial mammals"
GROUP_REPTILES = "reptiles"
GROUP_MARINE = "marine"
GROUP_HUMAN = "humans/domestic"
GROUP_UNKNOWN = "Unknown"

COLUMN_NAME = "broad_taxonomic_group"
IGNORED_SPECIES_TOKENS = {"unknown", "calibration"}

SPECIES_OVERRIDES: Dict[str, str] = {
    "Bos taurus": GROUP_HUMAN,
    "Canis": GROUP_HUMAN,
    "Capra hircus": GROUP_HUMAN,
    "Felis catus": GROUP_HUMAN,
    "Homo sapiens": GROUP_HUMAN,
    "Ovis aries": GROUP_HUMAN,
}

GENUS_TO_GROUP: Dict[str, str] = {}


def _register_genera(group: str, genera: Iterable[str]) -> None:
    for genus in genera:
        if genus in GENUS_TO_GROUP and GENUS_TO_GROUP[genus] != group:
            raise ValueError(
                f"Genus {genus} already registered for {GENUS_TO_GROUP[genus]}"
            )
        GENUS_TO_GROUP[genus] = group


AVIAN_GENERA = {
    "Anas",
    "Anser",
    "Anthropoides",
    "Antrostomus",
    "Aquila",
    "Ardea",
    "Bartramia",
    "Branta",
    "Buceros",
    "Buteo",
    "Cairina",
    "Calonectris",
    "Cathartes",
    "Ciconia",
    "Circaetus",
    "Columba",
    "Coragyps",
    "Corvus",
    "Creagrus",
    "Chordeiles",
    "Eudocimus",
    "Falco",
    "Grus",
    "Gyps",
    "Hydroprogne",
    "Larus",
    "Luscinia",
    "Milvus",
    "Morus",
    "Mycteria",
    "Neochen",
    "Neophron",
    "Numenius",
    "Nucifraga",
    "Oceanodroma",
    "Pelecanus",
    "Pernis",
    "Phaethon",
    "Phoebastria",
    "Progne",
    "Pterodroma",
    "Puffinus",
    "Rhyticeros",
    "Sarcoramphus",
    "Seiurus",
    "Sturnella",
    "Thalassarche",
    "Thalassoica",
    "Tockus",
    "Torgos",
    "Tyto",
    "Uria",
    "Zonotrichia",
}

TERRESTRIAL_GENERA = {
    "Acinonyx",
    "Antilocapra",
    "Ateles",
    "Brachylagus",
    "Canis",
    "Caracal",
    "Cebus",
    "Cervus",
    "Eidolon",
    "Equus",
    "Gulo",
    "Hypsignathus",
    "Lavia",
    "Leopardus",
    "Leptonycteris",
    "Loxodonta",
    "Lycalopex",
    "Lynx",
    "Martes",
    "Myotis",
    "Nasua",
    "Noctilio",
    "Nyctalus",
    "Nyctereutes",
    "Odocoileus",
    "Panthera",
    "Papio",
    "Phyllostomus",
    "Potos",
    "Prionailurus",
    "Pteropus",
    "Rangifer",
    "Syncerus",
    "Tapirus",
    "Tremarctos",
    "Vulpes",
    "Wallabia",
}

REPTILE_GENERA = {
    "Chelonia",
    "Chelonoidis",
    "Eretmochelys",
    "Gopherus",
    "Testudinidae",
}

MARINE_GENERA = {
    "Balaenoptera",
    "Halichoerus",
    "Manta",
    "Mobula",
    "Physeter",
}

_register_genera(GROUP_AVIAN, sorted(AVIAN_GENERA))
_register_genera(GROUP_TERRESTRIAL, sorted(TERRESTRIAL_GENERA))
_register_genera(GROUP_REPTILES, sorted(REPTILE_GENERA))
_register_genera(GROUP_MARINE, sorted(MARINE_GENERA))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add a broad taxonomic group column (avian, terrestrial mammals, "
            "reptiles, marine, humans/domestic) to the Movebank metadata CSV."
        )
    )
    parser.add_argument(
        "--input",
        default="mbr_studies_metadata.csv",
        help="Metadata CSV to read (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional output path. When omitted, overwrites the input after "
            "backing it up to *.bak."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity (default: %(default)s).",
    )
    return parser.parse_args()


def _tokenize_species_field(species_field: str) -> List[str]:
    tokens = []
    for raw in (species_field or "").split(";"):
        cleaned = raw.strip()
        if not cleaned:
            continue
        if cleaned.lower() in IGNORED_SPECIES_TOKENS:
            continue
        tokens.append(cleaned)
    return tokens


def _classify_species(species_name: str) -> Optional[str]:
    if species_name in SPECIES_OVERRIDES:
        return SPECIES_OVERRIDES[species_name]
    genus = species_name.split()[0]
    return GENUS_TO_GROUP.get(genus)


def determine_group(species_field: str) -> Tuple[str, Set[str], bool]:
    tokens = _tokenize_species_field(species_field)
    if not tokens:
        return GROUP_UNKNOWN, set(), False

    groups: Set[str] = set()
    unmapped: Set[str] = set()
    for token in tokens:
        group = _classify_species(token)
        if group:
            groups.add(group)
        else:
            unmapped.add(token)

    if not groups:
        return GROUP_UNKNOWN, unmapped, False
    if len(groups) == 1:
        return groups.pop(), unmapped, False
    return GROUP_UNKNOWN, unmapped, True


def load_rows(path: Path) -> Tuple[List[dict], List[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        if reader.fieldnames is None:
            raise ValueError(f"No header detected in {path}")
        return rows, list(reader.fieldnames)


def write_rows(path: Path, fieldnames: Sequence[str], rows: Sequence[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def process_file(input_path: Path, output_path: Path) -> None:
    rows, fieldnames = load_rows(input_path)
    if COLUMN_NAME not in fieldnames:
        fieldnames = list(fieldnames) + [COLUMN_NAME]

    unmapped_species: Set[str] = set()
    mixed_rows = 0
    updated_rows = 0

    for row in rows:
        species_field = row.get("study_species", "")
        group, unmapped, mixed = determine_group(species_field)
        if unmapped:
            unmapped_species.update(unmapped)
        if mixed:
            mixed_rows += 1
        if row.get(COLUMN_NAME) != group:
            row[COLUMN_NAME] = group
            updated_rows += 1

    write_rows(output_path, fieldnames, rows)

    logging.info("Wrote %s with %s row updates", output_path, updated_rows)
    if mixed_rows:
        logging.warning(
            "Detected %s studies containing species from multiple groups; "
            "labeled them as Unknown",
            mixed_rows,
        )
    if unmapped_species:
        logging.warning(
            "Encountered %s species without a mapping: %s",
            len(unmapped_species),
            ", ".join(sorted(unmapped_species)),
        )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {input_path}")

    if args.output:
        output_path = Path(args.output)
    else:
        backup_path = input_path.with_suffix(input_path.suffix + ".bak")
        input_path.replace(backup_path)
        logging.info("Backed up original file to %s", backup_path)
        output_path = input_path

    process_file(backup_path if not args.output else input_path, output_path)


if __name__ == "__main__":
    main()
