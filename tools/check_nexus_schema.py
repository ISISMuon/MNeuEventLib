#!/usr/bin/env python3
"""
NeXus Output File vs. NXDL XML Schema Comparator

Compares an output histogram NeXus (HDF5) file against an NXDL XML definition file
(defaults to the packaged NXmuonTDh.nxdl.xml).

Features:
1. Identifies missing required groups, datasets, and attributes defined in the XML schema.
2. Validates dataset data types against NeXus schema type definitions (NX_INT, NX_FLOAT, NX_CHAR, etc.).
3. Validates attribute values and units against expected schema enumerations.
4. Returns a structured JSON hierarchy preserving parent container groups for any missing or invalid metadata.
5. Does NOT flag optional groups, datasets, or attributes defined in the schema as unexpected.
6. Prints any groups, datasets, or attributes present in the output file but not defined in the XML schema to the screen.
"""

import sys
import os
import json
import argparse
import h5py

try:
    from .nxdl_schema_parser import NXDLSchemaParser, get_default_xml_path
    from .nexus_schema_comparator import NeXusSchemaComparator
except ImportError:
    from nxdl_schema_parser import NXDLSchemaParser, get_default_xml_path
    from nexus_schema_comparator import NeXusSchemaComparator


def compare_nexus_to_xml(hdf5_path, xml_path=None):
    """
    Compare a NeXus HDF5 histogram output file against an NXDL XML schema definition file.
    :param hdf5_path: the file path to the output histogram NeXus (HDF5) file.
    :param xml_path: the file path to the NXDL XML schema file (defaults to the packaged NXmuonTDh.nxdl.xml if None).
    :returns: structured dictionary containing missing metadata, formatted with parent groups present.
    """
    resolved_xml_path = xml_path if xml_path is not None else get_default_xml_path()
    parser = NXDLSchemaParser(resolved_xml_path)
    comparator = NeXusSchemaComparator(parser)

    with h5py.File(hdf5_path, 'r') as h5_file:
        missing_json = comparator.compare(h5_file)

    return missing_json


def main():
    """
    Parse command-line arguments and run the NeXus to NXDL XML schema comparison.
    """
    parser = argparse.ArgumentParser(
        description="Compare output histogram NeXus file against NXDL XML definition."
    )
    parser.add_argument("hdf5_file", help="Path to the output NeXus/HDF5 histogram file.")
    parser.add_argument(
        "xml_file",
        nargs="?",
        default=None,
        help="Path to the NXDL XML schema file (default: packaged NXmuonTDh.nxdl.xml).",
    )
    parser.add_argument(
        "--json-out",
        "-o",
        help="Optional path to write structured JSON output to file.",
    )

    args = parser.parse_args()
    xml_file = args.xml_file if args.xml_file is not None else get_default_xml_path()

    if not os.path.exists(args.hdf5_file):
        print(f"Error: HDF5 file '{args.hdf5_file}' does not exist.", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(xml_file):
        print(f"Error: XML file '{xml_file}' does not exist.", file=sys.stderr)
        sys.exit(1)

    print(f"Comparing output file '{args.hdf5_file}' against XML schema '{xml_file}'...\n")
    missing_json = compare_nexus_to_xml(args.hdf5_file, xml_file)

    json_str = json.dumps(missing_json, indent=2)
    print("\n--- Structured Missing Metadata JSON ---")
    print(json_str)

    if args.json_out:
        with open(args.json_out, "w") as f:
            f.write(json_str)
        print(f"\nSaved structured missing metadata JSON to '{args.json_out}'.")


if __name__ == "__main__":
    main()
