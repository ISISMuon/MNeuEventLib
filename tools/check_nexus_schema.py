#!/usr/bin/env python3
"""
NeXus Output File vs. NXDL XML Schema Comparator

Compares an output histogram NeXus (HDF5) file against an NXDL XML definition file
(e.g., NXmuonTD.nxdl.xml).

Features:
1. Identifies missing required groups, datasets, and attributes defined in the XML schema.
2. Validates dataset data types against NeXus schema type definitions (NX_INT, NX_FLOAT, NX_CHAR, etc.).
3. Validates attribute values and units against expected schema enumerations.
4. Returns a structured JSON hierarchy preserving parent container groups for any missing or invalid metadata.
5. Does NOT flag optional groups, datasets, or attributes defined in the schema or NeXus base classes as unexpected.
6. Prints any groups, datasets, or attributes present in the output file but not defined anywhere in the XML schema or NeXus base classes to the screen.
"""

import sys
import os
import json
import argparse
import xml.etree.ElementTree as ET
import h5py
import numpy as np


# Common optional fields and subgroups defined in NeXus base classes
BASE_CLASS_OPTIONAL_FIELDS = {
    'NXentry': {
        'script_name', 'seci_config', 'name', 'monitor_events_not_saved',
        'total_discarded_bad_frame_events', 'total_discarded_bin0_events',
        'total_discarded_spec_events', 'total_uncounted_counts',
        'proton_charge_raw', 'measurement_first_run', 'measurement_id',
        'measurement_label', 'measurement_subid', 'measurement_type',
        'IDF_version', 'definition_local', 'program_name', 'beamline', 'notes',
        'run_cycle', 'duration', 'collection_time', 'total_counts', 'good_frames',
        'raw_frames', 'proton_charge', 'title', 'start_time', 'end_time', 'run_number',
        'experiment_identifier', 'definition', 'discarded_good_frames',
        'discarded_raw_frames', 'good_duration', 'first_run', 'id', 'label',
        'subid', 'type'
    },
    'NXsample': {
        'description', 'distance', 'height', 'width', 'id', 'shape', 'thickness',
        'mass', 'density', 'temperature', 'magnetic_field', 'name', 'type',
        'situation', 'flypast', 'magnetic_field_state', 'sample_component'
    },
    'NXperiod': {
        'highest_used', 'proton_charge', 'proton_charge_raw', 'total_counts',
        'output', 'labels', 'raw_frames', 'good_frames', 'good_frames_daq', 'sequences',
        'number', 'type', 'frames_requested'
    },
    'NXdetector': {
        'orientation', 'corrected_time', 'resolution', 'spectrum_labels',
        'period_index', 'period_labels', 'detector_index', 'detector_count',
        'detector_list', 'crate', 'slot', 'input', 'type', 'voltage',
        'discriminator', 'threshold', 'output_width', 'solid_angle',
        'calibrated_angles', 'dead_time', 'time_zero', 'first_good_time',
        'last_good_time', 'name', 'azimuthal_angle', 'polar_angle', 'distance',
        'delt', 'source_detector_distance', 'alpha', 'counts', 'grouping',
        'raw_time', 'spectrum_index'
    },
    'NXdata': {
        'counts', 'raw_time', 'corrected_time', 'spectrum_index', 'spectrum_labels',
        'period_index', 'period_output', 'period_labels', 'alpha', 'dead_time',
        'grouping', 'time_zero', 'name', 'resolution'
    },
    'NXdae': {
        'type', 'interface', 'detector_table_file', 'spectra_table_file',
        'wiring_table_file', 'period_index'
    },
    'NXuser': {
        'name', 'affiliation', 'address', 'telephone_number', 'email', 'role', 'facility_user_id'
    },
    'NXgeometry': {
        'name', 'description', 'component_index'
    },
    'NXlog': {
        'value', 'time', 'read_control', 'set_control', 'setpoint', 'setpoint_spread',
        'value_spread', 'value_log', 'vi_name', 'description', 'duration', 'target',
        'units', 'average_value', 'minimum_value', 'maximum_value'
    }
}

BASE_CLASS_OPTIONAL_GROUPS = {
    'NXentry': {
        'framelog', 'measurement', 'runlog', 'selog', 'periods', 'CHARACTERIZATION',
        'uif', 'sample', 'instrument', 'user_1', 'USER', 'DETECTOR'
    },
    'NXsample': {
        'geometry', 'temperature_log', 'magnetic_field_log', 'TEMPERATURE_ENV',
        'TEMPERATURE_LOG', 'MAGNETIC_FIELD_ENV', 'MAGNETIC_FIELD_LOG'
    },
    'NXperiod': {
        'counts'
    },
    'NXdetector': {
        'geometry', 'grouping'
    },
    'NXdae': {
        'vetos', 'time_channels_1'
    }
}


def _get_attr_str_val(raw_val):
    """Converts HDF5 attribute or dataset scalar value to clean string."""
    if isinstance(raw_val, (bytes, bytearray)):
        return raw_val.decode('utf-8', errors='ignore').strip()
    if hasattr(raw_val, 'item'):
        val_item = raw_val.item()
        if isinstance(val_item, (bytes, bytearray)):
            return val_item.decode('utf-8', errors='ignore').strip()
        return str(val_item).strip()
    if isinstance(raw_val, (list, tuple, np.ndarray)):
        if len(raw_val) > 0:
            return _get_attr_str_val(raw_val[0])
        return ""
    return str(raw_val).strip()


def _check_dtype_match(actual_dtype, expected_nexus_type):
    """Checks if h5py dataset dtype matches NeXus XML type definition."""
    if not expected_nexus_type:
        return True

    kind = actual_dtype.kind
    expected_upper = expected_nexus_type.upper()

    if expected_upper in ['NX_INT', 'NX_POSINT', 'NX_UINT']:
        return kind in ['i', 'u']
    elif expected_upper in ['NX_FLOAT']:
        return kind in ['f']
    elif expected_upper in ['NX_NUMBER']:
        return kind in ['i', 'u', 'f']
    elif expected_upper in ['NX_CHAR', 'ISO8601', 'NX_DATE_TIME']:
        return kind in ['S', 'U', 'O'] or h5py.check_string_dtype(actual_dtype) is not None
    elif expected_upper in ['NX_BINARY', 'NX_BOOLEAN']:
        return kind in ['b', 'V', 'S', 'i', 'u']
    return True


class NXDLSchemaParser:
    """Parses an NXDL XML schema file into structured group specifications."""

    def __init__(self, xml_path):
        self.xml_path = xml_path
        self.type_defs = {}
        self.entry_spec = {}
        self._parse()

    def _parse(self):
        tree = ET.parse(self.xml_path)
        root = tree.getroot()

        # Remove XML namespace prefixes
        for elem in root.iter():
            if '}' in elem.tag:
                elem.tag = elem.tag.split('}', 1)[1]

        # Collect standalone group type definitions (e.g., <group type="NXsample">)
        for g in root.findall('group'):
            gtype = g.attrib.get('type')
            if gtype and 'name' not in g.attrib:
                self.type_defs[gtype] = g

        # Resolve top-level NXentry group (e.g., <group name="raw_data_1" type="NXentry">)
        entry_elem = None
        for g in root.findall('group'):
            if g.attrib.get('name') == 'raw_data_1' or g.attrib.get('type') == 'NXentry':
                entry_elem = g
                break

        if entry_elem is not None:
            self.entry_spec = self._resolve_group_spec(entry_elem)

    def _resolve_group_spec(self, group_elem):
        gtype = group_elem.attrib.get('type', 'NXobject')
        spec = {
            'type': gtype,
            'attributes': {},
            'fields': {},
            'groups': {}
        }

        # Inherit definitions from base class group type if present
        if gtype in self.type_defs and self.type_defs[gtype] != group_elem:
            parent_spec = self._resolve_group_spec(self.type_defs[gtype])
            spec['attributes'].update(parent_spec['attributes'])
            spec['fields'].update(parent_spec['fields'])
            spec['groups'].update(parent_spec['groups'])

        # Group attributes
        for attr in group_elem.findall('attribute'):
            aname = attr.attrib.get('name')
            if aname:
                enums = [item.attrib.get('value') for item in attr.findall('enumeration/item') if item.attrib.get('value')]
                spec['attributes'][aname] = {
                    'min_occurs': attr.attrib.get('minOccurs', '0'),
                    'type': attr.attrib.get('type', 'NX_CHAR'),
                    'enumeration': enums
                }

        # Fields (Datasets)
        for field in group_elem.findall('field'):
            fname = field.attrib.get('name')
            if not fname:
                continue
            ftype = field.attrib.get('type', 'NX_CHAR')
            field_enums = [item.attrib.get('value') for item in field.findall('enumeration/item') if item.attrib.get('value')]
            f_attrs = {}
            for fattr in field.findall('attribute'):
                faname = fattr.attrib.get('name')
                if faname:
                    enums = [item.attrib.get('value') for item in fattr.findall('enumeration/item') if item.attrib.get('value')]
                    f_attrs[faname] = {
                        'min_occurs': fattr.attrib.get('minOccurs', '0'),
                        'type': fattr.attrib.get('type', 'NX_CHAR'),
                        'enumeration': enums
                    }
            spec['fields'][fname] = {
                'type': ftype,
                'name_type': field.attrib.get('nameType'),
                'min_occurs': field.attrib.get('minOccurs', '0'),
                'enumeration': field_enums,
                'attributes': f_attrs
            }

        # Child Groups
        for grp in group_elem.findall('group'):
            gname = grp.attrib.get('name')
            sub_type = grp.attrib.get('type')
            sub_spec = self._resolve_group_spec(grp)
            sub_spec['type'] = sub_type or sub_spec.get('type', 'NXobject')
            
            # Store subgroup spec
            key_name = gname if gname else sub_type
            if key_name:
                spec['groups'][key_name] = {
                    'name': gname,
                    'type': sub_type,
                    'name_type': grp.attrib.get('nameType'),
                    'min_occurs': grp.attrib.get('minOccurs', '0'),
                    'spec': sub_spec
                }

        return spec


class NeXusSchemaComparator:
    """Compares an HDF5 NeXus file against a parsed NXDL schema spec."""

    def __init__(self, schema_parser):
        self.spec = schema_parser.entry_spec

    def compare(self, h5_file):
        """
        Compare the HDF5 file against the XML schema.
        Does NOT flag optional items as unexpected. Returns structured JSON dict of missing or invalid items.
        """
        missing_result = {}

        # Locate raw_data_1 or top-level NXentry group in HDF5 file
        entry_key = 'raw_data_1'
        if entry_key not in h5_file:
            for key in h5_file.keys():
                if isinstance(h5_file[key], h5py.Group):
                    entry_key = key
                    break

        if entry_key in h5_file:
            raw_group = h5_file[entry_key]
            entry_missing = {}
            self._compare_group(f'/{entry_key}', raw_group, self.spec, entry_missing)
            if entry_missing:
                missing_result[entry_key] = entry_missing
        else:
            missing_result['missing_groups'] = ['raw_data_1']

        return missing_result

    def _match_field(self, name, fields_spec, parent_gtype=None):
        # 1. Exact match in parsed XML spec
        if name in fields_spec:
            return name, fields_spec[name]

        # 2. Match pattern fields in XML spec (*NAME, *_VALUE, nameType=any)
        for fname, fspec in fields_spec.items():
            ntype = fspec.get('name_type')
            if fname.endswith('NAME') and name.lower().startswith(fname[:-4].lower()):
                return fname, fspec
            if fname.endswith('_VALUE') and name.lower().startswith(fname[:-6].lower()):
                return fname, fspec
            if ntype == 'any' and (name.lower() == fname.lower() or name.lower().startswith(fname.lower())):
                return fname, fspec

        # 3. Match optional fields across NeXus base classes
        for bclass_fields in BASE_CLASS_OPTIONAL_FIELDS.values():
            if name in bclass_fields:
                return name, {'name_type': 'any', 'min_occurs': '0', 'attributes': {}}

        # 4. Any subgroup / base class container allows optional datasets present in standard NeXus
        if parent_gtype in ['NXlog', 'NXselog', 'NXrunlog', 'NXuif', 'NXobject', 'NXgeometry', 'NXdae', 'NXsample', 'NXentry'] or 'log' in (parent_gtype or '').lower() or parent_gtype == 'vetos':
            return name, {'name_type': 'any', 'min_occurs': '0', 'attributes': {}}

        # 5. Wildcard dataset fields (dataset_*, measurement_*, user_table*, time_channels_*)
        if any(name.startswith(prefix) for prefix in ['dataset_', 'measurement_', 'user_table', 'time_channels_']):
            return name, {'name_type': 'any', 'min_occurs': '0', 'attributes': {}}

        return None, None

    def _match_group(self, name, groups_spec, parent_gtype=None):
        # 1. Exact match in parsed XML spec
        if name in groups_spec:
            return name, groups_spec[name]

        # 2. Match pattern groups by specific prefix
        name_lower = name.lower()
        for gname, gspec in groups_spec.items():
            gname_lower = gname.lower()
            if gname_lower == 'detector' and name_lower.startswith('detector'):
                return gname, gspec
            if gname_lower == 'user' and (name_lower.startswith('user') or name_lower == 'user_1'):
                return gname, gspec
            if gname_lower == 'characterization' and 'characterization' in name_lower:
                return gname, gspec
            if gname_lower.startswith('temperature_') and name_lower.startswith('temperature_'):
                return gname, gspec
            if gname_lower.startswith('magnetic_field_') and name_lower.startswith('magnetic_field_'):
                return gname, gspec
            if gname_lower == 'component' and name_lower.startswith('component'):
                return gname, gspec

        # 3. Facility-specific container groups (NXselog, NXrunlog, NXuif) allow arbitrary log subgroups (NXlog)
        if parent_gtype in ['NXselog', 'NXrunlog', 'NXuif', 'NXlog'] or 'log' in (parent_gtype or '').lower():
            return name, {
                'type': 'NXlog',
                'min_occurs': '0',
                'spec': {
                    'type': 'NXlog',
                    'attributes': {},
                    'fields': {},
                    'groups': {}
                }
            }

        # 4. Match optional groups in NeXus base classes
        if parent_gtype in BASE_CLASS_OPTIONAL_GROUPS and name in BASE_CLASS_OPTIONAL_GROUPS[parent_gtype]:
            sub_type = 'NXobject'
            if name in ['runlog', 'selog']:
                sub_type = 'NX' + name
            elif name == 'periods':
                sub_type = 'NXperiod'
            elif name in ['counts', 'temperature_log', 'magnetic_field_log', 'framelog', 'measurement', 'geometry']:
                sub_type = 'NXlog'
            return name, {
                'type': sub_type,
                'min_occurs': '0',
                'spec': {
                    'type': sub_type,
                    'attributes': {},
                    'fields': {},
                    'groups': {}
                }
            }

        # 5. Wildcard subgroup patterns (time_channels_*, dataset_*, vetos, etc.)
        if name.startswith('time_channels_') or name.startswith('dataset_') or name in ['vetos', 'geometry', 'measurement', 'value_log', 'framelog']:
            sub_type = 'NXlog' if (name.endswith('_log') or name in ['vetos', 'measurement', 'framelog']) else 'NXobject'
            return name, {
                'type': sub_type,
                'min_occurs': '0',
                'spec': {'type': sub_type, 'attributes': {}, 'fields': {}, 'groups': {}}
            }

        # 6. Fallback nameType="any" match in XML spec (e.g. CHARACTERIZATION)
        for gname, gspec in groups_spec.items():
            if gspec.get('name_type') == 'any':
                return gname, gspec

        return None, None

    def _compare_group(self, current_path, h5_group, group_spec, missing_dict):
        current_gtype = group_spec.get('type', 'NXobject')

        # 1. Check unexpected items in HDF5 output file
        for key in h5_group.keys():
            item = h5_group[key]
            item_path = f'{current_path}/{key}'

            if isinstance(item, h5py.Group):
                matched_name, matched_gspec = self._match_group(key, group_spec['groups'], parent_gtype=current_gtype)
                if not matched_gspec:
                    print(f"[UNEXPECTED GROUP] '{item_path}' in output file is not defined in XML schema")
                else:
                    sub_missing = {}
                    sub_type = matched_gspec.get('type') or matched_gspec.get('spec', {}).get('type') or 'NXobject'
                    sub_spec = dict(matched_gspec.get('spec', {}))
                    sub_spec['type'] = sub_type
                    self._compare_group(item_path, item, sub_spec, sub_missing)
                    if sub_missing:
                        missing_dict[key] = sub_missing
            else:  # Dataset
                matched_fname, matched_fspec = self._match_field(key, group_spec['fields'], parent_gtype=current_gtype)
                if not matched_fspec and key not in group_spec['attributes']:
                    print(f"[UNEXPECTED DATASET] '{item_path}' in output file is not defined in XML schema")
                else:
                    # Validate Dataset Data Type & Attributes
                    if matched_fspec:
                        ds_missing_attrs = {}
                        ds_invalid_attrs = {}

                        # Data Type Validation
                        expected_ftype = matched_fspec.get('type')
                        if expected_ftype and not _check_dtype_match(item.dtype, expected_ftype):
                            if key not in missing_dict:
                                missing_dict[key] = {}
                            missing_dict[key]['invalid_data_type'] = {
                                'actual': str(item.dtype),
                                'expected': expected_ftype
                            }

                        # Value Enumeration Validation (for scalar string / int fields)
                        field_enums = matched_fspec.get('enumeration', [])
                        if field_enums and item.shape in [(), (1,)]:
                            actual_val = _get_attr_str_val(item[()]) if item.shape == () else _get_attr_str_val(item[0])
                            if actual_val and actual_val not in field_enums:
                                if key not in missing_dict:
                                    missing_dict[key] = {}
                                missing_dict[key]['invalid_value'] = {
                                    'actual': actual_val,
                                    'expected': field_enums if len(field_enums) > 1 else field_enums[0]
                                }

                        # Attribute Validation (units, signal, axes, etc.)
                        for fattr_name, fattr_info in matched_fspec.get('attributes', {}).items():
                            is_required = (
                                fattr_info.get('min_occurs') in ['1', 1] or
                                fattr_name in ['units', 'signal', 'axes'] or
                                len(fattr_info.get('enumeration', [])) > 0
                            )
                            if fattr_name not in item.attrs:
                                if is_required:
                                    enums = fattr_info.get('enumeration', [])
                                    ds_missing_attrs[fattr_name] = {
                                        'possible_values': enums if enums else None
                                    }
                            else:
                                expected_enums = fattr_info.get('enumeration', [])
                                if expected_enums:
                                    actual_val = _get_attr_str_val(item.attrs[fattr_name])
                                    if actual_val not in expected_enums:
                                        ds_invalid_attrs[fattr_name] = {
                                            'actual': actual_val,
                                            'expected': expected_enums if len(expected_enums) > 1 else expected_enums[0]
                                        }

                        if ds_missing_attrs or ds_invalid_attrs:
                            if key not in missing_dict:
                                missing_dict[key] = {}
                            if ds_missing_attrs:
                                missing_dict[key]['missing_attributes'] = ds_missing_attrs
                            if ds_invalid_attrs:
                                missing_dict[key]['invalid_attributes'] = ds_invalid_attrs

        # 2. Check group attributes
        grp_missing_attrs = {}
        grp_invalid_attrs = {}
        for attr_name, attr_info in group_spec['attributes'].items():
            if attr_info.get('min_occurs') in ['1', 1] or attr_name == 'IDF_version':
                if attr_name not in h5_group.attrs and attr_name not in h5_group:
                    enums = attr_info.get('enumeration', [])
                    grp_missing_attrs[attr_name] = {
                        'possible_values': enums if enums else None
                    }
            if attr_name in h5_group.attrs:
                expected_enums = attr_info.get('enumeration', [])
                if expected_enums:
                    actual_val = _get_attr_str_val(h5_group.attrs[attr_name])
                    if actual_val not in expected_enums:
                        grp_invalid_attrs[attr_name] = {
                            'actual': actual_val,
                            'expected': expected_enums if len(expected_enums) > 1 else expected_enums[0]
                        }

        if grp_missing_attrs:
            missing_dict['missing_attributes'] = grp_missing_attrs
        if grp_invalid_attrs:
            missing_dict['invalid_attributes'] = grp_invalid_attrs

        # 3. Check missing required datasets (fields)
        missing_fields = []
        for fname, fspec in group_spec['fields'].items():
            if fspec.get('min_occurs') in ['1', 1]:
                found = any(self._match_field(key, {fname: fspec}, parent_gtype=current_gtype)[0] is not None for key in h5_group.keys())
                if not found:
                    missing_fields.append(fname)
        if missing_fields:
            missing_dict['missing_fields'] = missing_fields

        # 4. Check missing required subgroups
        missing_subgroups = []
        for gname, gspec in group_spec['groups'].items():
            if gspec.get('min_occurs') in ['1', 1]:
                found = any(self._match_group(key, {gname: gspec}, parent_gtype=current_gtype)[0] is not None for key in h5_group.keys())
                if not found:
                    missing_subgroups.append(gname)
        if missing_subgroups:
            missing_dict['missing_groups'] = missing_subgroups


def compare_nexus_to_xml(hdf5_path, xml_path):
    """
    Compare a NeXus HDF5 histogram output file against an NXDL XML schema definition file.

    Parameters
    ----------
    hdf5_path: str
        Path to the output histogram NeXus (HDF5) file.
    xml_path: str
        Path to the NXDL XML schema file (e.g. NXmuonTD.nxdl.xml).

    Returns
    -------
    dict
        Structured dictionary containing missing metadata, formatted with parent groups present.
    """
    parser = NXDLSchemaParser(xml_path)
    comparator = NeXusSchemaComparator(parser)

    with h5py.File(hdf5_path, 'r') as h5_file:
        missing_json = comparator.compare(h5_file)

    return missing_json


def main():
    parser = argparse.ArgumentParser(
        description="Compare output histogram NeXus file against NXmuonTD NXDL XML definition."
    )
    parser.add_argument("hdf5_file", help="Path to the output NeXus/HDF5 histogram file.")
    parser.add_argument(
        "xml_file",
        nargs="?",
        default="NXmuonTD.nxdl.xml",
        help="Path to the NXDL XML schema file (default: NXmuonTD.nxdl.xml).",
    )
    parser.add_argument(
        "--json-out",
        "-o",
        help="Optional path to write structured JSON output to file.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.hdf5_file):
        print(f"Error: HDF5 file '{args.hdf5_file}' does not exist.", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.xml_file):
        print(f"Error: XML file '{args.xml_file}' does not exist.", file=sys.stderr)
        sys.exit(1)

    print(f"Comparing output file '{args.hdf5_file}' against XML schema '{args.xml_file}'...\n")
    missing_json = compare_nexus_to_xml(args.hdf5_file, args.xml_file)

    json_str = json.dumps(missing_json, indent=2)
    print("\n--- Structured Missing Metadata JSON ---")
    print(json_str)

    if args.json_out:
        with open(args.json_out, "w") as f:
            f.write(json_str)
        print(f"\nSaved structured missing metadata JSON to '{args.json_out}'.")


if __name__ == "__main__":
    main()

