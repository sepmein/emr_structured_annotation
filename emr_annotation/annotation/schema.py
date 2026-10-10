"""Project Schema readers with explicit, distinct export and review contracts."""
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET


def read_label_groups(path: Path) -> dict[str, list[str]]:
    """Only entity Labels controls count; Choices/relations are separate tasks."""
    groups: dict[str, list[str]] = {}
    seen: set[str] = set()
    for control in ET.parse(path).getroot().iter():
        if control.tag.rsplit("}", 1)[-1] != "Labels":
            continue
        name = control.attrib.get("name", "")
        if not name or name in groups:
            raise ValueError("Entity controls require distinct nonempty names")
        labels = []
        for child in control.iter():
            if child.tag.rsplit("}", 1)[-1] != "Label":
                continue
            label = child.attrib.get("value", "")
            if not label or label in seen:
                raise ValueError("Entity label values must be nonempty and unique")
            seen.add(label)
            labels.append(label)
        groups[name] = labels
    if not seen:
        raise ValueError("No entity Labels found in label configuration")
    return groups


def load_export_schema(path: Path) -> dict[str, Any]:
    groups = read_label_groups(path)
    root = ET.parse(path).getroot()
    nodes = list(root.iter())
    texts = [node for node in nodes if node.tag.rsplit("}", 1)[-1] == "Text"]
    if len(texts) != 1 or not texts[0].get("value", "").startswith("$"):
        raise ValueError("This converter requires one Text control backed by a task data field")
    choices = {}
    for node in nodes:
        if node.tag.rsplit("}", 1)[-1] != "Choices":
            continue
        name = node.get("name")
        if not name or name in choices:
            raise ValueError("Choices controls require distinct nonempty names")
        value_map = {}
        for child in node:
            if child.tag.rsplit("}", 1)[-1] != "Choice":
                continue
            display = child.get("value")
            if not display:
                raise ValueError("Choice values must be nonempty")
            for exported in (display, child.get("alias")):
                if exported:
                    if exported in value_map and value_map[exported] != display:
                        raise ValueError("Ambiguous Choice value/alias in a control")
                    value_map[exported] = display
        choices[name] = {
            "values": set(value_map.values()),
            "value_map": value_map,
            "per_region": node.get("perRegion", "false").lower() == "true",
            "required": node.get("required", "false").lower() == "true",
        }
    return {
        "groups": groups,
        "entity_choices": {node.get("name"): node.get("choice", "single")
                           for node in nodes if node.tag.rsplit("}", 1)[-1] == "Labels"},
        "choices": choices,
        "text_control": texts[0].get("name"),
        "text_field": texts[0].get("value")[1:],
        "relations": {n.get("value") for n in nodes if n.tag.rsplit("}", 1)[-1] == "Relation"},
    }


def load_double_annotation_schema(path):
    root = ET.parse(path).getroot()
    controls = {x.get('name'): [l.get('value') for l in x.iter('Label')]
                for x in root.iter('Labels')}
    fields = {}
    for x in root.iter('Choices'):
        values = {c.get('alias', c.get('value')): c.get('value') for c in x.iter('Choice')}
        aliases = {c.get('value'): c.get('alias', c.get('value')) for c in x.iter('Choice')}
        fields[x.get('name')] = dict(values=values, aliases=aliases,
                                    scope=x.get('whenLabelValue', '').split(','),
                                    required=x.get('required') == 'true',
                                    per_region=x.get('perRegion') == 'true')
    return dict(controls=controls, labels=[v for vs in controls.values() for v in vs],
                fields=fields, relations=[x.get('value') for x in root.iter('Relation')])
