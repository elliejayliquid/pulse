"""YAML loading that rejects duplicate mapping keys.

PyYAML's SafeLoader silently lets the last duplicate key win, so a second
`context:` section in config.yaml can shadow the first with no warning.
This loader raises a YAMLError naming the key and both line numbers.
"""

import yaml


class UniqueKeyLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        seen = {}
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                if key in seen:
                    raise yaml.constructor.ConstructorError(
                        "while constructing a mapping",
                        node.start_mark,
                        f"found duplicate key {key!r} "
                        f"(first defined at line {seen[key].start_mark.line + 1}, "
                        f"duplicated at line {key_node.start_mark.line + 1})",
                        key_node.start_mark,
                    )
                seen[key] = key_node
            except TypeError:
                pass  # unhashable key — SafeLoader rejects it below
        return super().construct_mapping(node, deep=deep)


def safe_load(stream):
    """Like yaml.safe_load, but duplicate mapping keys raise yaml.YAMLError."""
    return yaml.load(stream, UniqueKeyLoader)
