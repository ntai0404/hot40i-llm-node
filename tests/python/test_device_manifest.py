import json

import jsonschema

from host.device_lab.adb import CommandResult, SAFE_PROBES, collect_manifest


class FakeClient:
    serial = "TEST-SERIAL"

    def devices(self):
        return CommandResult(["adb", "devices", "-l"], 0, "TEST-SERIAL device\n", "")

    def probe(self, name: str):
        assert name in SAFE_PROBES
        return CommandResult(["adb", "shell", SAFE_PROBES[name]], 0, f"{name}\n", "")


def test_collect_manifest_matches_schema():
    manifest = collect_manifest(FakeClient())
    schema = json.loads(open("schemas/device_manifest.schema.json", encoding="utf-8").read())
    jsonschema.validate(manifest, schema)
    assert manifest["schema_version"] == 1
    assert manifest["serial"] == "TEST-SERIAL"
