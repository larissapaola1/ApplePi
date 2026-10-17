"""Build a focused local review bundle; this does not certify competition eligibility."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MODULES = ("__init__", "schema", "cube", "basis", "baselines", "history", "features",
           "models", "pipeline", "config", "train", "infer")


def build(output):
    manifest = json.loads((ROOT/"artifacts_official/manifest.json").read_text())
    files = {f"aet/{name}.py": ROOT/f"aet/{name}.py" for name in MODULES}
    for name in ("requirements.txt", "configs/core.yaml", "configs/final.yaml",
                 "docs/METHODOLOGY.md", "docs/PROVENANCE.md", "docs/SUBMISSION.md"):
        files[name] = ROOT/name
    files["README.md"] = ROOT/"docs/SUBMISSION.md"
    artifact_names = ["manifest.json", "history_store.npz", *manifest["model"]["files"]]
    for name in artifact_names:
        if Path(name).name != name:
            raise ValueError("Artifact manifest must contain plain file names")
        files[f"artifacts_official/{name}"] = ROOT/"artifacts_official"/name
    for source in files.values():
        if not source.is_file():
            raise FileNotFoundError(source)
    hashes = {name: hashlib.sha256(source.read_bytes()).hexdigest() for name, source in files.items()}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, source in sorted(files.items()):
            archive.write(source, name)
        archive.writestr("BUNDLE_MANIFEST.json", json.dumps({
            "purpose": "local review; eligibility unresolved; read docs/PROVENANCE.md",
            "files_sha256": hashes}, indent=2)+"\n")
    print(f"Review bundle: {output}; {len(files)} files plus bundle manifest")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT/"dist/aet_review_bundle.zip")
    args = parser.parse_args()
    build(args.out)


if __name__ == "__main__":
    main()
