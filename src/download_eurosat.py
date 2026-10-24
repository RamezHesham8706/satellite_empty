import argparse
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

URLS = {
    "rgb": "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1",
    "ms": "https://zenodo.org/records/7711810/files/EuroSAT_MS.zip?download=1",
}


def download(kind: str, output_dir=None, url=None):
    kind = kind.lower()
    if kind not in URLS:
        raise ValueError(f"Unsupported EuroSAT kind: {kind!r}. Expected one of {sorted(URLS)}")

    target_dir = Path(output_dir) if output_dir is not None else RAW / f"EuroSAT_{kind.upper()}"
    target_dir.mkdir(parents=True, exist_ok=True)

    zip_path = target_dir.parent / f"EuroSAT_{kind.upper()}.zip"
    urlretrieve(url or URLS[kind], str(zip_path))

    with zipfile.ZipFile(zip_path) as z:
        z.extractall(target_dir)

    return target_dir


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Download and extract the EuroSAT dataset.")
    p.add_argument("--kind", choices=("rgb", "ms"), default="ms", help="Dataset variant to download.")
    p.add_argument("--ms", action="store_true", help="Backward-compatible alias for --kind ms.")
    p.add_argument("--output_dir", type=Path, default=None, help="Directory to extract into. Defaults to data/raw/EuroSAT_RGB or EuroSAT_MS.")
    p.add_argument("--download_url", default=None, help="Optional override URL for the dataset archive.")
    args = p.parse_args()
    kind = "ms" if args.ms else args.kind
    download(kind, output_dir=args.output_dir, url=args.download_url)
