import hashlib
import logging
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class HetionetDownloadSettings(BaseSettings):
    URL: str = "https://zenodo.org/records/268568/files/dhimmel/hetionet-v1.0.0.zip?download=1"
    CACHE_DIR: Path = Path.home() / ".cache" / "hetionet"
    OVERWRITE: bool = False

    model_config = SettingsConfigDict(env_prefix="HETIONET_")

    @property
    def zip_filename(self) -> str:
        return "hetionet-v1.0.0.zip"

    @property
    def zip_path(self) -> Path:
        return self.CACHE_DIR / self.zip_filename

    @property
    def json_bz2_path(self) -> Path:
        return self.CACHE_DIR / "hetionet-v1.0.json.bz2"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)


def sha256sum(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, destination: Path, overwrite: bool = False) -> Path:
    if destination.exists() and not overwrite:
        logger.info("[CACHE HIT] %s", destination)
        return destination

    logger.info("[DOWNLOAD] %s", url)
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()

    total = int(r.headers.get("content-length", 0))
    tmp = destination.with_suffix(destination.suffix + ".part")

    with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=destination.name) as pbar:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

    tmp.replace(destination)
    logger.info("[DONE] %s", destination)
    return destination


def extract_hetionet_json_bz2(
    zip_path: Path, out_path: Path, overwrite: bool = False
) -> Path:
    if out_path.exists() and not overwrite:
        logger.info("[CACHE HIT] %s", out_path)
        return out_path

    with zipfile.ZipFile(zip_path, "r") as z:
        candidates = [n for n in z.namelist() if n.endswith(
            "hetionet-v1.0.json.bz2") or n.endswith(".json.bz2")]
        if not candidates:
            raise FileNotFoundError(
                "No *.json.bz2 found inside zip "
                "(unexpected Hetionet archive layout).")

        # Prefer exact match if present
        member = next((n for n in candidates if n.endswith(
            "hetionet-v1.0.json.bz2")), candidates[0])
        logger.info("[EXTRACT] %s -> %s", member, out_path)

        tmp = out_path.with_suffix(out_path.suffix + ".part")
        with z.open(member) as src, open(tmp, "wb") as dst:
            for chunk in iter(lambda: src.read(1024 * 1024), b""):
                dst.write(chunk)
        tmp.replace(out_path)

    logger.info("[DONE] %s", out_path)
    return out_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    settings = HetionetDownloadSettings()
    zip_path = download_file(
        settings.URL, settings.zip_path, overwrite=settings.OVERWRITE)
    data_path = extract_hetionet_json_bz2(
        zip_path, settings.json_bz2_path, overwrite=settings.OVERWRITE)

    logger.info("SHA256 (zip): %s", sha256sum(zip_path))
    logger.info("SHA256 (json.bz2): %s", sha256sum(data_path))
