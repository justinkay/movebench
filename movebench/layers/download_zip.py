from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import shutil
import zipfile
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import requests
from tqdm import tqdm

from movebench import paths
from movebench.download_reports import _write_json_atomic
from movebench.study import NAMED_STUDY_SETS

PUBLIC_MANIFEST_URL = "https://data.csail.mit.edu/movebench/movebench_zips_v1_manifest.csv"
_STATE_DIRNAME = ".movebench_zip_download"


class _RangeReader:
    def __init__(
        self,
        *,
        session: requests.Session,
        url: str,
        size: int,
        chunk_size: int = 8 * 1024 * 1024,
        max_cached_chunks: int = 8,
    ) -> None:
        self.session = session
        self.url = str(url)
        self.size = int(size)
        self.chunk_size = int(chunk_size)
        self.max_cached_chunks = int(max(max_cached_chunks, 1))
        self.pos = 0
        self.cache: OrderedDict[int, bytes] = OrderedDict()

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return int(self.pos)

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.pos = int(offset)
        elif whence == io.SEEK_CUR:
            self.pos += int(offset)
        elif whence == io.SEEK_END:
            self.pos = int(self.size) + int(offset)
        else:
            raise ValueError(f"Unsupported whence: {whence}")
        if self.pos < 0:
            raise ValueError(f"Negative seek position: {self.pos}")
        self.pos = min(self.pos, self.size)
        return int(self.pos)

    def read(self, n: int = -1) -> bytes:
        if self.pos >= self.size:
            return b""
        if n is None or int(n) < 0:
            end = self.size
        else:
            end = min(self.pos + int(n), self.size)
        out = bytearray()
        while self.pos < end:
            chunk_index = self.pos // self.chunk_size
            chunk = self._get_chunk(chunk_index)
            chunk_start = chunk_index * self.chunk_size
            rel_start = self.pos - chunk_start
            take = min(end - self.pos, len(chunk) - rel_start)
            out.extend(chunk[rel_start : rel_start + take])
            self.pos += int(take)
        return bytes(out)

    def close(self) -> None:
        self.cache.clear()

    def _get_chunk(self, chunk_index: int) -> bytes:
        cached = self.cache.get(int(chunk_index))
        if cached is not None:
            self.cache.move_to_end(int(chunk_index))
            return cached
        start = int(chunk_index) * self.chunk_size
        end = min(start + self.chunk_size, self.size) - 1
        response = self.session.get(
            self.url,
            headers={"Range": f"bytes={start}-{end}"},
            timeout=120,
        )
        response.raise_for_status()
        chunk = bytes(response.content)
        if len(chunk) != end - start + 1:
            raise RuntimeError(
                f"Expected {end - start + 1} bytes from {self.url} range "
                f"{start}-{end}, got {len(chunk)}"
            )
        self.cache[int(chunk_index)] = chunk
        while len(self.cache) > self.max_cached_chunks:
            self.cache.popitem(last=False)
        return chunk


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _format_bytes(size_bytes: int) -> str:
    size = float(size_bytes)
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{int(size_bytes)} B"


def resolve_public_study_selection(
    selection: str | list[str] | None,
    available_studies: list[str],
) -> list[str]:
    if selection is None:
        return list(available_studies)
    values = [selection] if isinstance(selection, str) else list(selection)
    out: list[str] = []
    seen: set[str] = set()
    available_set = set(available_studies)
    for value in values:
        for part in str(value).split(","):
            token = part.strip()
            if not token:
                continue
            key = token.lower()
            if key == "all":
                candidates = available_studies
            elif key in NAMED_STUDY_SETS:
                candidates = [
                    study_id
                    for study_id in NAMED_STUDY_SETS[key]
                    if study_id in available_set
                ]
            else:
                candidates = [token]
            for study_id in candidates:
                if study_id in seen:
                    continue
                seen.add(study_id)
                out.append(study_id)
    return out


def fetch_public_manifest(
    *,
    session: requests.Session,
    manifest_url: str = PUBLIC_MANIFEST_URL,
) -> list[dict[str, object]]:
    response = session.get(manifest_url, timeout=120)
    response.raise_for_status()
    rows = csv.DictReader(io.StringIO(response.text))
    return [
        {
            "study_id": str(row["study_id"]),
            "file_name": str(row["file_name"]),
            "content_url": str(row["content_url"]),
            "content_size_bytes": int(row["content_size_bytes"]),
        }
        for row in rows
    ]


def _state_paths(study_dir: Path) -> tuple[Path, Path]:
    state_dir = Path(study_dir) / _STATE_DIRNAME
    return state_dir / "state.json", state_dir / "report.json"


def _archive_signature(record: dict[str, object], response: requests.Response) -> dict[str, object]:
    content_length = int(
        response.headers.get("Content-Length") or int(record["content_size_bytes"])
    )
    etag = str(response.headers.get("ETag", "") or "")
    last_modified = str(response.headers.get("Last-Modified", "") or "")
    payload = {
        "study_id": str(record["study_id"]),
        "content_url": str(record["content_url"]),
        "content_length": int(content_length),
        "etag": etag,
        "last_modified": last_modified,
    }
    signature = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()[:16]
    payload["signature"] = signature
    return payload


def _load_entries(session: requests.Session, archive: dict[str, object]) -> list[dict[str, object]]:
    prefix = f"{archive['study_id']}/"
    reader = _RangeReader(
        session=session,
        url=str(archive["content_url"]),
        size=int(archive["content_length"]),
    )
    try:
        with zipfile.ZipFile(reader) as zf:
            entries: list[dict[str, object]] = []
            for info in zf.infolist():
                if info.is_dir():
                    continue
                if info.compress_type != zipfile.ZIP_STORED:
                    raise ValueError(
                        f"Expected stored zip entries for {archive['study_id']}, "
                        f"got compression method {info.compress_type} for {info.filename!r}"
                    )
                if not str(info.filename).startswith(prefix):
                    raise ValueError(
                        f"Archive member {info.filename!r} does not live under {prefix!r}"
                    )
                relative_path = str(info.filename)[len(prefix) :]
                if not relative_path:
                    continue
                if ".." in Path(relative_path).parts:
                    raise ValueError(f"Unsafe archive path: {info.filename!r}")
                entries.append(
                    {
                        "archive_path": str(info.filename),
                        "relative_path": relative_path,
                        "size": int(info.file_size),
                        "mode": int((info.external_attr >> 16) & 0o777),
                        "header_offset": int(info.header_offset),
                    }
                )
    finally:
        reader.close()
    entries.sort(key=lambda entry: int(entry["header_offset"]))
    return entries


def _entry_exists(study_dir: Path, entry: dict[str, object]) -> bool:
    path = Path(study_dir) / str(entry["relative_path"])
    return path.exists() and path.is_file() and path.stat().st_size == int(entry["size"])


def _cleanup_old_entries(study_dir: Path, entries: list[dict[str, object]]) -> None:
    for entry in entries:
        path = Path(study_dir) / str(entry["relative_path"])
        if path.exists():
            if path.is_file():
                path.unlink()
            else:
                shutil.rmtree(path)
        parent = path.parent
        while parent != Path(study_dir):
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent


def _extract_entries(
    study_dir: Path,
    session: requests.Session,
    archive: dict[str, object],
    pending: list[dict[str, object]],
) -> tuple[int, list[dict[str, object]]]:
    if not pending:
        return 0, []
    reader = _RangeReader(
        session=session,
        url=str(archive["content_url"]),
        size=int(archive["content_length"]),
    )
    downloaded = 0
    failures: list[dict[str, object]] = []
    try:
        with zipfile.ZipFile(reader) as zf:
            info_by_name = {str(info.filename): info for info in zf.infolist()}
            with tqdm(
                total=sum(int(entry["size"]) for entry in pending),
                desc=str(archive["study_id"]),
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                dynamic_ncols=True,
            ) as pbar:
                for entry in pending:
                    info = info_by_name.get(str(entry["archive_path"]))
                    if info is None:
                        failures.append(
                            {
                                "relative_path": str(entry["relative_path"]),
                                "size": int(entry["size"]),
                                "error": "missing_archive_entry",
                            }
                        )
                        continue
                    out_path = Path(study_dir) / str(entry["relative_path"])
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    os.chmod(out_path.parent, 0o775)
                    tmp_path = out_path.with_name(f"{out_path.name}.part")
                    if tmp_path.exists():
                        tmp_path.unlink()
                    try:
                        with zf.open(info) as src, open(tmp_path, "wb") as dst:
                            while True:
                                chunk = src.read(1024 * 1024)
                                if not chunk:
                                    break
                                dst.write(chunk)
                                pbar.update(len(chunk))
                        mode = int(entry["mode"]) if int(entry["mode"]) > 0 else 0o664
                        os.chmod(tmp_path, mode | 0o660)
                        tmp_path.replace(out_path)
                        os.chmod(out_path, mode | 0o660)
                        downloaded += 1
                    except Exception as e:
                        if tmp_path.exists():
                            tmp_path.unlink()
                        failures.append(
                            {
                                "relative_path": str(entry["relative_path"]),
                                "size": int(entry["size"]),
                                "error": str(e),
                            }
                        )
    finally:
        reader.close()
    return downloaded, failures


def install_public_study(
    *,
    study_dir: Path,
    session: requests.Session,
    record: dict[str, object],
) -> dict[str, object]:
    study_dir = Path(study_dir)
    study_dir.mkdir(parents=True, exist_ok=True)
    state_path, report_path = _state_paths(study_dir)
    state_dir = state_path.parent
    state_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(state_dir, 0o775)

    response = session.head(str(record["content_url"]), allow_redirects=True, timeout=120)
    response.raise_for_status()
    archive = _archive_signature(record, response)

    state = {}
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if not isinstance(state, dict):
            raise ValueError(f"Invalid zip download state: {state_path}")

    if (
        state.get("archive", {}).get("signature") == archive["signature"]
        and state.get("status") == "ok"
    ):
        return {
            "status": "ok",
            "downloaded": 0,
            "existing": len(state.get("entries", [])),
            "failed": 0,
            "report_path": None,
            "signature": str(archive["signature"]),
        }

    entries = _load_entries(session, archive)
    if (
        state.get("archive", {}).get("signature")
        and state.get("archive", {}).get("signature") != archive["signature"]
    ):
        _cleanup_old_entries(study_dir, list(state.get("entries", [])))

    pending = [entry for entry in entries if not _entry_exists(study_dir, entry)]
    if (
        state.get("archive", {}).get("signature") == archive["signature"]
        and state.get("status") == "failed"
        and report_path.exists()
    ):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        failure_paths = {
            str(item.get("relative_path", "")).strip()
            for item in report.get("failures", [])
            if isinstance(item, dict)
        }
        pending = [
            entry
            for entry in entries
            if str(entry["relative_path"]) in failure_paths
            and not _entry_exists(study_dir, entry)
        ]

    started_at = _utc_now_iso()
    state = {
        "schema_version": "1",
        "archive": archive,
        "entries": entries,
        "status": "running",
        "started_at": started_at,
    }
    _write_json_atomic(state_path, state)

    downloaded, failures = _extract_entries(study_dir, session, archive, pending)
    completed_at = _utc_now_iso()
    existing = max(len(entries) - downloaded - len(failures), 0)
    status = "failed" if failures else "ok"
    report = {
        "schema_version": "1",
        "study_id": str(record["study_id"]),
        "archive": archive,
        "requested_file_count": len(entries),
        "started_at": started_at,
        "completed_at": completed_at,
        "status": status,
        "downloaded_files": int(downloaded),
        "existing_files": int(existing),
        "failed_files": len(failures),
        "failures": failures,
    }
    _write_json_atomic(report_path, report)
    state = {
        "schema_version": "1",
        "archive": archive,
        "entries": entries,
        "status": status,
        "started_at": started_at,
        "completed_at": completed_at,
    }
    _write_json_atomic(state_path, state)
    return {
        "status": status,
        "downloaded": int(downloaded),
        "existing": int(existing),
        "failed": len(failures),
        "report_path": str(report_path),
        "signature": str(archive["signature"]),
    }


def download_public_studies(*, studies: str | list[str] | None) -> int:
    with requests.Session() as session:
        records = fetch_public_manifest(session=session)
        record_by_study = {
            str(record["study_id"]): record
            for record in records
        }
        selected = resolve_public_study_selection(
            studies,
            available_studies=sorted(record_by_study.keys()),
        )
        total_size_bytes = sum(
            int(record_by_study[study_id]["content_size_bytes"])
            for study_id in selected
            if study_id in record_by_study
        )
        print(f"Studies to download: {len(selected)}")
        print(f"Public manifest: {PUBLIC_MANIFEST_URL}")
        print(f"Selected size: {_format_bytes(total_size_bytes)}")
        print(f"Destination root: {paths.studies_root()}")
        print()
        for i, study_id in enumerate(selected, 1):
            if study_id not in record_by_study:
                print(f"[{i}/{len(selected)}] SKIP {study_id} (not found)")
                continue
            print(f"[{i}/{len(selected)}] Downloading {study_id}...")
            try:
                result = install_public_study(
                    study_dir=paths.study_dir(study_id),
                    session=session,
                    record=record_by_study[study_id],
                )
                failed = int(result["failed"])
                status = "WARN" if failed > 0 else "OK"
                line = (
                    f"{status} {study_id}: {int(result['downloaded'])} downloaded, "
                    f"{int(result['existing'])} existing"
                )
                if failed > 0:
                    line += f", {failed} failures, report={result['report_path']}"
                print(f"[{i}/{len(selected)}] {line}")
            except Exception as e:
                print(f"[{i}/{len(selected)}] ERROR {study_id}: {e}")
    print("\nDone.")
    return 0
