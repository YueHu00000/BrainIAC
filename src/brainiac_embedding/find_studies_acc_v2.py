#!/usr/bin/env python3
"""Query even-year accession batches with durable, per-input-record checkpoints."""

import argparse
from contextlib import contextmanager
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

import pydicom
import pynetdicom

from AET_INFO import CALLING_AET, CALLED_AET, CALLED_HOST, CALLED_PORT


COLUMNS = ["AccessionNumber", "StudyInstanceUID", "StudyDate", "StudyTime",
           "StudyDescription", "match_status"]


def discover(folder):
    batches = []
    for path in folder.iterdir():
        match = re.fullmatch(r"study_accession_([0-9]{6})\.csv", path.name)
        if path.is_file() and match and int(match[1][:4]) % 2 == 0:
            batches.append((match[1], path))
    return sorted(batches)


@contextmanager
def directory_lock(folder):
    """OS releases this lock even after a killed process; keep the lock file."""
    with (folder / ".find_studies_acc_v2.lock").open("a+b") as handle:
        if os.name == "nt":
            import msvcrt
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def save_checkpoint(path, state):
    pending = path.with_suffix(path.suffix + ".new")
    with pending.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(pending, path)


def write_csv_row(handle, values):
    buffer = io.StringIO(newline="")
    csv.writer(buffer).writerow(values)
    handle.write(buffer.getvalue().encode("utf-8"))
    handle.flush()
    os.fsync(handle.fileno())


def publish(temp_csv, output_csv):
    """Copy across filesystems, then expose a complete CSV by atomic rename."""
    staging = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output_csv.parent, prefix="." + output_csv.name + ".",
            suffix=".publishing", delete=False,
        ) as handle:
            staging = Path(handle.name)
            with temp_csv.open("rb") as source:
                shutil.copyfileobj(source, handle)
            handle.flush()
            os.fsync(handle.fileno())
        # All v2 writers hold the output directory lock.
        if output_csv.exists():
            raise FileExistsError(f"Output appeared during processing: {output_csv}")
        os.replace(staging, output_csv)
    finally:
        if staging is not None and staging.exists():
            staging.unlink()


def process_batch(source, output_csv, temp_dir, query, target):
    if output_csv.is_file():
        print(f"SKIP completed: {output_csv.name}", flush=True)
        return
    if output_csv.exists():
        raise FileExistsError(f"Output is not a file: {output_csv}")

    content = source.read_bytes()
    reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig"), newline=""))
    if not reader.fieldnames or "Accession Number" not in reader.fieldnames:
        raise ValueError(f'{source.name}: missing column "Accession Number"')
    accessions = [(row.get("Accession Number") or "").strip() for row in reader]
    fingerprint = hashlib.sha256(content).hexdigest()
    temp_csv = temp_dir / output_csv.name
    checkpoint = temp_dir / (output_csv.name + ".checkpoint.json")

    if checkpoint.exists():
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        if state["source_sha256"] != fingerprint or state["target"] != target:
            raise ValueError(f"Input or VNA configuration changed; preserve and inspect {checkpoint}")
        if not 0 <= state["next_record"] <= len(accessions):
            raise ValueError(f"Invalid input position: {checkpoint}")
        if not temp_csv.exists():
            if state["offset"] != 0:
                raise ValueError(f"Checkpoint data missing: {temp_csv}")
            temp_csv.touch(exist_ok=False)
    else:
        if temp_csv.exists():
            raise ValueError(f"Cannot safely resume CSV without checkpoint: {temp_csv}")
        state = {"source_sha256": fingerprint, "target": target,
                 "next_record": 0, "offset": 0}
        # Save before creating CSV, so an interruption during header writing is recoverable.
        save_checkpoint(checkpoint, state)
        temp_csv.touch(exist_ok=False)

    if temp_csv.stat().st_size < state["offset"]:
        raise ValueError(f"Temporary CSV is shorter than its checkpoint: {temp_csv}")
    print(f"PROCESS {source.name}: resume at input record {state['next_record'] + 1}; "
          f"total={len(accessions)}", flush=True)
    with temp_csv.open("r+b") as handle:
        # Remove only uncommitted output from an interrupted accession, including torn rows.
        handle.truncate(state["offset"])
        handle.seek(state["offset"])
        if state["offset"] == 0:
            write_csv_row(handle, COLUMNS)
            state["offset"] = handle.tell()
            save_checkpoint(checkpoint, state)
        for index in range(state["next_record"], len(accessions)):
            accession = accessions[index]
            if accession:
                for row in query(accession):
                    write_csv_row(handle, [row.get(column, "") for column in COLUMNS])
            state["next_record"] = index + 1
            state["offset"] = handle.tell()
            save_checkpoint(checkpoint, state)
            print(f"{source.name}: {index + 1}/{len(accessions)} input records committed", flush=True)

    publish(temp_csv, output_csv)
    # If interrupted during cleanup, output existence still makes the next run skip safely.
    temp_csv.unlink()
    checkpoint.unlink()
    print(f"DONE {output_csv}", flush=True)


def query_one(association, accession):
    query = pydicom.Dataset()
    query.QueryRetrieveLevel = "STUDY"
    query.AccessionNumber = accession
    for keyword in COLUMNS[1:5]:
        setattr(query, keyword, "")
    matched = False
    for status, identifier in association.send_c_find(
        query, pynetdicom.sop_class.StudyRootQueryRetrieveInformationModelFind,
    ):
        code = getattr(status, "Status", None)
        if code in (0xFF00, 0xFF01) and identifier is not None:
            matched = True
            description = str(getattr(identifier, "StudyDescription", "")).strip()
            if "series " in description.lower():
                continue  # Preserve the original script's default filtering.
            row = {key: str(getattr(identifier, key, "")).strip() for key in COLUMNS[:5]}
            row["AccessionNumber"] = str(getattr(identifier, "AccessionNumber", accession)).strip()
            row["match_status"] = "matched"
            yield row
        elif code == 0x0000:
            if not matched:
                yield {"AccessionNumber": accession, "match_status": "not_found"}
            return
        else:
            raise RuntimeError(f"C-FIND failed (status={code}); current input record will be retried")
    raise RuntimeError("C-FIND ended without final success; current input record will be retried")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--CSV_ACC", type=Path, required=True, help="Input directory")
    parser.add_argument("--output", type=Path, required=True, help="Completed CSV directory")
    parser.add_argument("--TEMP", type=Path, required=True, help="Partial CSV/checkpoint directory")
    parser.add_argument("--host", default=CALLED_HOST)
    parser.add_argument("--port", type=int, default=CALLED_PORT)
    parser.add_argument("--calling-aet", default=CALLING_AET)
    parser.add_argument("--called-aet", default=CALLED_AET)
    args = parser.parse_args()
    if not args.CSV_ACC.is_dir():
        parser.error("--CSV_ACC must be a directory")
    if args.output.resolve() == args.TEMP.resolve():
        parser.error("--output and --TEMP must be different directories")
    args.output.mkdir(parents=True, exist_ok=True)
    args.TEMP.mkdir(parents=True, exist_ok=True)
    batches = discover(args.CSV_ACC)
    print(f"Found {len(batches)} even-year monthly files", flush=True)
    ae = pynetdicom.AE(ae_title=args.calling_aet)
    ae.add_requested_context(pynetdicom.sop_class.StudyRootQueryRetrieveInformationModelFind)
    ae.connection_timeout = 10
    ae.acse_timeout = 30
    ae.dimse_timeout = 120
    association = None

    def query(accession):
        nonlocal association
        if association is None or not association.is_established:
            association = ae.associate(args.host, args.port, ae_title=args.called_aet)
        if not association.is_established:
            raise RuntimeError("VNA association failed; restart to resume")
        yield from query_one(association, accession)

    try:
        with directory_lock(args.TEMP), directory_lock(args.output):
            target = [args.host, args.port, args.calling_aet, args.called_aet]
            for date, source in batches:
                output_csv = args.output / f"study_uid_{date}.csv"
                # Check BEFORE reading this month's input or invoking query().
                if output_csv.is_file():
                    print(f"SKIP completed: {output_csv.name}", flush=True)
                    continue
                process_batch(source, output_csv, args.TEMP, query, target)
    finally:
        if association is not None and association.is_established:
            association.release()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted. Restart the same command to resume.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
