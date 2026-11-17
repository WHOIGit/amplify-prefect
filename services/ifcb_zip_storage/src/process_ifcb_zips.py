#!/usr/bin/env python3
"""
Process IFCB data directory and upload ZIPs to object storage with multiprocessing.

This script:
1. Finds IFCB filesets (.hdr/.adc/.roi) in the data directory
2. Converts each fileset to ZIP format using bin2zip_stream (in parallel)
3. Uploads ZIP buffers to object store defined by YAML configuration (in parallel)

The ZIP format is the one pyifcb's bin2zip_stream writes, reproduced here on
top of ifcbkit so the output is byte-identical to ZIPs made with pyifcb.
"""
import argparse
import asyncio
import json
import logging
import os
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import BytesIO

import pandas as pd
from ifcbkit.fileset import DEFAULT_EXCLUDE, DEFAULT_INCLUDE, validate_path
from ifcbkit.header import parse_hdr_file
from ifcbkit.identifiers import add_target, bin_timestamp
from ifcbkit.stitching import bin_images
from storage.config_builder import StoreFactory


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def list_filesets(data_dir: str) -> list:
    """
    Find every complete .hdr/.adc/.roi fileset under data_dir.

    Uses the same directory rules as pyifcb and ifcbkit (skip 'skip' and
    'beads' directories, validate paths against the bin name), but skips
    directories it cannot read with a warning, as pyifcb did, instead of
    stopping the whole scan.

    Returns:
        list of dicts: {pid, hdr, adc, roi} paths for each fileset
    """
    def warn(error):
        logger.warning(f"Skipping unreadable directory: {error}")

    filesets = []
    for dirpath, dirnames, filenames in os.walk(data_dir, onerror=warn):
        dirnames[:] = sorted(d for d in dirnames if d not in DEFAULT_EXCLUDE)
        names = set(filenames)
        for name in sorted(filenames):
            pid, ext = name[:-4], name[-4:]
            if ext != '.hdr' or pid + '.adc' not in names or pid + '.roi' not in names:
                continue
            reldir = os.path.relpath(dirpath, data_dir)
            reldir = '' if reldir == '.' else reldir
            if not validate_path(os.path.join(reldir, pid), include=DEFAULT_INCLUDE, exclude=DEFAULT_EXCLUDE):
                continue
            base = os.path.join(dirpath, pid)
            filesets.append({'pid': pid, 'hdr': base + '.hdr', 'adc': base + '.adc', 'roi': base + '.roi'})
    return filesets


def adc_to_csv(adc_bytes: bytes, schema: int) -> str:
    """
    Convert raw ADC bytes to the CSV stored in the ZIP.

    The ADC is parsed by pandas and written back out, which is what pyifcb did,
    so floats are reformatted exactly as before. pandas 3 parses some floats
    differently (e.g. 2.5822532 -> 2.5822532000000002), so pandas must stay
    below 3 for the output to match.
    """
    adc = pd.read_csv(BytesIO(adc_bytes), header=None, index_col=False)
    if schema == 1:
        adc.pop(adc.columns[-1])  # I-style ADC lines end with a comma
    return adc.to_csv(header=False, index=False)


def bin2zip_stream(pid: str, paths: dict) -> BytesIO:
    """
    Build the ZIP for one bin, in the same format and entry order as pyifcb.

    Entries: metadata.json, <pid>_headers.json, <pid>.csv (the ADC), and one
    PNG per ROI, with I-style pairs stitched and infilled.

    Args:
        pid: the bin ID, e.g. D20260709T105717_IFCB188
        paths: dict with 'hdr', 'adc' and 'roi' file paths

    Returns:
        BytesIO positioned at the start of the ZIP
    """
    schema = 1 if pid.startswith('IFCB') else 2  # I-style bins are schema 1
    with open(paths['adc'], 'rb') as f:
        adc_bytes = f.read()
    with open(paths['roi'], 'rb') as f:
        roi_bytes = f.read()

    zip_stream = BytesIO()
    with zipfile.ZipFile(zip_stream, 'w', compression=zipfile.ZIP_STORED) as zf:
        metadata = {'lid': pid, 'schema': schema, 'timestamp': bin_timestamp(pid).isoformat()}
        zf.writestr('metadata.json', json.dumps(metadata))
        zf.writestr(pid + '_headers.json', json.dumps(parse_hdr_file(paths['hdr'])))

        zf.writestr(pid + '.csv', adc_to_csv(adc_bytes, schema))

        images = bin_images(pid, adc_bytes, roi_bytes)
        for target in images:
            buf = BytesIO()
            images[target].save(buf, format='PNG')
            zf.writestr(add_target(pid, target) + '.png', buf.getvalue())
    zip_stream.seek(0)
    return zip_stream


def process_single_bin(fileset: dict, storage_yaml: str) -> tuple:
    """
    Worker function to process a single bin: zip and upload.

    Args:
        fileset: dict with the bin's 'pid' and its 'hdr', 'adc', 'roi' paths
        storage_yaml: Path to storage YAML config

    Returns:
        tuple: (bin_pid, success: bool, error_message: str or None)
    """
    bin_pid = fileset['pid']
    try:
        # Generate ZIP stream
        buffer = bin2zip_stream(bin_pid, fileset)

        # Object key is bin name with .zip extension
        key = f"{bin_pid}.zip"

        # Upload to object store (async)
        async def upload():
            async with StoreFactory(storage_yaml).build() as store:
                await store.put(key, buffer)

        asyncio.run(upload())

        return (bin_pid, True, None)

    except Exception as e:
        return (bin_pid, False, str(e))


def process_ifcb_directory(data_dir: str, storage_yaml: str, num_workers: int):
    """
    Process IFCB data directory and upload ZIPs to object store with multiprocessing.

    Args:
        data_dir: Path to IFCB data directory
        storage_yaml: Path to YAML file defining storage configuration
        num_workers: Number of parallel workers
    """
    # Scan the data directory once; workers get each bin's file paths
    logger.info(f"Scanning IFCB data from: {data_dir}")
    filesets = list_filesets(data_dir)
    total_bins = len(filesets)

    if total_bins == 0:
        logger.warning("No bins found to process")
        return

    logger.info(f"Found {total_bins} bins to process")
    logger.info(f"Using {num_workers} parallel workers")

    # Track progress
    total_uploaded = 0
    total_failed = 0
    start_time = time.time()
    last_log_count = 0

    try:
        # Create process pool and submit work
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all bins to the executor
            future_to_bin = {
                executor.submit(process_single_bin, fileset, storage_yaml): fileset['pid']
                for fileset in filesets
            }

            # Process completed futures as they finish
            for future in as_completed(future_to_bin):
                bin_pid, success, error = future.result()

                if success:
                    total_uploaded += 1
                else:
                    total_failed += 1
                    logger.error(f"Failed to process {bin_pid}: {error}")

                # Log progress every 10 bins
                total_processed = total_uploaded + total_failed
                if total_processed - last_log_count >= 10:
                    elapsed = time.time() - start_time
                    rate = total_processed / elapsed if elapsed > 0 else 0
                    remaining = total_bins - total_processed
                    eta_seconds = remaining / rate if rate > 0 else 0

                    # Format ETA
                    eta_mins = int(eta_seconds // 60)
                    eta_secs = int(eta_seconds % 60)

                    logger.info(
                        f"Processed: {total_processed}/{total_bins} "
                        f"({total_processed/total_bins*100:.1f}%) | "
                        f"Uploaded: {total_uploaded} | Failed: {total_failed} | "
                        f"Rate: {rate:.2f} bins/sec | "
                        f"ETA: {eta_mins}m {eta_secs}s"
                    )
                    last_log_count = total_processed

    except KeyboardInterrupt:
        logger.warning("Process interrupted by user (Ctrl+C)")
        sys.exit(1)

    # Final summary
    elapsed = time.time() - start_time
    logger.info(
        f"\nProcessing complete. Total: {total_bins}, "
        f"Uploaded: {total_uploaded}, Failed: {total_failed}, "
        f"Time: {elapsed:.1f}s"
    )

    # Only fail if ALL bins failed
    if total_failed > 0 and total_uploaded == 0:
        logger.error("All bins failed to process!")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Process IFCB data and store ZIPs in object storage with multiprocessing'
    )
    parser.add_argument(
        '--data-dir',
        required=True,
        help='Path to IFCB data directory'
    )
    parser.add_argument(
        '--storage-config',
        required=True,
        help='Path to storage YAML configuration file'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=16,
        help='Number of parallel workers (default: 16)'
    )

    args = parser.parse_args()

    process_ifcb_directory(
        args.data_dir,
        args.storage_config,
        args.num_workers
    )


if __name__ == "__main__":
    main()
