#!/usr/bin/env python3
"""
Process IFCB data directory and upload ZIPs to object storage with multiprocessing.

This script:
1. Reads IFCB data using pyifcb DataDirectory
2. Converts each fileset to ZIP format using bin2zip_stream (in parallel)
3. Uploads ZIP buffers to object store defined by YAML configuration (in parallel)
"""
import argparse
import asyncio
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from ifcb.data.files import DataDirectory
from ifcb.data.zip import bin2zip_stream
from storage.config_builder import StoreFactory


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_single_bin(data_dir: str, bin_pid: str, storage_yaml: str) -> tuple:
    """
    Worker function to process a single bin: zip and upload.

    Args:
        data_dir: Path to IFCB data directory
        bin_pid: Bin PID to process
        storage_yaml: Path to storage YAML config

    Returns:
        tuple: (bin_pid, success: bool, error_message: str or None)
    """
    try:
        # Recreate DataDirectory and get the fileset by PID
        dd = DataDirectory(data_dir)
        fileset_bin = dd[bin_pid]

        # Generate ZIP stream
        buffer = bin2zip_stream(fileset_bin)

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
    # Initialize IFCB data directory and collect all bin PIDs
    logger.info(f"Scanning IFCB data from: {data_dir}")
    dd = DataDirectory(data_dir)

    bin_pids = [str(fileset_bin.pid) for fileset_bin in dd]
    total_bins = len(bin_pids)

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
                executor.submit(process_single_bin, data_dir, bin_pid, storage_yaml): bin_pid
                for bin_pid in bin_pids
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
