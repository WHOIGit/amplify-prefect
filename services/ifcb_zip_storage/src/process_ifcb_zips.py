#!/usr/bin/env python3
"""
Process IFCB data directory and upload ZIPs to object storage.

This script:
1. Reads IFCB data using pyifcb DataDirectory
2. Converts each fileset to ZIP format using bin2zip_stream
3. Uploads ZIP buffers to object store defined by YAML configuration
"""
import argparse
import asyncio
import logging
import sys
from ifcb.data.files import DataDirectory
from ifcb.data.zip import bin2zip_stream
from storage.config_builder import StoreFactory


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def process_ifcb_directory(data_dir: str, storage_yaml: str):
    """
    Process IFCB data directory and upload ZIPs to object store.

    Args:
        data_dir: Path to IFCB data directory
        storage_yaml: Path to YAML file defining storage configuration
    """
    # Initialize storage from YAML config
    logger.info(f"Initializing storage from: {storage_yaml}")

    # Build store from YAML and enter async context manager
    async with StoreFactory(storage_yaml).build() as store:
        # Initialize IFCB data directory
        logger.info(f"Processing IFCB data from: {data_dir}")
        dd = DataDirectory(data_dir)

        total_processed = 0
        total_uploaded = 0
        total_failed = 0

        # Iterate through filesets
        for fileset_bin in dd:
            bin_name = fileset_bin.pid
            total_processed += 1

            try:
                logger.info(f"Processing bin: {bin_name}")

                # Generate ZIP stream
                buffer = bin2zip_stream(fileset_bin)

                # Object key is bin name with .zip extension
                key = f"{bin_name}.zip"

                # Upload to object store
                await store.put(key, buffer)
                logger.info(f"Uploaded: {key}")
                total_uploaded += 1

            except Exception as e:
                logger.error(f"Failed to process {bin_name}: {str(e)}")
                total_failed += 1
                continue

        logger.info(
            f"Processing complete. Total: {total_processed}, "
            f"Uploaded: {total_uploaded}, Failed: {total_failed}"
        )

        # Only fail if ALL bins failed
        if total_failed > 0 and total_uploaded == 0:
            logger.error("All bins failed to process!")
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Process IFCB data and store ZIPs in object storage'
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

    args = parser.parse_args()

    asyncio.run(process_ifcb_directory(args.data_dir, args.storage_config))


if __name__ == "__main__":
    main()
