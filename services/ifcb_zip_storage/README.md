# IFCB ZIP Storage Service

This service processes IFCB data directories, generates ZIP files for each fileset using `pyifcb`, and uploads them to object storage using `amplify-storage-utils`.

## Overview

- **Input**: IFCB data directory path
- **Process**:
  1. Iterates through IFCB filesets using `pyifcb.DataDirectory`
  2. Converts each fileset to ZIP format using `bin2zip_stream()`
  3. Uploads ZIP buffers to object storage
- **Output**: ZIP files stored in configured object store with keys: `{bin_name}.zip`

## Usage

This service is called by the Prefect flow `ifcb_zip_storage` which requires:
- `data_dir`: Path to IFCB data directory
- `storage_yaml`: Path to YAML file defining the object store configuration
- `env_file` (optional): Path to .env file containing environment variables for the storage YAML

## Storage Configuration

The service uses `amplify-storage-utils` for flexible object storage backends. Storage is configured via a YAML file with environment variable substitution.

### Step 1: Create Environment File

Create a `.env` file with your credentials:

```bash
# storage.env
S3_ENDPOINT_URL=https://s3.example.com
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
```

### Step 2: Create Storage YAML

Create a YAML file that references the environment variables:

```yaml
# storage.yaml
stores:
  # Base S3/MinIO store
  s3_base:
    type: AsyncBucketStore
    config:
      endpoint_url: ${S3_ENDPOINT_URL}
      s3_access_key: ${AWS_ACCESS_KEY_ID}
      s3_secret_key: ${AWS_SECRET_ACCESS_KEY}
      bucket_name: my-ifcb-data

  # Wrap with prefix store to add a prefix to all keys
  prefixed_store:
    type: PrefixStore
    config:
      prefix: ifcb/zips/
    base: s3_base

main: prefixed_store
```

### Step 3: Run the Flow

Pass all three parameters to the flow:
```python
{
    "data_dir": "/path/to/ifcb/data",
    "storage_yaml": "/path/to/storage.yaml",
    "env_file": "/path/to/storage.env"
}
```

### How It Works

1. The `.env` file is parsed and loaded into the container's environment
2. `amplify-storage-utils` substitutes `${VARIABLE_NAME}` placeholders in the YAML with the actual values
3. All ZIP files are stored with the configured prefix: `ifcb/zips/D20241217T120000_IFCB001.zip`

## Dependencies

- `pyifcb` v1.2.1 - IFCB data processing
- `amplify-storage-utils` v1.4.2 - Object storage abstraction
- `PyYAML` - YAML configuration parsing
