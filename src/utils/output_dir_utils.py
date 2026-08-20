import os

# Group-writable, world-readable, setgid. The setgid bit makes everything created
# underneath inherit the directory's group rather than the creating process's
# primary group, and nested directories inherit the bit itself, so it cascades.
OUTPUT_DIR_MODE = 0o2775


def create_output_dir(output_dir: str, logger=None) -> str:
    """
    Create an output directory that the owning group can write to.

    Two problems this solves:

    1. ``os.makedirs``' own ``mode`` argument is masked by the umask, so under the
       Prefect process umask (0022) an output root lands 0755. The group gets the
       right group via an inherited setgid bit but cannot write to the directory.
    2. When a flow does not create the directory at all, Docker creates any
       missing bind-mount source itself, as ``root:root`` 0755 with no setgid. A
       ``umask`` inside the image cannot recover from that, because the directory
       is already wrong before the container starts.

    Group write is not inherited from the parent -- it comes from the creating
    process's umask -- so it has to be set explicitly here. Containers writing
    underneath still need their own ``umask 0002`` for the files they create.

    Args:
        output_dir: Directory to create.
        logger: Optional logger for reporting a chmod that could not be applied.

    Returns:
        The directory path, for convenience.
    """
    os.makedirs(output_dir, exist_ok=True)

    try:
        os.chmod(output_dir, OUTPUT_DIR_MODE)
    except OSError as e:
        # A pre-existing directory owned by another user cannot be chmod'ed. That
        # is not worth failing a run over -- the directory exists either way.
        message = f"Could not set {OUTPUT_DIR_MODE:o} on {output_dir}: {e}"
        if logger is not None:
            logger.warning(message)
        else:
            print(message)

    return output_dir
