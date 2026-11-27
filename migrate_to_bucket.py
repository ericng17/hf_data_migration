import os
import re
import time
import logging
import argparse

from huggingface_hub import HfApi, hf_hub_download

# ================= Configuration =================

USERNAME = "ericng17"
BUCKET_NAME = "trafic_bucket"
DATASET_PREFIX = "hcm_traffic_snapshots"

BUCKET_ID = f"{USERNAME}/{BUCKET_NAME}"

BATCH_SIZE = 100
MAX_RETRIES = 5
RETRY_DELAY = 5
MAX_RETRY_DELAY = 120

# Stop accepting new batches before GitHub timeout.
MAX_RUNTIME = int(os.getenv("MAX_RUNTIME", "19800"))

# ================= Initialization =================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)
api = HfApi()


# ================= Retry =================

def retry(func, *args, **kwargs):
    """Execute an operation with exponential backoff."""

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return func(*args, **kwargs)

        except Exception as error:
            response = getattr(error, "response", None)
            status = getattr(response, "status_code", None)

            # Permanent errors should not be retried.
            if status in (400, 401, 403, 404, 413, 422):
                raise

            if attempt == MAX_RETRIES:
                raise

            delay = min(
                RETRY_DELAY * 2 ** (attempt - 1),
                MAX_RETRY_DELAY,
            )

            logger.warning(
                "Attempt %d/%d failed: %s. Retry in %ds.",
                attempt,
                MAX_RETRIES,
                error,
                delay,
            )

            time.sleep(delay)


# ================= Discovery =================

def get_datasets():
    pattern = re.compile(
        rf"^{re.escape(USERNAME)}/"
        rf"{re.escape(DATASET_PREFIX)}(\d*)$"
    )

    datasets = retry(
        lambda: list(api.list_datasets(author=USERNAME))
    )

    matches = [
        dataset.id
        for dataset in datasets
        if pattern.fullmatch(dataset.id)
    ]

    return sorted(
        matches,
        key=lambda name: int(
            pattern.fullmatch(name).group(1) or 0
        ),
    )


def get_source_files(repo_id, revision):
    """Get all files from a pinned repository revision."""

    entries = retry(
        lambda: list(
            api.list_repo_tree(
                repo_id=repo_id,
                repo_type="dataset",
                revision=revision,
                recursive=True,
            )
        )
    )

    return [
        entry
        for entry in entries
        if hasattr(entry, "size")
    ]


def get_destination_files(paths):
    """Read existing bucket metadata for requested paths."""

    if not paths:
        return {}

    entries = retry(
        lambda: list(
            api.get_bucket_paths_info(
                bucket_id=BUCKET_ID,
                paths=paths,
            )
        )
    )

    return {
        entry.path: entry
        for entry in entries
    }


# ================= Resume Logic =================

def is_copied(source, destination):
    """Check whether the destination matches the source."""

    if destination is None:
        return False

    if source.size != destination.size:
        return False

    source_hash = getattr(source, "xet_hash", None)
    destination_hash = getattr(
        destination, "xet_hash", None
    )

    # Compare content hashes when available.
    if source_hash:
        return source_hash == destination_hash

    # Non-Xet files cannot be reliably compared using
    # the same hash algorithm. Re-copy them safely.
    return False


def chunks(items, size):
    for index in range(0, len(items), size):
        yield items[index:index + size]


# ================= Migration =================

def migrate_batch(repo_id, revision, prefix, files):
    destinations = get_destination_files([
        f"{prefix}/{file.path}"
        for file in files
    ])

    copy_operations = []
    upload_operations = []
    skipped = 0

    for file in files:
        target = f"{prefix}/{file.path}"

        existing = destinations.get(target)

        if is_copied(file, existing):
            skipped += 1
            continue

        xet_hash = getattr(file, "xet_hash", None)

        if xet_hash:
            # Server-side copy without transferring
            # the file through the GitHub runner.
            copy_operations.append((
                "dataset",
                repo_id,
                xet_hash,
                target,
            ))

        else:
            # Fallback for files not stored using Xet.
            local_path = retry(
                hf_hub_download,
                repo_id=repo_id,
                repo_type="dataset",
                filename=file.path,
                revision=revision,
            )

            upload_operations.append((
                local_path,
                target,
            ))

    if copy_operations or upload_operations:
        retry(
            api.batch_bucket_files,
            bucket_id=BUCKET_ID,
            copy=copy_operations or None,
            add=upload_operations or None,
        )

        # Verify objects after the write succeeds.
        pending = [
            file
            for file in files
            if not is_copied(
                file,
                destinations.get(
                    f"{prefix}/{file.path}"
                ),
            )
        ]

        verified = get_destination_files([
            f"{prefix}/{file.path}"
            for file in pending
        ])

        for file in pending:
            target = f"{prefix}/{file.path}"
            remote = verified.get(target)

            if remote is None or remote.size != file.size:
                raise RuntimeError(
                    f"Verification failed: {target}"
                )

            if getattr(file, "xet_hash", None):
                if not is_copied(file, remote):
                    raise RuntimeError(
                        f"Hash mismatch: {target}"
                    )

    return {
        "copied": len(copy_operations),
        "uploaded": len(upload_operations),
        "skipped": skipped,
    }


def migrate_dataset(repo_id, deadline):
    prefix = repo_id.split("/")[-1]

    # Pin source revision to prevent inconsistencies
    # if the dataset changes while migrating.
    info = retry(api.dataset_info, repo_id)
    revision = info.sha

    if not revision:
        raise RuntimeError(
            f"Cannot resolve revision: {repo_id}"
        )

    files = get_source_files(repo_id, revision)

    total = len(files)

    stats = {
        "copied": 0,
        "uploaded": 0,
        "skipped": 0,
    }

    logger.info(
        "Dataset: %s | Revision: %s | Files: %d",
        repo_id,
        revision,
        total,
    )

    processed = 0

    for batch in chunks(files, BATCH_SIZE):

        # Stop before GitHub Actions forces cancellation.
        if time.monotonic() >= deadline:
            logger.warning(
                "Runtime limit reached. "
                "Re-run the workflow to resume."
            )

            return False, stats

        result = retry(
            migrate_batch,
            repo_id,
            revision,
            prefix,
            batch,
        )

        for key in stats:
            stats[key] += result[key]

        processed += len(batch)

        logger.info(
            "[%s] %d/%d (%.2f%%) | "
            "Copied: %d | Uploaded: %d | Skipped: %d",
            prefix,
            processed,
            total,
            processed / total * 100,
            stats["copied"],
            stats["uploaded"],
            stats["skipped"],
        )

    logger.info("COMPLETED: %s", repo_id)

    return True, stats


# ================= Main =================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        help="Migrate a specific dataset only",
    )

    args = parser.parse_args()

    started_at = time.monotonic()
    deadline = started_at + MAX_RUNTIME

    if args.dataset:
        repo_id = args.dataset

        if "/" not in repo_id:
            repo_id = f"{USERNAME}/{repo_id}"

        datasets = [repo_id]

    else:
        datasets = get_datasets()

    if not datasets:
        raise SystemExit("No matching datasets found.")

    retry(
        api.create_bucket,
        bucket_id=BUCKET_ID,
        private=True,
        exist_ok=True,
    )

    logger.info("Destination: %s", BUCKET_ID)
    logger.info("Found %d datasets", len(datasets))

    totals = {
        "copied": 0,
        "uploaded": 0,
        "skipped": 0,
    }

    completed = []
    failed = []
    interrupted = False

    for index, repo_id in enumerate(datasets, start=1):

        if time.monotonic() >= deadline:
            interrupted = True
            break

        logger.info(
            "[%d/%d] Processing %s",
            index,
            len(datasets),
            repo_id,
        )

        try:
            success, stats = migrate_dataset(
                repo_id,
                deadline,
            )

            for key in totals:
                totals[key] += stats[key]

            if success:
                completed.append(repo_id)
            else:
                interrupted = True
                break

        except Exception:
            logger.exception("FAILED: %s", repo_id)
            failed.append(repo_id)

    elapsed = time.monotonic() - started_at

    print("\n" + "=" * 60)
    print("MIGRATION SUMMARY")
    print("=" * 60)

    print(f"Datasets       : {len(datasets)}")
    print(f"Completed      : {len(completed)}")
    print(f"Failed         : {len(failed)}")
    print(f"Copied files   : {totals['copied']}")
    print(f"Uploaded files : {totals['uploaded']}")
    print(f"Skipped files  : {totals['skipped']}")
    print(f"Elapsed        : {elapsed / 60:.2f} minutes")
    print(f"Interrupted    : {interrupted}")
    print(f"Bucket         : {BUCKET_ID}")

    if failed:
        print("\nFailed datasets:")

        for repo_id in failed:
            print(f"  - {repo_id}")

    if interrupted or failed:
        print("\nRe-run the workflow to resume.")
        raise SystemExit(1)

    print("\nMigration completed successfully.")


if __name__ == "__main__":
    main()
