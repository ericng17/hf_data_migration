import re
import time
import logging

from huggingface_hub import HfApi

# ================= Configuration =================

USERNAME = "ericng17"
BUCKET_NAME = "trafic_bucket"
DATASET_PREFIX = "hcm_traffic_snapshots"
MAX_RETRIES = 3
RETRY_DELAY = 5

BUCKET_ID = f"{USERNAME}/{BUCKET_NAME}"

# ================= Initialization =================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)
api = HfApi()


def get_datasets():
    """Find all matching traffic snapshot repositories."""

    pattern = re.compile(
        rf"^{re.escape(USERNAME)}/"
        rf"{re.escape(DATASET_PREFIX)}(\d*)$"
    )

    datasets = [
        dataset.id
        for dataset in api.list_datasets(author=USERNAME)
        if pattern.fullmatch(dataset.id)
    ]

    def sort_key(repo_id):
        match = pattern.fullmatch(repo_id)
        return int(match.group(1) or 1)

    return sorted(datasets, key=sort_key)


def copy_dataset(repo_id):
    """Copy a dataset repository to the destination bucket."""

    repo_name = repo_id.split("/")[-1]

    source = f"hf://datasets/{repo_id}/"
    destination = f"hf://buckets/{BUCKET_ID}/{repo_name}/"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info("Source: %s", source)
            logger.info("Destination: %s", destination)

            api.copy_files(source, destination)

            logger.info("SUCCESS: %s", repo_id)
            return True

        except Exception as error:
            logger.error(
                "Attempt %d/%d failed: %s",
                attempt,
                MAX_RETRIES,
                error,
            )

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)

    return False


def main():
    started_at = time.monotonic()

    datasets = get_datasets()

    if not datasets:
        logger.warning("No matching datasets found.")
        return

    logger.info("Found %d datasets:", len(datasets))

    for dataset in datasets:
        print(f"  - {dataset}")

    # Create the bucket if it does not exist.
    api.create_bucket(
        BUCKET_ID,
        private=True,
        exist_ok=True,
    )

    logger.info("Destination bucket: %s", BUCKET_ID)

    successful = []
    failed = []

    for index, repo_id in enumerate(datasets, start=1):
        logger.info(
            "[%d/%d] Copying %s",
            index,
            len(datasets),
            repo_id,
        )

        if copy_dataset(repo_id):
            successful.append(repo_id)
        else:
            failed.append(repo_id)

    elapsed = time.monotonic() - started_at

    print("\n" + "=" * 60)
    print("MIGRATION SUMMARY")
    print("=" * 60)
    print(f"Total datasets : {len(datasets)}")
    print(f"Successful     : {len(successful)}")
    print(f"Failed         : {len(failed)}")
    print(f"Elapsed        : {elapsed:.2f} seconds")
    print(f"Bucket         : {BUCKET_ID}")

    if failed:
        print("\nFailed repositories:")

        for repo_id in failed:
            print(f"  - {repo_id}")

    print(f"\nhttps://huggingface.co/buckets/{BUCKET_ID}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
