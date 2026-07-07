"""
scripts/clear_training_labels.py

Wipes the training_labels table so you can start collecting fresh labels
(e.g. after changing signal_logic thresholds like TP_RISK_REWARD — old
labels were generated under the OLD rules and mixing them with new-rule
labels would train the AI confidence model on inconsistent data).

Usage:
    python scripts/clear_training_labels.py            # asks for confirmation
    python scripts/clear_training_labels.py --yes      # skips confirmation
    python scripts/clear_training_labels.py --count    # just show the count, don't delete
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger
from dotenv import load_dotenv
from app.database.client import get_client

load_dotenv()

# training_labels.id is `uuid primary key default gen_random_uuid()` per
# app/database/schema.sql — PostgREST requires a filter on delete, so we
# match every row with "id != an id that can never exist" rather than a
# numeric comparison (which would fail against a uuid column).
_NEVER_MATCHES_UUID = "00000000-0000-0000-0000-000000000000"


def get_label_count() -> int:
    client = get_client()
    result = client.table("training_labels").select("id", count="exact").execute()
    return result.count or 0


def clear_training_labels() -> int:
    client = get_client()
    result = client.table("training_labels").delete().neq("id", _NEVER_MATCHES_UUID).execute()
    return len(result.data or [])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yes", action="store_true", help="skip confirmation prompt")
    parser.add_argument("--count", action="store_true", help="only show the row count, don't delete")
    args = parser.parse_args()

    count = get_label_count()
    logger.info(f"training_labels currently has {count} rows")

    if args.count:
        return

    if count == 0:
        logger.info("Nothing to delete.")
        return

    if not args.yes:
        confirm = input(f"Delete all {count} rows from training_labels? Type 'yes' to confirm: ")
        if confirm.strip().lower() != "yes":
            logger.info("Aborted — no rows deleted.")
            return

    deleted = clear_training_labels()
    logger.info(f"Deleted {deleted} rows from training_labels. Table is now empty.")
    logger.info("Next: re-run scripts/backtest.py to regenerate labels under the current signal_logic rules.")


if __name__ == "__main__":
    main()