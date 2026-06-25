import time
from supabase import create_client, Client
from loguru import logger
from app.config import SUPABASE

_client: Client | None = None


def get_client() -> Client:
    global _client
    if _client is None:
        _client = _connect()
    return _client


def _connect(retries: int = 3, delay: float = 2.0) -> Client:
    for attempt in range(1, retries + 1):
        try:
            client = create_client(SUPABASE.url, SUPABASE.key)
            logger.info("Supabase connected")
            return client
        except Exception as e:
            logger.warning(f"Supabase connection attempt {attempt}/{retries} failed: {e}")
            if attempt < retries:
                time.sleep(delay * attempt)
    raise ConnectionError("Could not connect to Supabase after multiple attempts")


def check_connection() -> bool:
    try:
        client = get_client()
        client.table("bot_logs").select("id").limit(1).execute()
        return True
    except Exception as e:
        logger.error(f"Supabase health check failed: {e}")
        return False