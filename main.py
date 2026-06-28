import asyncio
import sys
import os
from loguru import logger


def setup_logging() -> None:
    os.makedirs("logs", exist_ok=True)
    logger.remove()
    logger.add(
        sys.stdout,
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{line}</cyan> — {message}",
    )
    logger.add(
        "logs/bot_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        rotation="00:00",
        retention="30 days",
        compression="zip",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} — {message}",
    )


async def main() -> None:
    setup_logging()
    logger.info("=" * 50)
    logger.info("  crypto-bot starting up")
    logger.info("=" * 50)

    try:
        from app.engine import TradingEngine
        engine = TradingEngine()
        await engine.start()
    except ValueError as e:
        logger.critical(f"Configuration error: {e}")
        logger.critical("Check your .env file and try again")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.critical(f"Unexpected error: {e}")
        raise


if __name__ == "__main__":
    # Start health server to keep Render free tier awake
    from keep_alive import start_health_server
    start_health_server()

    asyncio.run(main())