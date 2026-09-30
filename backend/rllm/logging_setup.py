import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    # Windows consoles default to a legacy code page; don't crash on non-ASCII text.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "httpcore", "urllib3", "sentence_transformers", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
