import os
import re
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import sys
sys.path.append('/Users/ccy/Documents/CMU/Fall2025/17645-ML-in-production/group-project-f25-the-real-reel-deal')

from pathlib import Path
import pandas as pd

from ..config import BRONZE_DIR  
from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------
# Built-in parser for Kafka log lines
# ---------------------------------------------------------------------
RE_WATCH = re.compile(r"GET /data/m/(?P<movieid>[^/]+)/(?P<minute>\d+)\.mpg")
RE_RATE  = re.compile(r"GET /rate/(?P<movieid>[^=]+)=(?P<rating>[1-5])")


def _split_head(line: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    parts = line.strip().split(",", 2)
    if len(parts) < 3:
        return None, None, None
    ts_raw, user_id, rest = parts
    return ts_raw.strip(), user_id.strip(), rest.strip()


def _safe_ts_parse(ts_raw: str) -> Optional[pd.Timestamp]:
    ts = pd.to_datetime(ts_raw, errors="coerce", utc=True)
    if pd.isna(ts):
        return None
    return ts


def parse_line(line: str) -> Tuple[Optional[str], Optional[Dict]]:
    ts_raw, user_id, rest = _split_head(line)
    if not ts_raw or not user_id or rest is None:
        return None, None

    ts = _safe_ts_parse(ts_raw)
    if ts is None:
        return None, None

    m = RE_WATCH.search(rest)
    if m:
        movie_id = m.group("movieid").strip()
        try:
            minute = int(m.group("minute"))
        except Exception:
            minute = None
        return "watch", {"ts": ts, "user_id": user_id, "movie_id": movie_id, "minute": minute}

    m = RE_RATE.search(rest)
    if m:
        movie_id = m.group("movieid").strip()
        try:
            rating = int(m.group("rating"))
        except Exception:
            rating = None
        return "rating", {"ts": ts, "user_id": user_id, "movie_id": movie_id, "rating": rating}

    # Request parsing - Fixed regex patterns
    if ("status" in rest) or ("result:" in rest) or ("recommendation request" in rest):
        status = None
        ms = None
        result_raw = ""
        
        # Fixed: Match "status 200" format
        try:
            m_status = re.search(r"status\s+(\d+)", rest)
            if m_status:
                status = m_status.group(1)  # Keep as string initially
        except Exception:
            status = None
        
        # Fixed: Match "137 ms" format
        try:
            m_ms = re.search(r"(\d+)\s+ms", rest)
            if m_ms:
                ms = m_ms.group(1)  # Keep as string initially
        except Exception:
            ms = None
        
        # Extract result (movies list)
        try:
            idx = rest.find("result:")
            if idx != -1:
                # Get everything after "result:"
                result_part = rest[idx + len("result:"):].strip()
                
                # Find where the result ends (before ", <number> ms")
                # Look for the pattern: ", <digits> ms" at the end
                end_match = re.search(r",\s+\d+\s+ms\s*$", result_part)
                if end_match:
                    result_raw = result_part[:end_match.start()].strip()
                else:
                    # Fallback: take everything
                    result_raw = result_part.strip()
        except Exception:
            result_raw = ""
        
        return "request", {
            "ts": ts, 
            "user_id": user_id, 
            "status": status,  # String or None
            "response_ms": ms,  # String or None
            "result_raw": result_raw
        }

    return None, None


# ---------------------------------------------------------------------
# Bronze Layer: Kafka ingestion (uses built-in parser)
# ---------------------------------------------------------------------
def load_kafka_events(
    num_messages: Optional[int] = 10000,
    event_types: Optional[List[str]] = None,
) -> pd.DataFrame:
    print("🚀 Starting Kafka event ingestion...")

    try:
        from confluent_kafka import Consumer
    except Exception as e:
        raise ImportError(
            "confluent_kafka is required for Kafka ingestion. Install it or skip calling load_kafka_events()."
        ) from e

    bootstrap = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
    topic     = os.getenv("KAFKA_TOPIC", "movielogN")
    group     = os.getenv("KAFKA_GROUP_ID", "mlip_ingestor")
    reset     = os.getenv("AUTO_OFFSET_RESET", "earliest")
    flush_n   = int(os.getenv("FLUSH_EVERY", "5000"))  # kept for future batching

    conf = {
        "bootstrap.servers": bootstrap,
        "group.id": group,
        "auto.offset.reset": reset,
        "enable.auto.commit": False,
    }

    consumer = Consumer(conf)
    consumer.subscribe([topic])
    print(f"Subscribed to topic={topic} bootstrap={bootstrap} group={group} ")

    records: List[Dict] = []
    count = 0

    try:
        while num_messages is None or count < num_messages:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                print(f"Kafka error: {msg.error()}")
                continue
            line = msg.value().decode("utf-8", errors="ignore")
            etype, payload = parse_line(line)
            if payload is None:
                continue
            if event_types and etype not in event_types:
                continue
            payload["event_type"] = etype
            records.append(payload)
            count += 1
            if count % 1000 == 0:
                print(f"Processed {count} messages...")

    except KeyboardInterrupt:
        print("Kafka ingestion interrupted.")
    finally:
        consumer.close()

    if not records:
        raise RuntimeError("No valid Kafka messages parsed.")

    df = pd.DataFrame(records)
    print(f"Parsed {len(df)} Kafka events.")

    # Save by event type (partition buckets) -> use BRONZE_DIR
    for etype, group in df.groupby("event_type"):
        out_dir = Path(BRONZE_DIR) / f"{etype}_events"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"{etype}_{ts}.parquet"
        group.to_parquet(out_path, index=False)
        print(f"💾 Saved {len(group)} {etype} events to {out_path}")

    return df


def run_kafka_ingest_continuous(
    event_types: Optional[List[str]] = None,
    flush_every: int = 5000,       # Flush when record count reaches this threshold
    flush_secs: int = 30,          # Or flush after this many seconds
    partition_by_hour: bool = True # Partition directories by day/hour
):
    """
    Continuously consumes data from Kafka and writes rolling Parquet files (Bronze layer).
    Designed to run continuously on a VM.

    - Offsets are manually committed only after successful file writes (at-least-once guarantee)
    - Buffers and outputs are separated by event_type
    """
    try:
        from confluent_kafka import Consumer, KafkaException
    except Exception as e:
        raise ImportError("confluent_kafka is required. Please install with `pip install confluent-kafka`.") from e

    import os
    import time
    import signal
    from collections import defaultdict
    from datetime import datetime, timezone
    from pathlib import Path
    import pandas as pd

    bootstrap = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
    topic     = os.getenv("KAFKA_TOPIC", "movielog2")
    group     = os.getenv("KAFKA_GROUP_ID", "mlip_ingestor")
    reset     = os.getenv("AUTO_OFFSET_RESET", "earliest")

    conf = {
        "bootstrap.servers": bootstrap,
        "group.id": group,
        "auto.offset.reset": reset,
        "enable.auto.commit": False,   # Manual commit
        "max.poll.interval.ms": 300000, # Prevent rebalance due to long processing time
        # Memory limits to prevent VM RAM exhaustion and crashes
        "fetch.message.max.bytes": 1048576,      # 1MB max per message
        "queued.max.messages.kbytes": 65536,     # 64MB max queue size
    }

    consumer = Consumer(conf)
    consumer.subscribe([topic])
    print(f"Ingesting from topic={topic} on {bootstrap} as group={group}")

    # Buffers grouped by event_type
    buffers: dict[str, List[dict]] = defaultdict(list)
    last_flush_ts = time.time()
    running = True
    pending_commits = []  # Messages pending commit (committed only after successful write)

    def _flush_all(reason: str):
        nonlocal last_flush_ts, pending_commits
        total_rows = 0

        for etype, rows in buffers.items():
            if not rows:
                continue

            df = pd.DataFrame(rows)

            # Directory: bronze/<etype>_events/dt=YYYY-MM-DD[/hour=HH]
            now = datetime.now(timezone.utc)
            base = Path(BRONZE_DIR) / f"{etype}_events" / f"dt={now.strftime('%Y-%m-%d')}"
            if partition_by_hour:
                base = base / f"hour={now.strftime('%H')}"
            base.mkdir(parents=True, exist_ok=True)

            # File name: part-<unix>-<etype>-<rows>.parquet (simple way to avoid collisions)
            out_path = base / f"part-{int(now.timestamp())}-{etype}-{len(rows)}.parquet"
            df.to_parquet(out_path, index=False)
            print(f"[{reason}] Flushed {len(rows)} rows → {out_path}")

            total_rows += len(rows)
            buffers[etype].clear()

        if total_rows > 0 and pending_commits:
            try:
                consumer.commit(asynchronous=False)
                print(f"Committed offsets for {len(pending_commits)} messages.")
            except KafkaException as ke:
                print(f"Commit failed: {ke}")
            finally:
                pending_commits.clear()

        last_flush_ts = time.time()

    def _graceful_stop(signum, frame):
        nonlocal running
        print(f"Signal {signum} received, flushing and stopping...")
        running = False

    signal.signal(signal.SIGINT, _graceful_stop)
    signal.signal(signal.SIGTERM, _graceful_stop)

    try:
        while running:
            msg = consumer.poll(1.0)
            now = time.time()

            if msg is None:
                # Time-based flush
                if now - last_flush_ts >= flush_secs:
                    _flush_all("time")
                continue

            if msg.error():
                print(f"Kafka error: {msg.error()}")
                continue

            line = msg.value().decode("utf-8", errors="ignore")
            etype, payload = parse_line(line)
            if payload is None:
                continue
            if event_types and etype not in event_types:
                continue

            payload["event_type"] = etype
            buffers[etype].append(payload)
            pending_commits.append(msg)

            # Record-based flush
            total_buffered = sum(len(v) for v in buffers.values())
            if total_buffered >= flush_every:
                _flush_all("count")
                time.sleep(5)  # Throttle processing to prevent memory spikes

            # Time-based flush
            if now - last_flush_ts >= flush_secs:
                _flush_all("time")
                time.sleep(5)  # Throttle processing to prevent memory spikes

        # Final flush before shutdown
        _flush_all("shutdown")

    finally:
        consumer.close()
        print("Consumer closed.")
