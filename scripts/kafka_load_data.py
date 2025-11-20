from ml_pipeline.data_io_refactored.bronze_ingest import run_kafka_ingest_continuous

run_kafka_ingest_continuous(
    event_types= None,  # None represent all
    flush_every=1000,                 # write every 1000 pieces (reduced from 2000 to prevent memory issues)
    flush_secs=30,                    # write every 30 seconds
    partition_by_hour=True            # dt=YYYY-MM-DD/hour=HH/
)