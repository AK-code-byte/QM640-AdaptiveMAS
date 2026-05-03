import sqlite3
import pandas as pd
from typing import Dict

SCHEMA = """
CREATE TABLE IF NOT EXISTS results (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    question_id              TEXT NOT NULL,
    dataset_source           TEXT,
    agent_config             TEXT,
    llm_backbone             TEXT,
    is_correct               INTEGER,
    token_cost               INTEGER,
    routing_overhead_tokens  INTEGER DEFAULT 0,
    inference_tokens         INTEGER DEFAULT 0,
    routing_decision         TEXT,
    task_complexity          TEXT,
    complexity_score         REAL,
    num_symptoms             INTEGER,
    num_conditions           INTEGER,
    clinical_vignette_length INTEGER,
    predicted_answer         TEXT,
    reasoning_steps          INTEGER,
    latency_ms               REAL,
    created_at               TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

class ResultsDB:
    def __init__(self, path: str = 'results.db'):
        self.path = path
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute(SCHEMA)
        self._conn.commit()

    def insert(self, row: Dict):
        cols = ', '.join(row.keys())
        placeholders = ', '.join('?' * len(row))
        self._conn.execute(
            f'INSERT INTO results ({cols}) VALUES ({placeholders})',
            list(row.values())
        )
        self._conn.commit()

    def insert_many(self, rows):
        for row in rows:
            self.insert(row)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.read_sql('SELECT * FROM results', self._conn)

    def close(self):
        self._conn.close()
