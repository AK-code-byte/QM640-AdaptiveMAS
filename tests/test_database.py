import tempfile, os
import pandas as pd
from src.database import ResultsDB

def test_write_and_read_row():
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    try:
        db = ResultsDB(db_path)
        row = {
            'question_id': 'MEDQA_1', 'dataset_source': 'MedQA',
            'agent_config': 'single-agent', 'llm_backbone': 'GPT-4o-mini',
            'is_correct': 1, 'token_cost': 500, 'routing_overhead_tokens': 0,
            'inference_tokens': 500, 'routing_decision': 'solo',
            'task_complexity': 'Simple', 'complexity_score': 0.2,
            'num_symptoms': 3, 'num_conditions': 1,
            'clinical_vignette_length': 120, 'predicted_answer': 'flu',
            'reasoning_steps': 1, 'latency_ms': 1200.0,
        }
        db.insert(row)
        df = db.to_dataframe()
        assert len(df) == 1
        assert df.iloc[0]['question_id'] == 'MEDQA_1'
    finally:
        os.unlink(db_path)

def test_idempotent_schema_creation():
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    try:
        ResultsDB(db_path)
        ResultsDB(db_path)  # second open should not raise
    finally:
        os.unlink(db_path)
