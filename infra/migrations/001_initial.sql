-- ChemAI initial schema (reference; applied via SQLAlchemy models)

CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username VARCHAR(64) UNIQUE NOT NULL,
    email VARCHAR(128) DEFAULT '',
    password_hash VARCHAR(256) DEFAULT '',
    reputation REAL DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS reactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(256) NOT NULL,
    rxn_smarts TEXT NOT NULL,
    reactants_json TEXT DEFAULT '[]',
    products_json TEXT DEFAULT '[]',
    domain VARCHAR(64) DEFAULT 'organic',
    tags_json TEXT DEFAULT '[]',
    created_by VARCHAR(64) DEFAULT 'system',
    is_public BOOLEAN DEFAULT 1,
    forked_from INTEGER REFERENCES reactions(id),
    base_yield REAL DEFAULT 0.75,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS reaction_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    reaction_id INTEGER REFERENCES reactions(id),
    conditions_json TEXT DEFAULT '{}',
    predicted_json TEXT DEFAULT '{}',
    actual_json TEXT DEFAULT '{}',
    status VARCHAR(32) DEFAULT 'predicted',
    user_id VARCHAR(64) DEFAULT 'anonymous',
    provenance VARCHAR(64) DEFAULT 'internal_experiment',
    quality_score REAL DEFAULT 0.0,
    visibility VARCHAR(16) DEFAULT 'public',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    exp_type TEXT NOT NULL,
    name TEXT NOT NULL,
    pred_value REAL,
    actual_value REAL,
    metric TEXT,
    notes TEXT,
    composition TEXT,
    user TEXT DEFAULT 'anonymous',
    version_tag TEXT DEFAULT 'v1',
    data_quality TEXT DEFAULT 'good',
    source_provenance TEXT DEFAULT 'internal_experiment',
    reaction_run_id INTEGER REFERENCES reaction_runs(id)
);

CREATE TABLE IF NOT EXISTS model_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    exp_type TEXT NOT NULL,
    mae REAL,
    rmse REAL,
    n_samples INTEGER,
    model_name TEXT DEFAULT 'default',
    task TEXT DEFAULT 'yield',
    metrics_json TEXT DEFAULT '{}',
    promoted BOOLEAN DEFAULT 0
);

CREATE TABLE IF NOT EXISTS reaction_requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_input TEXT NOT NULL,
    status VARCHAR(32) DEFAULT 'pending',
    user_id VARCHAR(64) DEFAULT 'anonymous',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS job_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_type VARCHAR(64) NOT NULL,
    payload_json TEXT DEFAULT '{}',
    status VARCHAR(32) DEFAULT 'queued',
    scheduled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    result_json TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS benchmark_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    suite_name VARCHAR(128) NOT NULL,
    metrics_json TEXT DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS flags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    target_type VARCHAR(32) NOT NULL,
    target_id INTEGER NOT NULL,
    user_id VARCHAR(64) DEFAULT 'anonymous',
    reason TEXT DEFAULT '',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
