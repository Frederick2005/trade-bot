-- Every trained model version — never deleted, always versioned
model_versions (
  id            uuid primary key default gen_random_uuid(),
  version       text not null unique,     -- 'v1.0', 'v1.1' etc
  is_active     boolean default false,    -- only one active at a time
  accuracy      numeric,                  -- on validation set
  precision     numeric,
  recall        numeric,
  sharpe_ratio  numeric,                  -- backtest result
  trained_on    int,                      -- number of trades trained on
  model_path    text,                     -- local file path to saved model
  created_at    timestamptz default now(),
  notes         text
)

-- Each closed trade converted to a labelled training example
training_labels (
  id            uuid primary key default gen_random_uuid(),
  trade_id      uuid references trades(id),
  features      jsonb not null,           -- the feature vector at entry
  label         int not null,             -- 1 = win, 0 = loss
  pnl_pct       numeric,                  -- actual % gain/loss
  created_at    timestamptz default now()
)