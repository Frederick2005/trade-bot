-- ═══════════════════════════════════════════════════════════════════
-- Trading Bot — Supabase Schema
-- Run this in: Supabase Dashboard → SQL Editor → New Query
-- ═══════════════════════════════════════════════════════════════════

-- Enable UUID generation
create extension if not exists "pgcrypto";


-- ── market_data ──────────────────────────────────────────────────────────────
-- Every candle ever received. The bot's raw historical memory.
create table if not exists market_data (
    id          uuid primary key default gen_random_uuid(),
    symbol      text not null,
    timeframe   text not null,
    open_time   timestamptz not null,
    open        numeric not null,
    high        numeric not null,
    low         numeric not null,
    close       numeric not null,
    volume      numeric not null,
    created_at  timestamptz default now(),
    unique(symbol, timeframe, open_time)
);

create index if not exists idx_market_data_symbol_tf
    on market_data(symbol, timeframe, open_time desc);


-- ── trades ────────────────────────────────────────────────────────────────────
-- Every trade the bot opens and closes.
create table if not exists trades (
    id               uuid primary key default gen_random_uuid(),
    symbol           text not null,
    side             text not null check (side in ('LONG', 'SHORT')),
    entry_price      numeric not null,
    exit_price       numeric,
    stop_loss        numeric not null,
    take_profit      numeric not null,
    lot_size         numeric not null,
    profit_loss      numeric,
    profit_pct       numeric,
    status           text not null default 'OPEN'
                         check (status in ('OPEN', 'CLOSED', 'CANCELLED')),
    exit_reason      text check (exit_reason in
                         ('TP_HIT', 'SL_HIT', 'MANUAL', 'TIMEOUT', 'EMERGENCY')),
    strategy_version text not null,
    account_balance  numeric not null,
    order_id         text,
    opened_at        timestamptz not null default now(),
    closed_at        timestamptz
);

create index if not exists idx_trades_status   on trades(status);
create index if not exists idx_trades_symbol   on trades(symbol, opened_at desc);
create index if not exists idx_trades_opened   on trades(opened_at desc);


-- ── trade_context ─────────────────────────────────────────────────────────────
-- Exact market conditions + AI features at the moment of entry.
-- This is what the model trains on.
create table if not exists trade_context (
    id               uuid primary key default gen_random_uuid(),
    trade_id         uuid references trades(id) on delete cascade,

    -- Indicator snapshot (1H)
    ema50_1h         numeric,
    ema200_1h        numeric,
    rsi_1h           numeric,
    atr_1h           numeric,

    -- Indicator snapshot (4H)
    ema50_4h         numeric,
    ema200_4h        numeric,
    rsi_4h           numeric,
    atr_4h           numeric,

    -- Derived features
    price_vs_ema50   numeric,     -- % distance from EMA50
    trend_strength   numeric,     -- (EMA50 - EMA200) / EMA200 * 100
    volatility_pct   numeric,     -- ATR / price * 100
    volume_ratio     numeric,     -- volume / 20-period avg
    ema_gap_pct      numeric,     -- EMA50 vs EMA200 gap %
    candle_body_pct  numeric,     -- body size / full range
    rsi_divergence   numeric,     -- RSI 1H minus RSI 4H

    -- Time context
    hour_of_day      int,
    day_of_week      int,

    -- 4H confluence
    trend_4h         int,         -- 1 = bullish, -1 = bearish

    created_at       timestamptz default now()
);

create index if not exists idx_trade_context_trade
    on trade_context(trade_id);


-- ── decision_log ──────────────────────────────────────────────────────────────
-- Every signal generated — whether the bot entered, skipped, or was blocked.
-- Critical for debugging why the bot didn't trade on a given day.
create table if not exists decision_log (
    id             uuid primary key default gen_random_uuid(),
    symbol         text not null,
    signal_type    text,            -- 'LONG', 'SHORT', null if no signal
    action         text not null    -- 'ENTERED', 'SKIPPED', 'BLOCKED'
                       check (action in ('ENTERED', 'SKIPPED', 'BLOCKED')),
    reason         text not null,
    rsi_at_signal  numeric,
    atr_at_signal  numeric,
    confidence     numeric,         -- AI confidence 0-100
    created_at     timestamptz default now()
);

create index if not exists idx_decision_log_created
    on decision_log(created_at desc);


-- ── training_labels ───────────────────────────────────────────────────────────
-- Closed trades converted into ML training examples.
-- label = 1 (win) or 0 (loss).
create table if not exists training_labels (
    id          uuid primary key default gen_random_uuid(),
    trade_id    uuid references trades(id) on delete cascade,
    features    jsonb not null,     -- full feature vector as JSON
    label       int not null check (label in (0, 1)),
    pnl_pct     numeric,
    created_at  timestamptz default now()
);

create index if not exists idx_training_labels_created
    on training_labels(created_at desc);


-- ── model_versions ────────────────────────────────────────────────────────────
-- Every trained AI model. Never deleted — always versioned.
create table if not exists model_versions (
    id           uuid primary key default gen_random_uuid(),
    version      text not null unique,
    is_active    boolean default false,
    accuracy     numeric,
    precision    numeric,
    recall       numeric,
    f1_score     numeric,
    sharpe_ratio numeric,
    trained_on   int,               -- number of trades in training set
    model_path   text,              -- path to saved .pkl file
    notes        text,
    created_at   timestamptz default now()
);


-- ── param_history ─────────────────────────────────────────────────────────────
-- Every version of the rule-based strategy parameters.
-- The bot never overwrites — it creates new versions.
create table if not exists param_history (
    id               uuid primary key default gen_random_uuid(),
    version          text not null unique,
    is_active        boolean default false,
    rsi_lower        numeric default 45,
    rsi_upper        numeric default 60,
    atr_multiplier   numeric default 1.5,
    ema_gap_min      numeric default 0.0,
    min_volume_ratio numeric default 1.0,
    min_rr           numeric default 2.0,
    reason           text,
    created_at       timestamptz default now()
);

-- Seed the first parameter version
insert into param_history
    (version, is_active, rsi_lower, rsi_upper, atr_multiplier,
     ema_gap_min, min_volume_ratio, min_rr, reason)
values
    ('v1.0', true, 45, 60, 1.5, 0.0, 1.0, 2.0, 'Initial strategy parameters')
on conflict (version) do nothing;


-- ── performance_metrics ───────────────────────────────────────────────────────
-- Weekly performance snapshots. Tracks improvement over time.
create table if not exists performance_metrics (
    id               uuid primary key default gen_random_uuid(),
    period_start     date not null,
    period_end       date not null,
    strategy_version text,
    model_version    text,
    total_trades     int default 0,
    winning_trades   int default 0,
    win_rate         numeric,
    profit_factor    numeric,
    sharpe_ratio     numeric,
    max_drawdown     numeric,
    total_pnl        numeric,
    avg_rsi_wins     numeric,
    avg_rsi_losses   numeric,
    best_hour        int,
    worst_hour       int,
    created_at       timestamptz default now()
);


-- ── bot_logs ──────────────────────────────────────────────────────────────────
-- Structured log storage. Mirrors local log files in the cloud.
create table if not exists bot_logs (
    id         uuid primary key default gen_random_uuid(),
    level      text not null check (level in ('DEBUG','INFO','WARNING','ERROR','CRITICAL')),
    message    text not null,
    context    jsonb,              -- optional extra data
    created_at timestamptz default now()
);

create index if not exists idx_bot_logs_level   on bot_logs(level);
create index if not exists idx_bot_logs_created on bot_logs(created_at desc);


-- ═══════════════════════════════════════════════════════════════════
-- Row Level Security (RLS)
-- Locks all tables to authenticated users only.
-- ═══════════════════════════════════════════════════════════════════

alter table market_data         enable row level security;
alter table trades              enable row level security;
alter table trade_context       enable row level security;
alter table decision_log        enable row level security;
alter table training_labels     enable row level security;
alter table model_versions      enable row level security;
alter table param_history       enable row level security;
alter table performance_metrics enable row level security;
alter table bot_logs            enable row level security;

-- Allow the service role (used by the bot's API key) full access
create policy "service_role_all" on market_data
    for all using (auth.role() = 'service_role');
create policy "service_role_all" on trades
    for all using (auth.role() = 'service_role');
create policy "service_role_all" on trade_context
    for all using (auth.role() = 'service_role');
create policy "service_role_all" on decision_log
    for all using (auth.role() = 'service_role');
create policy "service_role_all" on training_labels
    for all using (auth.role() = 'service_role');
create policy "service_role_all" on model_versions
    for all using (auth.role() = 'service_role');
create policy "service_role_all" on param_history
    for all using (auth.role() = 'service_role');
create policy "service_role_all" on performance_metrics
    for all using (auth.role() = 'service_role');
create policy "service_role_all" on bot_logs
    for all using (auth.role() = 'service_role');