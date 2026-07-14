-- ═══════════════════════════════════════════════════════════════════════════
-- SCHEMA UPGRADE — Based on Crypto AI Trading Data Architecture PDF
-- Run this in: Supabase Dashboard → SQL Editor → New Query
-- Run AFTER your existing schema.sql — this only ADDS new tables/columns
-- ═══════════════════════════════════════════════════════════════════════════

-- ── 1. EXTEND market_data with missing futures fields ───────────────────────
alter table market_data
    add column if not exists quote_volume      numeric,
    add column if not exists num_trades        int,
    add column if not exists buy_volume        numeric,
    add column if not exists sell_volume       numeric,
    add column if not exists vwap              numeric,
    add column if not exists funding_rate      numeric,
    add column if not exists mark_price        numeric,
    add column if not exists index_price       numeric,
    add column if not exists open_interest     numeric,
    add column if not exists liquidation_vol   numeric,
    add column if not exists long_short_ratio  numeric,
    add column if not exists bid_price         numeric,
    add column if not exists ask_price         numeric,
    add column if not exists bid_size          numeric,
    add column if not exists ask_size          numeric,
    add column if not exists spread            numeric,
    add column if not exists premium_index     numeric;


-- ── 2. EXTEND trades with full lifecycle fields ──────────────────────────────
alter table trades
    add column if not exists signal_id             uuid,
    add column if not exists execution_latency_ms  numeric,
    add column if not exists fees_paid             numeric default 0,
    add column if not exists funding_cost          numeric default 0,
    add column if not exists slippage              numeric default 0,
    add column if not exists spread_cost           numeric default 0,
    add column if not exists trailing_stop         numeric,
    add column if not exists max_favourable_excursion numeric,
    add column if not exists max_adverse_excursion    numeric,
    add column if not exists holding_time_minutes  int,
    add column if not exists r_multiple            numeric,
    add column if not exists risk_amount           numeric,
    add column if not exists reward_amount         numeric,
    add column if not exists equity_before         numeric,
    add column if not exists equity_after          numeric,
    add column if not exists margin_used           numeric,
    add column if not exists liquidation_distance  numeric,
    add column if not exists net_profit            numeric,
    add column if not exists leverage_used         int;


-- ── 3. EXTEND trade_context with full time + session features ────────────────
alter table trade_context
    add column if not exists session_name     text,  -- 'ASIAN','LONDON','NEW_YORK','OVERLAP'
    add column if not exists is_weekend       boolean default false,
    add column if not exists is_month_end     boolean default false,
    add column if not exists is_quarter_end   boolean default false,
    add column if not exists week_of_year     int,
    add column if not exists quarter          int,
    -- Additional indicators missing from current context
    add column if not exists macd             numeric,
    add column if not exists macd_signal      numeric,
    add column if not exists macd_histogram   numeric,
    add column if not exists stoch_rsi        numeric,
    add column if not exists adx              numeric,
    add column if not exists cci              numeric,
    add column if not exists obv              numeric,
    add column if not exists bb_upper         numeric,
    add column if not exists bb_lower         numeric,
    add column if not exists bb_width         numeric,
    add column if not exists supertrend       numeric,
    add column if not exists supertrend_dir   int,   -- 1 up, -1 down
    add column if not exists vwap_1h          numeric,
    add column if not exists realized_vol     numeric,
    add column if not exists atr_percentile   numeric,
    add column if not exists market_regime    text;  -- 'TRENDING','RANGING','VOLATILE'


-- ── 4. FULL INDICATORS TABLE — all indicators all timeframes ────────────────
create table if not exists indicators (
    id              uuid primary key default gen_random_uuid(),
    symbol          text        not null,
    timeframe       text        not null,
    open_time       timestamptz not null,

    -- Trend
    ema5            numeric, ema10  numeric, ema20  numeric,
    ema50           numeric, ema100 numeric, ema200 numeric,
    sma20           numeric, sma50  numeric, sma200 numeric,
    vwap            numeric,
    supertrend      numeric, supertrend_direction int,
    parabolic_sar   numeric,
    ichimoku_tenkan numeric, ichimoku_kijun   numeric,
    ichimoku_senkou_a numeric, ichimoku_senkou_b numeric,
    donchian_upper  numeric, donchian_lower   numeric,
    keltner_upper   numeric, keltner_lower    numeric,

    -- Momentum
    rsi14           numeric, stoch_rsi       numeric,
    macd            numeric, macd_signal     numeric, macd_histogram numeric,
    cci             numeric, roc             numeric,
    momentum        numeric, mfi             numeric,

    -- Volatility
    atr14           numeric, bb_upper        numeric,
    bb_middle       numeric, bb_lower        numeric,
    bb_width        numeric, bb_pct_b        numeric,
    realized_vol    numeric, historical_vol  numeric,
    atr_percentile  numeric,

    -- Volume
    obv             numeric, cmf             numeric,
    volume_ratio    numeric,

    -- Structure
    support         numeric, resistance      numeric,
    trend_strength  numeric, trend_direction int,
    volatility_regime text,  -- 'HIGH','NORMAL','LOW'
    market_regime   text,    -- 'TRENDING','RANGING','VOLATILE'

    -- Pivot points
    pivot           numeric, r1 numeric, r2 numeric, r3 numeric,
    s1              numeric, s2 numeric, s3 numeric,

    -- Fibonacci
    fib_236         numeric, fib_382 numeric, fib_500 numeric,
    fib_618         numeric, fib_786 numeric,

    created_at      timestamptz default now(),
    unique(symbol, timeframe, open_time)
);
create index if not exists idx_indicators_sym_tf
    on indicators(symbol, timeframe, open_time desc);
alter table indicators enable row level security;
create policy "anon_all" on indicators for all using (true) with check (true);


-- ── 5. MARKET REGIME TABLE — label every candle ─────────────────────────────
create table if not exists market_regime (
    id              uuid primary key default gen_random_uuid(),
    symbol          text        not null,
    timeframe       text        not null,
    open_time       timestamptz not null,

    regime          text not null, -- 'BULL','BEAR','RANGE','BREAKOUT','BREAKDOWN',
                                   -- 'RECOVERY','ACCUMULATION','DISTRIBUTION',
                                   -- 'HIGH_VOL','LOW_VOL','TRENDING','MEAN_REVERSION',
                                   -- 'EXPANSION','COMPRESSION'
    confidence      numeric,       -- 0-1 how confident in the label
    trend_direction int,           -- 1 up, -1 down, 0 sideways
    volatility_tier text,          -- 'HIGH','NORMAL','LOW'
    momentum_state  text,          -- 'STRONG','WEAK','NEUTRAL','DIVERGING'
    labeled_by      text,          -- 'rule_based', 'ml_model', 'manual'
    created_at      timestamptz default now(),
    unique(symbol, timeframe, open_time)
);
create index if not exists idx_regime_sym_tf
    on market_regime(symbol, timeframe, open_time desc);
alter table market_regime enable row level security;
create policy "anon_all" on market_regime for all using (true) with check (true);


-- ── 6. AI PREDICTION LOGS — every prediction stored ─────────────────────────
create table if not exists ai_prediction_logs (
    id                      uuid primary key default gen_random_uuid(),
    trade_id                uuid references trades(id),
    signal_id               uuid,
    model_version           text not null,
    feature_version         text,
    training_dataset_version text,
    prediction_time         timestamptz not null default now(),

    -- Prediction output
    confidence              numeric,    -- 0-1
    probability_win         numeric,    -- 0-1
    probability_loss        numeric,    -- 0-1
    chosen_action           text,       -- 'LONG','SHORT','FLAT'
    rejected_actions        jsonb,      -- other considered actions

    -- Feature snapshot
    feature_vector          jsonb,      -- full input vector
    feature_importance      jsonb,      -- per-feature importance
    shap_values             jsonb,      -- explainability (if computed)
    reasoning               text,       -- human-readable explanation

    -- Performance
    inference_time_ms       numeric,
    memory_usage_mb         numeric,

    -- Outcome (filled after trade closes)
    prediction_outcome      text,       -- 'CORRECT','INCORRECT','PARTIAL'
    actual_result           text,       -- 'WIN','LOSS'
    outcome_recorded_at     timestamptz,

    created_at              timestamptz default now()
);
create index if not exists idx_ai_pred_model
    on ai_prediction_logs(model_version, prediction_time desc);
create index if not exists idx_ai_pred_trade
    on ai_prediction_logs(trade_id);
alter table ai_prediction_logs enable row level security;
create policy "anon_all" on ai_prediction_logs for all using (true) with check (true);


-- ── 7. RISK METRICS — daily snapshots ───────────────────────────────────────
create table if not exists risk_metrics (
    id                  uuid primary key default gen_random_uuid(),
    snapshot_time       timestamptz not null default now(),
    period              text not null,  -- 'DAILY','WEEKLY','MONTHLY'

    -- Balance
    balance             numeric,
    equity              numeric,
    peak_balance        numeric,

    -- Drawdown
    current_drawdown    numeric,
    max_drawdown        numeric,
    max_drawdown_start  timestamptz,
    max_drawdown_end    timestamptz,
    recovery_time_days  numeric,

    -- Exposure
    total_exposure      numeric,
    portfolio_heat      numeric,
    daily_risk_used     numeric,

    -- Risk measures
    value_at_risk_95    numeric,
    value_at_risk_99    numeric,
    expected_shortfall  numeric,
    kelly_fraction      numeric,
    risk_of_ruin        numeric,

    -- Performance ratios
    sharpe_ratio        numeric,
    sortino_ratio       numeric,
    calmar_ratio        numeric,
    omega_ratio         numeric,
    recovery_factor     numeric,
    profit_factor       numeric,
    expectancy_r        numeric,

    -- Trade stats
    total_trades        int,
    win_rate            numeric,
    avg_win             numeric,
    avg_loss            numeric,

    created_at          timestamptz default now()
);
create index if not exists idx_risk_snapshot
    on risk_metrics(snapshot_time desc);
alter table risk_metrics enable row level security;
create policy "anon_all" on risk_metrics for all using (true) with check (true);


-- ── 8. ERROR ANALYSIS — auto-classify every losing trade ────────────────────
create table if not exists error_analysis (
    id              uuid primary key default gen_random_uuid(),
    trade_id        uuid references trades(id),
    symbol          text,
    error_type      text not null,
    -- Possible values:
    -- 'FALSE_BREAKOUT','FALSE_BREAKDOWN','LATE_ENTRY','EARLY_ENTRY',
    -- 'LATE_EXIT','EARLY_EXIT','HIGH_VOLATILITY','LOW_VOLUME',
    -- 'TREND_REVERSAL','RANGE_FAILURE','LIQUIDITY_SWEEP','NEWS_IMPACT',
    -- 'STOP_TOO_TIGHT','STOP_TOO_WIDE','OVERCONFIDENCE','SIGNAL_CONFLICT',
    -- 'INDICATOR_FAILURE','EXECUTION_FAILURE','UNKNOWN'
    confidence      numeric,    -- how confident in this classification
    description     text,
    rsi_at_entry    numeric,
    volume_at_entry numeric,
    volatility      numeric,
    market_regime   text,
    auto_classified boolean default true,
    reviewed        boolean default false,
    created_at      timestamptz default now()
);
create index if not exists idx_error_trade
    on error_analysis(trade_id);
create index if not exists idx_error_type
    on error_analysis(error_type);
alter table error_analysis enable row level security;
create policy "anon_all" on error_analysis for all using (true) with check (true);


-- ── 9. EXPERIMENT TRACKING ──────────────────────────────────────────────────
create table if not exists experiment_tracking (
    id                  uuid primary key default gen_random_uuid(),
    experiment_name     text not null,
    git_commit_hash     text,
    strategy_version    text,
    model_version       text,
    parameter_set       jsonb,
    indicator_params    jsonb,
    training_data_ver   text,
    backtest_version    text,
    optimizer_version   text,
    exchange            text default 'BINANCE',
    date_tested         date not null default current_date,
    author              text default 'bot',
    notes               text,

    -- Results
    win_rate            numeric,
    profit_factor       numeric,
    sharpe_ratio        numeric,
    max_drawdown        numeric,
    total_return        numeric,
    total_trades        int,

    status              text default 'RUNNING',  -- 'RUNNING','COMPLETED','FAILED'
    created_at          timestamptz default now()
);
alter table experiment_tracking enable row level security;
create policy "anon_all" on experiment_tracking for all using (true) with check (true);


-- ── 10. MACRO DATA — external market signals ─────────────────────────────────
create table if not exists macro_data (
    id                  uuid primary key default gen_random_uuid(),
    recorded_at         timestamptz not null default now(),
    date                date not null,

    -- Crypto sentiment
    fear_greed_index    int,           -- 0-100
    fear_greed_label    text,          -- 'Extreme Fear','Fear','Neutral','Greed','Extreme Greed'
    btc_dominance       numeric,       -- BTC market cap dominance %
    total_market_cap    numeric,
    stablecoin_supply   numeric,

    -- On-chain
    btc_hash_rate       numeric,
    btc_difficulty      numeric,
    exchange_inflows    numeric,
    exchange_outflows   numeric,
    whale_transactions  int,

    -- Macro
    us_10y_yield        numeric,
    dxy                 numeric,       -- US Dollar Index
    gold_price          numeric,
    sp500               numeric,
    vix                 numeric,

    -- Events (boolean flags)
    fomc_today          boolean default false,
    cpi_today           boolean default false,
    ppi_today           boolean default false,
    major_news          boolean default false,
    news_description    text,

    created_at          timestamptz default now(),
    unique(date)
);
alter table macro_data enable row level security;
create policy "anon_all" on macro_data for all using (true) with check (true);


-- ── 11. WALK-FORWARD VALIDATION PARTITIONS ──────────────────────────────────
create table if not exists walk_forward_partitions (
    id              uuid primary key default gen_random_uuid(),
    fold_number     int not null,
    partition_type  text not null,     -- 'TRAIN','VALIDATION','TEST'
    window_type     text not null,     -- 'ROLLING','EXPANDING','WALK_FORWARD'
    symbol          text not null,
    start_date      date not null,
    end_date        date not null,
    num_candles     int,
    num_trades      int,
    win_rate        numeric,
    profit_factor   numeric,
    sharpe_ratio    numeric,
    data_leakage_check boolean default false,
    created_at      timestamptz default now()
);
alter table walk_forward_partitions enable row level security;
create policy "anon_all" on walk_forward_partitions for all using (true) with check (true);


-- ── 12. DATA QUALITY LOG ────────────────────────────────────────────────────
create table if not exists data_quality_log (
    id              uuid primary key default gen_random_uuid(),
    check_time      timestamptz default now(),
    table_name      text not null,
    check_type      text not null,
    -- Types: 'DUPLICATE_CANDLES','MISSING_CANDLES','INCORRECT_TIMESTAMP',
    --        'FUTURE_LEAKAGE','OUTLIER','CORRUPTED_ROW','BROKEN_FK',
    --        'INCORRECT_PNL','NEGATIVE_BALANCE','INVALID_PRICE'
    severity        text,              -- 'CRITICAL','WARNING','INFO'
    records_affected int default 0,
    description     text,
    resolved        boolean default false,
    resolved_at     timestamptz,
    created_at      timestamptz default now()
);
alter table data_quality_log enable row level security;
create policy "anon_all" on data_quality_log for all using (true) with check (true);


-- ═══════════════════════════════════════════════════════════════════════════
-- MATERIALIZED VIEWS — ML-ready analytical views
-- ═══════════════════════════════════════════════════════════════════════════

-- ML feature view — every completed trade with all features for training
create or replace view ml_feature_view as
select
    t.id                                            as trade_id,
    t.symbol,
    t.side,
    t.opened_at,
    t.closed_at,
    t.profit_loss,
    t.profit_pct,
    t.exit_reason,
    t.r_multiple,
    t.holding_time_minutes,
    t.max_favourable_excursion,
    t.max_adverse_excursion,
    t.fees_paid,
    t.leverage_used,

    -- Entry indicators
    tc.ema50_1h, tc.ema200_1h, tc.rsi_1h, tc.atr_1h,
    tc.ema50_4h, tc.ema200_4h, tc.rsi_4h, tc.atr_4h,
    tc.price_vs_ema50, tc.trend_strength, tc.volatility_pct,
    tc.volume_ratio, tc.ema_gap_pct, tc.candle_body_pct,
    tc.rsi_divergence, tc.hour_of_day, tc.day_of_week, tc.trend_4h,
    tc.session_name, tc.is_weekend,
    tc.macd, tc.macd_signal, tc.adx, tc.cci, tc.stoch_rsi,
    tc.bb_width, tc.supertrend_dir, tc.realized_vol,
    tc.atr_percentile, tc.market_regime,

    -- AI prediction
    ap.confidence                                   as ai_confidence,
    ap.probability_win,
    ap.chosen_action,

    -- Labels for supervised learning
    case when t.profit_loss > 0 then 1 else 0 end  as label_win,
    t.profit_pct                                    as label_return_pct,
    t.r_multiple                                    as label_r_multiple,
    t.exit_reason                                   as label_exit_reason,
    tc.market_regime                                as label_regime

from trades t
left join trade_context     tc on tc.trade_id = t.id
left join ai_prediction_logs ap on ap.trade_id = t.id
where t.status = 'CLOSED';


-- Performance summary view
create or replace view performance_summary_view as
select
    date_trunc('day', closed_at)    as day,
    symbol,
    count(*)                        as total_trades,
    sum(case when profit_loss > 0 then 1 else 0 end) as wins,
    avg(case when profit_loss > 0 then 1.0 else 0.0 end) as win_rate,
    sum(profit_loss)                as total_pnl,
    avg(profit_loss)                as avg_pnl,
    sum(case when profit_loss > 0 then profit_loss else 0 end) as gross_profit,
    sum(case when profit_loss < 0 then abs(profit_loss) else 0 end) as gross_loss,
    max(profit_loss)                as best_trade,
    min(profit_loss)                as worst_trade
from trades
where status = 'CLOSED'
group by date_trunc('day', closed_at), symbol
order by day desc;


-- ═══════════════════════════════════════════════════════════════════════════
-- USEFUL QUERIES
-- Run these to analyse your data
-- ═══════════════════════════════════════════════════════════════════════════

-- Q1: Win rate by RSI bucket
-- select
--     floor(rsi_1h / 5) * 5               as rsi_bucket,
--     count(*)                             as total,
--     sum(case when t.profit_loss > 0 then 1 else 0 end) as wins,
--     avg(case when t.profit_loss > 0 then 1.0 else 0.0 end) as win_rate
-- from trade_context tc
-- join trades t on tc.trade_id = t.id
-- where t.status = 'CLOSED'
-- group by rsi_bucket order by rsi_bucket;

-- Q2: Win rate by session
-- select session_name, count(*) as trades,
--     avg(case when t.profit_loss > 0 then 1.0 else 0.0 end) as win_rate
-- from trade_context tc join trades t on tc.trade_id = t.id
-- where t.status = 'CLOSED'
-- group by session_name order by win_rate desc;

-- Q3: Best hours
-- select hour_of_day, count(*) as trades,
--     avg(case when t.profit_loss > 0 then 1.0 else 0.0 end) as win_rate
-- from trade_context tc join trades t on tc.trade_id = t.id
-- where t.status = 'CLOSED'
-- group by hour_of_day order by win_rate desc;

-- Q4: Error analysis breakdown
-- select error_type, count(*) as count,
--     avg(confidence) as avg_confidence
-- from error_analysis
-- group by error_type order by count desc;

-- Q5: Model version performance
-- select model_version, count(*) as predictions,
--     avg(confidence) as avg_confidence,
--     sum(case when prediction_outcome = 'CORRECT' then 1 else 0 end) as correct,
--     avg(case when prediction_outcome = 'CORRECT' then 1.0 else 0.0 end) as accuracy
-- from ai_prediction_logs
-- where prediction_outcome is not null
-- group by model_version order by accuracy desc;