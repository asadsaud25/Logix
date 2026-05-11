"""
ml_engine_real.py — Prophet + LSTM hybrid forecasting
======================================================
This module contains the EXACT Prophet and LSTM code from:
  - ProphetLSTM_Colab.ipynb  (ForecastEngine: data gen, Prophet fit, LSTM train,
                               blend weight optimisation, safety stock)
  - BigBillionDays_SupplyChain.ipynb  (supplier records, network config)

Entry point:
    result = real_pipeline_available()   → True / False
    result = run_real_forecast(run_type) → dict with same schema as ml_engine.py

Graceful fallback:
    If prophet or torch are not installed, returns None from run_real_forecast()
    and ml_engine.py falls back to the numpy engine automatically.

Install real deps with:
    pip install prophet torch pandas
"""

import numpy as np
import warnings
import logging
import time
import json
warnings.filterwarnings('ignore')
logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)

# ── Dependency check ──────────────────────────────────────────────────────────
def real_pipeline_available():
    try:
        import torch
        from prophet import Prophet
        import pandas as pd
        return True
    except ImportError:
        return False


# ══════════════════════════════════════════════════════════════════════════════
# DATA GENERATION  (from ProphetLSTM_Colab.ipynb cell 7)
# Produces 5-year daily sales for 20 products × 3 warehouses
# ══════════════════════════════════════════════════════════════════════════════

def generate_sales_data():
    """
    Exact synthetic data generation from ProphetLSTM_Colab.ipynb cell 7.
    Returns: sales_df, train_df, val_df, test_df, series_list, products_df,
             warehouses_df, suppliers_df, prod_supplier
    """
    import pandas as pd
    import random

    NUM_PRODUCTS   = 20
    NUM_WAREHOUSES = 3
    epsilon        = 1e-8

    np.random.seed(42); random.seed(42)

    # Products
    products = []
    for pid in range(1, NUM_PRODUCTS + 1):
        cat  = random.choice(['FMCG', 'Electronics', 'Clothing'])
        base = (np.random.randint(40, 120) if cat == 'FMCG'
                else np.random.randint(5, 30)  if cat == 'Electronics'
                else np.random.randint(10, 60))
        products.append({
            'product_id':   pid, 'category': cat, 'base_demand': base,
            'unit_price':   round(np.random.uniform(10, 500), 2),
            'yoy_growth':   round(np.random.uniform(-0.05, 0.25), 3),
            'intermittent': (cat == 'Electronics' and np.random.rand() < 0.4),
        })
    products_df   = pd.DataFrame(products)

    warehouses_df = pd.DataFrame([
        {'warehouse_id': w, 'region_factor': round(np.random.uniform(0.7, 1.4), 2)}
        for w in range(1, 4)
    ])
    suppliers_df  = pd.DataFrame([
        {'supplier_id': s, 'lead_time_mean': np.random.randint(3, 14),
         'lead_time_std': round(np.random.uniform(0.5, 3), 2)}
        for s in range(1, 6)
    ])
    prod_supplier = {
        p: random.sample(list(suppliers_df.supplier_id), random.randint(1, 3))
        for p in products_df.product_id
    }

    # Date range 2020-2025
    dates  = pd.date_range('2020-01-01', '2025-01-31')
    date_df = pd.DataFrame({
        'ds':         dates, 'doy': dates.dayofyear, 'dow': dates.dayofweek,
        'month':      dates.month, 'year': dates.year,
        'is_weekend': (dates.dayofweek >= 5).astype(int),
    })

    sales_df = date_df.merge(products_df, how='cross').merge(warehouses_df, how='cross')
    sales_df = sales_df.sort_values(['product_id', 'warehouse_id', 'ds']).reset_index(drop=True)
    n = len(sales_df)

    # Demand components
    yoy_mult = (1 + sales_df['yoy_growth']) ** (sales_df['year'] - 2020)
    doy      = sales_df['doy'].values
    cat_arr  = sales_df['category'].values
    seas     = np.ones(n)
    seas[cat_arr == 'FMCG']        = 1 + 0.15 * np.sin(2 * np.pi * doy[cat_arr == 'FMCG'] / 365)
    seas[cat_arr == 'Electronics'] = 1 + 2.0  * np.exp(-0.5 * ((doy[cat_arr == 'Electronics'] - 330) / 30) ** 2)
    seas[cat_arr == 'Clothing']    = (
        1 + 0.5 * np.exp(-0.5 * ((doy[cat_arr == 'Clothing'] - 100) / 25) ** 2)
          + 0.5 * np.exp(-0.5 * ((doy[cat_arr == 'Clothing'] - 270) / 25) ** 2))

    smooth_noise = np.random.normal(0, 0.25, n)
    sales_df['_n'] = smooth_noise
    sales_df['_n'] = sales_df.groupby(['product_id', 'warehouse_id'])['_n'].transform(
        lambda x: x.ewm(span=7).mean())
    smooth  = np.exp(sales_df['_n'])
    wk_mult = np.where(
        (sales_df['is_weekend'] == 1) & (cat_arr != 'Electronics'), 1.2,
        np.where((sales_df['is_weekend'] == 1) & (cat_arr == 'Electronics'), 0.7, 1.0))

    # Promotions
    promo_starts = (np.random.rand(n) < 0.013).astype(int)
    sales_df['promo_start']    = promo_starts
    sales_df['promotion_flag'] = sales_df.groupby(['product_id', 'warehouse_id'])['promo_start'].transform(
        lambda x: x.rolling(7, min_periods=1).max())
    promo_lift           = np.random.uniform(1.3, 3.0, n)
    sales_df['promo_lift'] = np.where(sales_df['promotion_flag'] == 1, promo_lift, 1.0)

    zero_p  = np.where(sales_df['intermittent'], 0.35, 0.02)
    active  = (np.random.rand(n) > zero_p).astype(float)
    lam     = np.maximum(0,
        sales_df['base_demand'] * sales_df['region_factor'] *
        yoy_mult * seas * smooth * wk_mult * sales_df['promo_lift'] * active)
    sales_df['quantity_sold'] = np.random.poisson(lam)

    sales_df['supplier_id'] = [np.random.choice(prod_supplier[p]) for p in sales_df['product_id']]
    sales_df = sales_df.merge(suppliers_df, on='supplier_id', how='left')
    sales_df['lead_time'] = np.maximum(1,
        np.random.normal(sales_df['lead_time_mean'], sales_df['lead_time_std']).astype(int))

    # Splits
    TRAIN_END  = '2024-11-30'; VAL_START = '2024-12-01'; VAL_END  = '2024-12-31'
    TEST_START = '2025-01-01'; TEST_END  = '2025-01-31'

    train_df = sales_df[sales_df['ds'] <= TRAIN_END].copy()
    val_df   = sales_df[(sales_df['ds'] >= VAL_START) & (sales_df['ds'] <= VAL_END)].copy()
    test_df  = sales_df[(sales_df['ds'] >= TEST_START) & (sales_df['ds'] <= TEST_END)].copy()

    series_list = sorted(train_df.groupby(['product_id', 'warehouse_id']).groups.keys())

    return {
        'sales_df':     sales_df,
        'train_df':     train_df,
        'val_df':       val_df,
        'test_df':      test_df,
        'series_list':  series_list,
        'products_df':  products_df,
        'warehouses_df':warehouses_df,
        'suppliers_df': suppliers_df,
        'prod_supplier':prod_supplier,
        'TRAIN_END':    TRAIN_END,
        'VAL_START':    VAL_START, 'VAL_END': VAL_END,
        'TEST_START':   TEST_START,'TEST_END': TEST_END,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PROPHET  (from ProphetLSTM_Colab.ipynb cell 11)
# ══════════════════════════════════════════════════════════════════════════════

def fit_prophet_series(s_train, series_mode='additive'):
    """Fit a Prophet model for one product-warehouse series. (Exact notebook code)"""
    from prophet import Prophet
    df_p = s_train[['ds', 'quantity_sold', 'promotion_flag']].copy()
    df_p = df_p.rename(columns={'quantity_sold': 'y'})

    m = Prophet(
        changepoint_prior_scale  = 0.15,
        seasonality_prior_scale  = 10.0,
        seasonality_mode         = series_mode,
        yearly_seasonality       = False,
        weekly_seasonality       = False,
        daily_seasonality        = False,
        interval_width           = 0.80,
    )
    m.add_seasonality('yearly', period=365.25, fourier_order=10)
    m.add_seasonality('weekly', period=7,      fourier_order=3)
    m.add_regressor('promotion_flag', standardize=False)
    m.fit(df_p)
    return m


def predict_prophet(model, future_ds, future_promo):
    """Predict over a date range with known future promo flags. (Exact notebook code)"""
    import pandas as pd
    future = pd.DataFrame({'ds': future_ds, 'promotion_flag': future_promo})
    fc     = model.predict(future)
    return np.maximum(0, fc['yhat'].values)


def run_prophet_all_series(data, log_cb):
    """
    Fits Prophet for all 60 series. Returns:
        val_prophet_7, val_prophet_30, test_prophet_7, test_prophet_30
    """
    sales_df    = data['sales_df']
    series_list = data['series_list']
    products_df = data['products_df']

    val_prophet_7, val_prophet_30   = {}, {}
    test_prophet_7, test_prophet_30 = {}, {}
    prophet_models                  = {}

    t0 = time.time()
    log_cb(f'Fitting Prophet for {len(series_list)} series (changepoint_prior=0.15, fourier_order=10)...')

    for i, (p, w) in enumerate(series_list):
        s    = sales_df[(sales_df['product_id'] == p) & (sales_df['warehouse_id'] == w)].sort_values('ds')
        s_tr = s[s['ds'] <= data['TRAIN_END']]

        cat  = products_df[products_df['product_id'] == p]['category'].values[0]
        mode = 'multiplicative' if cat == 'Electronics' else 'additive'

        mdl = fit_prophet_series(s_tr, mode)
        prophet_models[(p, w)] = mdl

        # Validation predictions (Dec 2024)
        s_val = s[(s['ds'] >= data['VAL_START']) & (s['ds'] <= data['VAL_END'])]
        p_val = predict_prophet(mdl, s_val['ds'], s_val['promotion_flag'].values)
        val_prophet_7[(p, w)]  = float(p_val[:7].sum())
        val_prophet_30[(p, w)] = float(p_val.sum())

        # Test predictions (Jan 2025)
        s_test = s[(s['ds'] >= data['TEST_START']) & (s['ds'] <= data['TEST_END'])]
        p_test = predict_prophet(mdl, s_test['ds'], s_test['promotion_flag'].values)
        test_prophet_7[(p, w)]  = float(p_test[:7].sum())
        test_prophet_30[(p, w)] = float(p_test.sum())

        if (i + 1) % 20 == 0 or (i + 1) == len(series_list):
            log_cb(f'  Prophet [{i+1:>2}/{len(series_list)}]  {time.time()-t0:.0f}s elapsed')

    log_cb(f'Prophet complete — {len(series_list)} series fitted in {time.time()-t0:.0f}s')
    return prophet_models, val_prophet_7, val_prophet_30, test_prophet_7, test_prophet_30


# ══════════════════════════════════════════════════════════════════════════════
# LSTM  (from ProphetLSTM_Colab.ipynb cells 13 + 15)
# ══════════════════════════════════════════════════════════════════════════════

def _build_lstm_classes():
    """Lazily build Dataset and Model classes after torch is confirmed available."""
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader

    class DemandDataset(Dataset):
        def __init__(self, X, y, seq_len=30):
            self.X = torch.FloatTensor(X)
            self.y = torch.FloatTensor(y)
            self.seq = seq_len
        def __len__(self): return len(self.y) - self.seq
        def __getitem__(self, i):
            return self.X[i:i + self.seq], self.y[i + self.seq]

    class DemandLSTM(nn.Module):
        def __init__(self, input_size=8, hidden_size=64, num_layers=2, dropout=0.2):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                batch_first=True, dropout=dropout)
            self.head = nn.Sequential(
                nn.Linear(hidden_size, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :]).squeeze(-1)

    return DemandDataset, DemandLSTM, DataLoader


def make_lstm_features(df_):
    """8 features: demand + cyclical time encodings + promo flag. (Exact notebook code)"""
    doy = df_['ds'].dt.dayofyear.values
    dow = df_['ds'].dt.dayofweek.values
    mon = df_['ds'].dt.month.values
    return np.column_stack([
        df_['quantity_sold'].values,
        np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25),
        np.sin(2 * np.pi * dow / 7),      np.cos(2 * np.pi * dow / 7),
        np.sin(2 * np.pi * mon / 12),     np.cos(2 * np.pi * mon / 12),
        df_['promotion_flag'].values,
    ]).astype(np.float32)


def train_lstm(X_raw, y_raw, seq_len=30, epochs=150, lr=5e-3,
               batch_size=64, patience=20, device=None):
    """Train one LSTM series. (Exact notebook code, device-agnostic)"""
    import torch
    import torch.nn as nn
    epsilon = 1e-8
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    DemandDataset, DemandLSTM, DataLoader = _build_lstm_classes()

    X_mean = X_raw.mean(0); X_std = X_raw.std(0) + epsilon
    y_mean = y_raw.mean();  y_std = y_raw.std()  + epsilon
    X_n = ((X_raw - X_mean) / X_std).astype(np.float32)
    y_n = ((y_raw - y_mean) / y_std).astype(np.float32)

    split = int(len(y_n) * 0.9)
    ds_tr = DemandDataset(X_n[:split], y_n[:split], seq_len)
    ds_vl = DemandDataset(X_n[split:], y_n[split:], seq_len)
    if len(ds_tr) == 0 or len(ds_vl) == 0:
        ds_tr = DemandDataset(X_n, y_n, seq_len); ds_vl = ds_tr

    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,  drop_last=False)
    dl_vl = DataLoader(ds_vl, batch_size=batch_size, shuffle=False, drop_last=False)

    model = DemandLSTM(input_size=X_raw.shape[1]).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=8, factor=0.5)
    crit  = nn.MSELoss()

    best_val = float('inf'); best_state = None; no_improve = 0
    for ep in range(epochs):
        model.train()
        for Xb, yb in dl_tr:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(Xb), yb)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        vl = 0
        with torch.no_grad():
            for Xb, yb in dl_vl:
                vl += crit(model(Xb.to(device)), yb.to(device)).item()
        vl /= max(len(dl_vl), 1)
        sched.step(vl)
        if vl < best_val:
            best_val = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    model.load_state_dict(best_state)
    return model, X_mean, X_std, y_mean, y_std


def predict_lstm_horizon(model, X_seed_raw, X_future_raw,
                          X_mean, X_std, y_mean, y_std, seq_len=30, device=None):
    """
    Seed LSTM state from last seq_len days, predict autoregressively.
    (Exact notebook code)
    """
    import torch
    epsilon = 1e-8
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    def norm(x): return ((x - X_mean) / (X_std + epsilon)).astype(np.float32)
    def denorm(y): return float(y) * y_std + y_mean

    seed = torch.FloatTensor(norm(X_seed_raw[-seq_len:])).unsqueeze(0).to(device)
    with torch.no_grad():
        _, (h, c) = model.lstm(seed)

    preds = []; prev_y_norm = float((X_seed_raw[-1, 0] - y_mean) / (y_std + epsilon))
    for row in X_future_raw:
        x_n    = norm(row.copy()); x_n[0] = prev_y_norm
        x_t    = torch.FloatTensor(x_n).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            out, (h, c) = model.lstm(x_t, (h, c))
            y_hat       = model.head(out[:, 0, :]).item()
        preds.append(max(0.0, denorm(y_hat)))
        prev_y_norm = y_hat
    return np.array(preds)


def run_lstm_all_series(data, log_cb):
    """Train LSTM for all 60 series."""
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_cb(f'Training LSTM for {len(data["series_list"])} series on {device} (epochs=150, patience=20, seq_len=30)...')
    if torch.cuda.is_available():
        log_cb(f'GPU: {torch.cuda.get_device_name(0)}')

    sales_df    = data['sales_df']
    series_list = data['series_list']
    SEQ_LEN     = 30

    val_lstm_7,  val_lstm_30  = {}, {}
    test_lstm_7, test_lstm_30 = {}, {}
    lstm_models               = {}

    t0 = time.time()
    for i, (p, w) in enumerate(series_list):
        s    = sales_df[(sales_df['product_id'] == p) & (sales_df['warehouse_id'] == w)].sort_values('ds')
        s_tr = s[s['ds'] <= data['TRAIN_END']]

        X_tr = make_lstm_features(s_tr)
        y_tr = s_tr['quantity_sold'].values.astype(np.float32)

        model, Xm, Xs, ym, ys = train_lstm(X_tr, y_tr, seq_len=SEQ_LEN,
                                             epochs=150, lr=5e-3, patience=20, device=device)
        lstm_models[(p, w)] = (model, Xm, Xs, ym, ys)

        # Validation
        s_val  = s[(s['ds'] >= data['VAL_START']) & (s['ds'] <= data['VAL_END'])]
        X_vf   = make_lstm_features(s_val)
        l_val  = predict_lstm_horizon(model, X_tr, X_vf, Xm, Xs, ym, ys, SEQ_LEN, device)
        val_lstm_7[(p, w)]  = float(l_val[:7].sum())
        val_lstm_30[(p, w)] = float(l_val.sum())

        # Test
        s_test = s[(s['ds'] >= data['TEST_START']) & (s['ds'] <= data['TEST_END'])]
        X_tf   = make_lstm_features(s_test)
        l_test = predict_lstm_horizon(model, X_tr, X_tf, Xm, Xs, ym, ys, SEQ_LEN, device)
        test_lstm_7[(p, w)]  = float(l_test[:7].sum())
        test_lstm_30[(p, w)] = float(l_test.sum())

        if (i + 1) % 20 == 0 or (i + 1) == len(series_list):
            elapsed = time.time() - t0
            eta     = elapsed / (i + 1) * (len(series_list) - i - 1)
            log_cb(f'  LSTM [{i+1:>2}/{len(series_list)}]  {elapsed:.0f}s elapsed  ETA {eta:.0f}s')

    log_cb(f'LSTM complete — all {len(series_list)} series trained in {time.time()-t0:.0f}s')
    return lstm_models, val_lstm_7, val_lstm_30, test_lstm_7, test_lstm_30


# ══════════════════════════════════════════════════════════════════════════════
# BLEND WEIGHT OPTIMISATION  (from ProphetLSTM_Colab.ipynb cell 17)
# ══════════════════════════════════════════════════════════════════════════════

def optimise_blend_weights(val_p7, val_l7, val_p30, val_l30,
                            val_act_7d, val_act_30d, log_cb):
    """
    Optimise blend weights to maximise W10% on Dec 2024 validation.
    (Exact notebook code)
    """
    from scipy.optimize import minimize_scalar
    epsilon = 1e-8

    def blend(w_p, pred_p, pred_l):
        return {k: w_p * pred_p[k] + (1 - w_p) * pred_l[k] for k in pred_p}

    def neg_w10(w_p, pred_p, pred_l, actuals):
        bl   = blend(w_p, pred_p, pred_l)
        errs = np.array([abs(bl[k] - actuals.get(k, 0)) / (actuals.get(k, 0) + epsilon) * 100
                         for k in bl])
        return -(errs <= 10).mean()

    res7  = minimize_scalar(lambda w: neg_w10(w, val_p7,  val_l7,  val_act_7d),  bounds=(0, 1), method='bounded')
    res30 = minimize_scalar(lambda w: neg_w10(w, val_p30, val_l30, val_act_30d), bounds=(0, 1), method='bounded')
    w7, w30 = float(res7.x), float(res30.x)

    # Validation metrics
    def show_val(w_p, pred_p, pred_l, actuals, label):
        bl   = {k: w_p * pred_p[k] + (1 - w_p) * pred_l[k] for k in pred_p}
        errs = np.array([abs(bl[k] - actuals.get(k, 0)) / (actuals.get(k, 0) + epsilon) * 100
                         for k in bl])
        w10   = (errs <= 10).mean() * 100
        w20   = (errs <= 20).mean() * 100
        wmape = np.mean(errs)
        log_cb(f'  Val {label}: W10={w10:.1f}%  W20={w20:.1f}%  WMAPE={wmape:.1f}%  max={errs.max():.1f}%')
        return wmape

    log_cb(f'Blend weights optimised: 7d → Prophet={w7:.2f} LSTM={1-w7:.2f} | 30d → Prophet={w30:.2f} LSTM={1-w30:.2f}')
    wmape_7d  = show_val(w7,  val_p7,  val_l7,  val_act_7d,  '7d ')
    wmape_30d = show_val(w30, val_p30, val_l30, val_act_30d, '30d')

    return w7, w30, wmape_7d, wmape_30d


def build_actuals(sales_df, start, end):
    """Aggregate actuals for a date window."""
    import pandas as pd
    df_ = (sales_df[(sales_df['ds'] >= start) & (sales_df['ds'] <= end)]
           .groupby(['product_id', 'warehouse_id'])['quantity_sold'].sum().reset_index())
    return {(r.product_id, r.warehouse_id): r.quantity_sold for _, r in df_.iterrows()}


# ══════════════════════════════════════════════════════════════════════════════
# SAFETY STOCK  (from ProphetLSTM_Colab.ipynb cell 26)
# ══════════════════════════════════════════════════════════════════════════════

def compute_safety_stocks(data, blend_30d, test_prophet_30, test_lstm_30):
    """
    Safety stock = 1.65 × uncertainty × √(lead_time)
    Uncertainty = |Prophet - LSTM| / 30  (daily disagreement)
    (Exact notebook code)
    """
    import pandas as pd
    train_df    = data['train_df']
    products_df = data['products_df']

    lt_avg = train_df.groupby(['product_id', 'warehouse_id'])['lead_time'].mean().reset_index()
    lt_avg.columns = ['product_id', 'warehouse_id', 'avg_lt']

    rows = []
    for (p, w) in data['series_list']:
        lt    = lt_avg[(lt_avg.product_id == p) & (lt_avg.warehouse_id == w)]['avg_lt'].values[0]
        p30   = blend_30d[(p, w)]
        uncert = abs(test_prophet_30[(p, w)] - test_lstm_30[(p, w)]) / 30
        ss_95  = round(1.65 * uncert * np.sqrt(lt))
        rop    = round(p30 / 30 * lt + ss_95)
        cat    = products_df[products_df['product_id'] == p]['category'].values[0]
        rows.append({
            'product_id': p, 'warehouse_id': w, 'category': cat,
            'forecast_30d': round(p30, 1),
            'lead_time':    round(lt, 1),
            'safety_stock': int(max(0, ss_95)),
            'reorder_point': int(max(0, rop)),
        })
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def run_real_forecast(run_type='full_retrain', log_cb=None):
    """
    Run the full real Prophet+LSTM pipeline.
    Returns same schema as ml_engine.run_pipeline() for drop-in compatibility.
    Returns None if dependencies are missing.
    """
    if not real_pipeline_available():
        return None

    if log_cb is None:
        log_cb = lambda msg: None

    epsilon = 1e-8
    log_cb('Real Prophet+LSTM pipeline starting...')
    log_cb(f'Run type: {run_type}')

    # ── 1. Generate data ────────────────────────────────────────────────────
    log_cb('Generating 5-year synthetic sales history (2020-2025, 20 products × 3 warehouses)...')
    data = generate_sales_data()
    log_cb(f'Dataset: {len(data["sales_df"]):,} rows | {len(data["series_list"])} series')

    # ── 2. Prophet ──────────────────────────────────────────────────────────
    (prophet_models,
     val_p7, val_p30,
     test_p7, test_p30) = run_prophet_all_series(data, log_cb)

    # ── 3. LSTM ─────────────────────────────────────────────────────────────
    if run_type in ('full_retrain', 'incremental'):
        (lstm_models,
         val_l7, val_l30,
         test_l7, test_l30) = run_lstm_all_series(data, log_cb)
    else:
        # forecast_only: use Prophet only, set LSTM weight = 0
        log_cb('Forecast-only mode: using Prophet predictions (no LSTM retraining)')
        val_l7  = {k: v for k, v in val_p7.items()}
        val_l30 = {k: v for k, v in val_p30.items()}
        test_l7  = {k: v for k, v in test_p7.items()}
        test_l30 = {k: v for k, v in test_p30.items()}
        lstm_models = {}

    # ── 4. Build validation actuals ─────────────────────────────────────────
    log_cb('Building validation actuals for blend weight optimisation...')
    sales_df   = data['sales_df']
    val_act_7d  = build_actuals(sales_df, data['VAL_START'], '2024-12-07')
    val_act_30d = build_actuals(sales_df, data['VAL_START'], data['VAL_END'])

    # ── 5. Optimise blend weights ────────────────────────────────────────────
    w7, w30, wmape_7d, wmape_30d = optimise_blend_weights(
        val_p7, val_l7, val_p30, val_l30, val_act_7d, val_act_30d, log_cb)

    # ── 6. Final blended forecasts ───────────────────────────────────────────
    blend_7d  = {k: w7  * test_p7[k]  + (1 - w7)  * test_l7[k]  for k in test_p7}
    blend_30d = {k: w30 * test_p30[k] + (1 - w30) * test_l30[k] for k in test_p30}

    # ── 7. Safety stocks ─────────────────────────────────────────────────────
    log_cb('Computing safety stocks (1.65σ, 95% service level)...')
    safety_rows = compute_safety_stocks(data, blend_30d, test_p30, test_l30)

    # ── 8. Per-category accuracy ─────────────────────────────────────────────
    test_act_7d  = build_actuals(sales_df, data['TEST_START'], '2025-01-07')
    test_act_30d = build_actuals(sales_df, data['TEST_START'], data['TEST_END'])

    cat_accuracy = {}
    for cat in ('FMCG', 'Electronics', 'Clothing'):
        keys  = [(p, w) for (p, w) in data['series_list']
                 if data['products_df'][data['products_df']['product_id'] == p]['category'].values[0] == cat]
        f30   = sum(blend_30d[k] for k in keys)
        a30   = sum(test_act_30d.get(k, 0) for k in keys)
        f7    = sum(blend_7d[k]  for k in keys)
        a7    = sum(test_act_7d.get(k, 0) for k in keys)
        cat_accuracy[cat] = {
            'forecast':   round(f30, 1),
            'actual':     round(a30, 1),
            'wmape':      round(abs(f30 - a30) / max(1, a30) * 100, 2),
            'forecast_7d':round(f7, 1),
            'actual_7d':  round(a7, 1),
            'blend_w7':   round(w7, 3),
            'blend_w30':  round(w30, 3),
        }

    # ── 9. Assemble forecast_rows (same schema as ml_engine.py) ──────────────
    from datetime import datetime
    month = datetime.utcnow().strftime('%Y-%m')
    products_df = data['products_df']

    forecast_rows = []
    for (p, w) in data['series_list']:
        f7  = round(blend_7d[(p, w)],  1)
        f30 = round(blend_30d[(p, w)], 1)

        # Confidence = 1 - normalised blend disagreement
        prophet_30 = test_p30[(p, w)]; lstm_30 = test_l30[(p, w)]
        disagree   = abs(prophet_30 - lstm_30) / max(1, (prophet_30 + lstm_30) / 2)
        conf       = round(max(0.55, min(0.98, 1 - disagree * 0.4)), 3)

        cat = products_df[products_df['product_id'] == p]['category'].values[0]
        forecast_rows.append({
            'product_id':   p,
            'warehouse_id': w,
            'category':     cat,
            'forecast_7d':  f7,
            'forecast_30d': f30,
            'daily_rate':   round(f30 / 30, 3),
            'confidence':   conf,
            'month_label':  month,
            'prophet_30d':  round(prophet_30, 1),
            'lstm_30d':     round(lstm_30, 1),
        })

    # Log per-series performance summary
    errs_30 = []
    for (p, w) in data['series_list']:
        act = test_act_30d.get((p, w), 0)
        pred = blend_30d[(p, w)]
        if act > 0:
            errs_30.append(abs(pred - act) / act * 100)
    errs_arr = np.array(errs_30)
    log_cb(f'Final 30d forecast performance (Jan 2025):')
    log_cb(f'  ≤10%: {(errs_arr<=10).sum()}/{len(errs_arr)} ({(errs_arr<=10).mean()*100:.0f}%)')
    log_cb(f'  ≤20%: {(errs_arr<=20).sum()}/{len(errs_arr)} ({(errs_arr<=20).mean()*100:.0f}%)')
    log_cb(f'  WMAPE: {errs_arr.mean():.1f}%  |  Max: {errs_arr.max():.1f}%')

    log_cb('Real pipeline complete ✓')

    return {
        'forecast_rows':  forecast_rows,
        'safety_rows':    safety_rows,
        'wmape_7d':       round(wmape_7d, 2),
        'wmape_30d':      round(wmape_30d, 2),
        'w_prophet_7':    round(w7, 3),
        'w_prophet_30':   round(w30, 3),
        'cat_accuracy':   cat_accuracy,
        'n_series':       len(forecast_rows),
        'engine':         'Prophet+LSTM',
    }


# ══════════════════════════════════════════════════════════════════════════════
# SHARED CATALOGUES
# (previously in ml_engine.py — moved here as the single source of truth)
# ══════════════════════════════════════════════════════════════════════════════

import random as _random
from scipy.optimize import linprog
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

SUPPLIER_CATALOGUE = [
    {'id':1,'name':'Factory Noida',   'avg_lt':4, 'lt_std':1.0,'reliability':0.95,'fill_mean':0.97,'price_mult':0.82,'capacity':2000,'min_order':10},
    {'id':2,'name':'Factory Pune',    'avg_lt':5, 'lt_std':1.5,'reliability':0.91,'fill_mean':0.93,'price_mult':0.78,'capacity':1800,'min_order':20},
    {'id':3,'name':'Factory Chennai', 'avg_lt':3, 'lt_std':0.8,'reliability':0.97,'fill_mean':0.98,'price_mult':0.75,'capacity':1500,'min_order':15},
    {'id':4,'name':'Factory Surat',   'avg_lt':6, 'lt_std':2.0,'reliability':0.88,'fill_mean':0.91,'price_mult':0.80,'capacity':1200,'min_order':25},
    {'id':5,'name':'Factory Kolkata', 'avg_lt':10,'lt_std':3.0,'reliability':0.80,'fill_mean':0.87,'price_mult':0.72,'capacity':1000,'min_order':50},
]

PRODUCT_CATALOGUE = [
    {'id': pid, 'category': ['FMCG','Electronics','Clothing'][(pid-1)%3],
     'unit_price': round(10 + (pid * 23.7) % 490, 2)}
    for pid in range(1, 21)
]


# ══════════════════════════════════════════════════════════════════════════════
# SUPPLIER SCORER  (GradientBoosting on synthetic transactions)
# ══════════════════════════════════════════════════════════════════════════════

class SupplierScorer:
    FEAT_COLS = ['avg_actual_lt','std_actual_lt','on_time_rate',
                 'avg_fill_rate','avg_price_ratio','std_price_ratio']

    def __init__(self):
        self.model  = None
        self.scaler = StandardScaler()
        self.scores = {}

    def _generate_transactions(self, n=2000):
        rows = []
        rng  = np.random.default_rng(7)
        for _ in range(n):
            s   = _random.choice(SUPPLIER_CATALOGUE)
            qty = int(rng.integers(10, 500))
            qf  = 1 + 0.08 * (qty / 500)
            lt  = max(1, int(rng.normal(s['avg_lt'] * qf, s['lt_std'])))
            ot  = int(lt <= s['avg_lt'] + 1)
            fr  = float(np.clip(rng.normal(s['fill_mean'], 0.04), 0.4, 1.0))
            surge = rng.choice([1.0,1.05,1.15,1.30], p=[0.85,0.08,0.05,0.02])
            pr  = float(s['price_mult'] * surge * rng.normal(1, 0.02))
            rows.append({'sid':s['id'],'lt':lt,'ot':ot,'fr':fr,'pr':pr})
        return rows

    def fit_and_score(self):
        txns = self._generate_transactions(2000)
        agg  = defaultdict(lambda: {'lts':[],'ots':[],'frs':[],'prs':[]})
        for t in txns:
            agg[t['sid']]['lts'].append(t['lt'])
            agg[t['sid']]['ots'].append(t['ot'])
            agg[t['sid']]['frs'].append(t['fr'])
            agg[t['sid']]['prs'].append(t['pr'])

        X, y, sids = [], [], []
        for sid, d in agg.items():
            lts = np.array(d['lts']); prs = np.array(d['prs'])
            feat   = [lts.mean(), lts.std(), np.mean(d['ots']),
                      np.mean(d['frs']), prs.mean(), prs.std()]
            lt_cons = 1 - lts.std() / max(0.1, lts.mean())
            target  = (np.mean(d['ots'])*40 + np.mean(d['frs'])*30 +
                       np.clip(1-prs.mean(), 0, 1)*20 + lt_cons*10)
            X.append(feat); y.append(target); sids.append(sid)

        X  = np.array(X, dtype=np.float32)
        Xs = self.scaler.fit_transform(X)
        self.model = GradientBoostingRegressor(
            n_estimators=150, max_depth=3, learning_rate=0.05,
            random_state=42, subsample=0.8)
        self.model.fit(Xs, y)

        raw = self.model.predict(Xs)
        mn, mx = raw.min(), raw.max()
        for sid, r in zip(sids, raw):
            self.scores[sid] = round(float((r - mn) / max(1e-8, mx - mn) * 40 + 55), 1)

        return self.scores, self.model.feature_importances_.tolist()

    def get_scores(self):
        if not self.scores:
            self.fit_and_score()
        return self.scores


# ══════════════════════════════════════════════════════════════════════════════
# PROCUREMENT LP  (scipy HiGHS)
# ══════════════════════════════════════════════════════════════════════════════

class ProcurementLP:
    def __init__(self, score_dict):
        self.scores = score_dict

    def solve(self, demand_30d, unit_price):
        n      = len(SUPPLIER_CATALOGUE)
        s_norm = np.array([self.scores.get(s['id'], 50) / 100 for s in SUPPLIER_CATALOGUE])
        costs  = np.array([s['price_mult'] * unit_price for s in SUPPLIER_CATALOGUE])
        adj    = costs * (2 - s_norm)

        A_ub = -np.ones((1, n))
        b_ub = np.array([-demand_30d])
        bounds = [(s['min_order'], s['capacity']) for s in SUPPLIER_CATALOGUE]

        res = linprog(adj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.status == 0:
            order = {s['id']: max(0, round(float(q)))
                     for s, q in zip(SUPPLIER_CATALOGUE, res.x)}
        else:
            best  = max(SUPPLIER_CATALOGUE, key=lambda s: self.scores.get(s['id'], 0))
            order = {s['id']: (int(demand_30d) if s['id'] == best['id'] else 0)
                     for s in SUPPLIER_CATALOGUE}

        total_cost = sum(order[s['id']] * s['price_mult'] * unit_price for s in SUPPLIER_CATALOGUE)
        naive_cost = demand_30d * min(s['price_mult'] for s in SUPPLIER_CATALOGUE) * unit_price
        cost_saving = round(max(0, (naive_cost - total_cost) / max(1, naive_cost) * 100), 1)
        best_sup    = max(order, key=order.get)

        return {
            'order_split':          {str(k): v for k, v in order.items()},
            'demand_30d':           int(demand_30d),
            'total_cost':           round(total_cost, 2),
            'cost_saving_pct':      cost_saving,
            'recommended_supplier': best_sup,
            'unit_price':           round(unit_price, 2),
        }


# ══════════════════════════════════════════════════════════════════════════════
# INVENTORY REBALANCING LP
# ══════════════════════════════════════════════════════════════════════════════

class InventoryLP:
    TRANSFER_COST = 5.0

    def solve(self, stocks, velocities):
        total   = sum(stocks.values())
        v_sum   = max(1e-8, sum(velocities.values()))
        targets = {wid: total * velocities[wid] / v_sum for wid in stocks}
        surplus = {wid: stocks[wid] - targets[wid] for wid in stocks}

        givers = sorted([(wid, s) for wid, s in surplus.items() if s > 5],  key=lambda x: -x[1])
        takers = sorted([(wid, -s) for wid, s in surplus.items() if s < -5], key=lambda x: -x[1])

        transfers = []
        for g_wid, g_avail in givers:
            for t_wid, t_need in takers:
                if g_avail <= 0 or t_need <= 0:
                    continue
                qty = min(g_avail, t_need)
                transfers.append({'from_wh': g_wid, 'to_wh': t_wid,
                                   'qty': round(qty), 'cost': round(qty * self.TRANSFER_COST, 2)})
                g_avail -= qty; t_need -= qty
        return transfers


# ══════════════════════════════════════════════════════════════════════════════
# 90-DAY (s,Q) SIMULATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class SimulationEngine:
    SIM_DAYS    = 90
    SAFETY_DAYS = 5

    def run(self, forecast_rows, score_dict, disruption_wh=None, extra_safety=3):
        rng    = np.random.default_rng(99)
        rates  = {(r['product_id'], r['warehouse_id']): r['daily_rate'] for r in forecast_rows}
        best_s = max(SUPPLIER_CATALOGUE, key=lambda s: score_dict.get(s['id'], 0))

        inventory = {(r['product_id'], r['warehouse_id']): int(r['daily_rate'] * 15)
                     for r in forecast_rows}
        pending   = []
        kpi       = dict(holding=0., ordering=0., stockout=0.,
                         sold=0, lost=0, orders=0, on_time=0, late=0)
        daily_log = []

        for day in range(1, self.SIM_DAYS + 1):
            still = []
            for o in pending:
                if o['arrive'] <= day:
                    inventory[o['key']] = inventory.get(o['key'], 0) + o['qty']
                    kpi['on_time' if o['arrive'] <= o['promise'] else 'late'] += 1
                else:
                    still.append(o)
            pending = still

            dh = ds = 0.; sold = lost = n_ord = 0
            for r in forecast_rows:
                key = (r['product_id'], r['warehouse_id'])
                pid, wid = key
                rate   = rates.get(key, 1.0)
                demand = int(rng.poisson(rate))
                stock  = inventory.get(key, 0)
                fill   = min(stock, demand)
                inventory[key] = stock - fill
                sold += fill; lost += demand - fill

                prod_idx = (pid - 1) % 20
                hold_c  = [0.5,1.2,2.5,0.8,1.5,1.0,2.0,0.9,1.8,0.7,
                            1.1,2.2,0.6,1.4,1.9,0.8,1.3,2.1,1.0,1.6][prod_idx]
                sout_pen= [8,20,35,10,15,12,25,9,18,7,
                            11,22,8,14,28,10,13,21,9,16][prod_idx]
                dh += inventory[key] * hold_c
                ds += (demand - fill) * sout_pen

                sd  = self.SAFETY_DAYS + (extra_safety if wid == disruption_wh else 0)
                rop = rate * (best_s['avg_lt'] + sd)
                if inventory.get(key, 0) <= rop:
                    qty = max(best_s['min_order'], int(rate * (best_s['avg_lt'] + sd + 15)))
                    lt_actual = max(1, int(rng.normal(best_s['avg_lt'], best_s['lt_std'])))
                    pending.append({'key': key, 'qty': qty,
                                    'arrive': day + lt_actual, 'promise': day + best_s['avg_lt']})
                    kpi['ordering'] += 100 + qty * best_s['price_mult'] * 50
                    kpi['orders'] += 1; n_ord += 1

            kpi['holding'] += dh; kpi['stockout'] += ds
            kpi['sold'] += sold;  kpi['lost'] += lost
            daily_log.append({'day': day, 'holding': round(dh,2),
                               'stockout': round(ds,2), 'sold': sold, 'lost': lost,
                               'service_level': round(sold/max(1,sold+lost),4),
                               'inventory': sum(inventory.values()), 'orders': n_ord})

        kpi['total_cost']    = round(kpi['holding'] + kpi['ordering'] + kpi['stockout'], 2)
        kpi['service_level'] = round(kpi['sold'] / max(1, kpi['sold'] + kpi['lost']), 4)
        kpi['on_time_rate']  = round(kpi['on_time'] / max(1, kpi['on_time'] + kpi['late']), 4)
        kpi['daily_log']     = daily_log
        return kpi


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE ORCHESTRATOR
# Drop-in replacement for ml_engine.run_pipeline(db, run_id, run_type, log_cb)
# Raises RuntimeError if Prophet / PyTorch are not installed.
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(db, run_id, run_type, log_cb=None):
    """
    Full SCM pipeline using the real Prophet+LSTM engine.

    Parameters
    ----------
    db       : NeonDB connection (already open, caller closes it)
    run_id   : int   — id of the ml_runs row created before this call
    run_type : str   — 'full_retrain' | 'incremental' | 'forecast_only'
    log_cb   : callable(str) | None

    Returns dict with the same schema as the old ml_engine.run_pipeline().
    Raises RuntimeError if Prophet or PyTorch are not installed.
    """
    def log(msg):
        full = f'[ML] {msg}'
        if log_cb:
            log_cb(full)
        return full

    # ── Dependency gate ───────────────────────────────────────────────────────
    if not real_pipeline_available():
        msg = (
            'Prophet and/or PyTorch not installed.\n'
            'Run: pip install prophet torch pandas\n'
            'Then restart the server and try again.'
        )
        log(msg)
        db.execute(
            "UPDATE ml_runs SET status='failed', finished_at=NOW(), log_text=? WHERE id=?",
            (msg, run_id))
        db.commit()
        raise RuntimeError(msg)

    # ── 1. Real Prophet+LSTM forecasts ────────────────────────────────────────
    real = run_real_forecast(run_type=run_type, log_cb=lambda m: log(m))
    forecast_rows = real['forecast_rows']
    safety_rows   = real['safety_rows']
    wmape_7d      = float(real['wmape_7d'])
    wmape_30d     = float(real['wmape_30d'])
    cat_accuracy  = real['cat_accuracy']

    # ── 2. Write forecasts to DB ──────────────────────────────────────────────
    log(f'Writing {len(forecast_rows)} forecast rows to DB...')
    db.execute('DELETE FROM forecasts WHERE ml_run_id=?', (run_id,))
    for r in forecast_rows:
        db.execute(
            'INSERT INTO forecasts (ml_run_id,product_id,warehouse_id,category,'
            'forecast_7d,forecast_30d,daily_rate,confidence,month_label) '
            'VALUES (?,?,?,?,?,?,?,?,?) RETURNING id',
            (run_id, int(r['product_id']), int(r['warehouse_id']), r['category'],
             float(r['forecast_7d']), float(r['forecast_30d']), float(r['daily_rate']),
             float(r['confidence']), r['month_label']))

    # ── 3. Safety stock recommendations ──────────────────────────────────────
    for r in safety_rows:
        db.execute(
            'INSERT INTO recommendations (role,rec_type,title,detail_json,confidence,priority) '
            'VALUES (?,?,?,?,?,?) RETURNING id',
            ('inventory', 'safety_stock',
             f'Safety Stock — P{r["product_id"]} WH-{r["warehouse_id"]}',
             json.dumps(r), 0.95, 'P2'))

    log(f'Forecasts written: {len(forecast_rows)} rows | Safety stock rows: {len(safety_rows)}')

    # ── 4. Supplier scoring ───────────────────────────────────────────────────
    log('Training GradientBoosting supplier scorer (2,000 transactions)...')
    scorer = SupplierScorer()
    score_dict, feat_imp = scorer.fit_and_score()
    log('Supplier scores: ' + ', '.join(f'S{k}={v:.0f}' for k, v in sorted(score_dict.items())))

    scores_payload = {
        'scores': score_dict,
        'feature_importances': {
            SupplierScorer.FEAT_COLS[i]: round(float(v), 4)
            for i, v in enumerate(feat_imp[:len(SupplierScorer.FEAT_COLS)])
        },
        'suppliers': [
            {**{k: s[k] for k in ('id','name','avg_lt','lt_std','reliability',
                                   'fill_mean','price_mult','capacity')},
             'score': score_dict.get(s['id'], 50)}
            for s in SUPPLIER_CATALOGUE
        ]
    }
    db.execute(
        "INSERT INTO recommendations (role,rec_type,title,detail_json,confidence,priority,status) "
        "VALUES (?,?,?,?,?,?,?) RETURNING id",
        ('supplychain', 'supplier_scores', 'Updated Supplier ML Scores',
         json.dumps(scores_payload), 0.93, 'P3', 'approved'))

    # ── 5. Procurement LP ─────────────────────────────────────────────────────
    log('Running procurement LP for 20 products...')
    lp = ProcurementLP(score_dict)
    for prod in PRODUCT_CATALOGUE:
        total_demand = sum(r['forecast_30d'] for r in forecast_rows
                           if r['product_id'] == prod['id'])
        detail   = lp.solve(total_demand, prod['unit_price'])
        priority = 'P1' if prod['id'] <= 5 else 'P2' if prod['id'] <= 12 else 'P3'
        db.execute(
            "INSERT INTO recommendations (role,rec_type,title,detail_json,confidence,priority) "
            "VALUES (?,?,?,?,?,?) RETURNING id",
            ('procurement', 'order_split',
             f"Order plan — Product {prod['id']} ({prod['category']})",
             json.dumps(detail),
             round(float(score_dict.get(detail['recommended_supplier'], 70)) / 100, 2),
             priority))

    # ── 6. Inventory velocity + rebalancing ───────────────────────────────────
    log('Running inventory velocity analysis and rebalancing LP...')
    inv_lp  = InventoryLP()
    rng_inv = np.random.default_rng(13)
    alert_count = 0

    for prod in PRODUCT_CATALOGUE:
        stocks = {}; velocities = {}
        for wid in range(1, 4):
            daily = next((r['daily_rate'] for r in forecast_rows
                          if r['product_id'] == prod['id'] and r['warehouse_id'] == wid), 1.0)
            stocks[wid]     = max(0, int(rng_inv.normal(daily * 20, daily * 5)))
            velocities[wid] = round(float(rng_inv.uniform(0.3, 3.5)), 2)

        for wid in range(1, 4):
            daily = next((r['daily_rate'] for r in forecast_rows
                          if r['product_id'] == prod['id'] and r['warehouse_id'] == wid), 1.0)
            cover = round(stocks[wid] / max(0.1, daily), 1)
            vel   = velocities[wid]

            if cover < 3:
                sev = 'P1'; atype = 'stockout'
                body = f'Only {cover}d of cover remaining at WH-{wid}. Immediate reorder needed.'
            elif vel > 2.5:
                sev = 'P2'; atype = 'fast_mover'
                body = f'Selling {vel:.1f}× faster than forecast at WH-{wid}. Risk of stockout in {cover}d.'
            elif vel < 0.4 and stocks[wid] > 50:
                sev = 'P3'; atype = 'ghost'
                body = f'Only {vel:.2f}× of forecast selling. {stocks[wid]} units sitting idle at WH-{wid}.'
            else:
                continue

            db.execute(
                "INSERT INTO alerts (target_role,alert_type,severity,title,body,detail_json) "
                "VALUES (?,?,?,?,?,?) RETURNING id",
                ('inventory', atype, sev,
                 f'{atype.replace("_"," ").title()} — P{prod["id"]} WH-{wid}',
                 body,
                 json.dumps({'product_id': prod['id'], 'warehouse_id': wid,
                              'days_cover': cover, 'velocity_ratio': vel,
                              'stock': stocks[wid]})))
            alert_count += 1

        for t in inv_lp.solve(stocks, velocities):
            db.execute(
                "INSERT INTO recommendations (role,rec_type,title,detail_json,confidence,priority) "
                "VALUES (?,?,?,?,?,?) RETURNING id",
                ('inventory', 'rebalance',
                 f'Rebalance P{prod["id"]}: WH-{t["from_wh"]} → WH-{t["to_wh"]}',
                 json.dumps({**t, 'product_id': prod['id'],
                              'velocity_from': round(velocities[t['from_wh']], 2),
                              'velocity_to':   round(velocities[t['to_wh']], 2)}),
                 round(float(min(velocities.values()) / max(0.1, max(velocities.values()))), 2),
                 'P2'))

    log(f'Inventory analysis complete: {alert_count} alerts raised')

    # ── 7. 90-day simulation ──────────────────────────────────────────────────
    log('Running 90-day (s,Q) inventory simulation...')
    sim = SimulationEngine()
    kpi = sim.run(forecast_rows, score_dict)
    log(f'Simulation: service_level={kpi["service_level"]*100:.1f}% '
        f'on_time={kpi["on_time_rate"]*100:.1f}% total_cost=₹{kpi["total_cost"]:,.0f}')

    db.execute(
        "INSERT INTO recommendations (role,rec_type,title,detail_json,confidence,priority,status) "
        "VALUES (?,?,?,?,?,?,?) RETURNING id",
        ('inventory', 'simulation', '90-Day Simulation Results',
         json.dumps({'service_level': kpi['service_level'], 'on_time_rate': kpi['on_time_rate'],
                     'total_cost': kpi['total_cost'], 'total_sold': kpi['sold'],
                     'total_lost': kpi['lost'], 'orders_placed': kpi['orders'],
                     'daily_log': kpi['daily_log']}),
         0.95, 'P3', 'approved'))

    # ── 8. Finalise ml_runs row ───────────────────────────────────────────────
    log_text = (
        f'[Real Prophet+LSTM engine]\n'
        f'WMAPE 7d={wmape_7d}%  30d={wmape_30d}%\n'
        f'Forecasts={len(forecast_rows)}  Alerts={alert_count}\n'
        '[ML] ✓ Pipeline complete'
    )
    db.execute(
        "UPDATE ml_runs SET status='complete', finished_at=NOW(), "
        "wmape_7d=?, wmape_30d=?, log_text=? WHERE id=?",
        (wmape_7d, wmape_30d, log_text, run_id))
    db.commit()

    return {
        'run_id':       run_id,
        'status':       'complete',
        'engine':       'Prophet+LSTM',
        'wmape_7d':     wmape_7d,
        'wmape_30d':    wmape_30d,
        'forecasts':    len(forecast_rows),
        'alerts':       alert_count,
        'score_dict':   score_dict,
        'cat_accuracy': cat_accuracy,
        'sim_kpi':      {k: v for k, v in kpi.items() if k != 'daily_log'},
        'log':          log_text,
    }