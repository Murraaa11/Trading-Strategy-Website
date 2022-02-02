import numpy as np
import pandas as pd
import math
import random
from scipy import stats
import statsmodels.api as sm


metrics_list = ['Ticker', 'Round-trip trades', 'Total return (%)', 'Buy & hold return (%)',
                'Winning trades (%)', 'Ratio of average winning amount to average losing amount (%)',
                'Maximum drawdown (%)']



def filter_rules(df, h_length, thre):
    num = int(len(df.columns) / 2)
    ticker = df.columns[:num]
    realized_r = pd.DataFrame(columns=ticker)
    strategy = pd.DataFrame(columns=ticker)
    for n in range(num):
        stra = []
        rr = []
        for i in range(h_length, (len(df) - 1)):
            if df[ticker[n]][i] / max(df[ticker[n]][(i - h_length):i]) > (1 + thre):
                stra.append(1)
                rr.append(df['{}_r'.format(ticker[n])][i + 1])
            elif df[ticker[n]][i] / min(df[ticker[n]][(i - h_length):i]) < (1 - thre):
                stra.append(-1)
                rr.append(-df['{}_r'.format(ticker[n])][i + 1])
            else:
                stra.append(0)
                rr.append(0)
        realized_r[ticker[n]] = rr
        strategy[ticker[n]] = stra
    return strategy, realized_r


def SMA(df, sn, ln, thre):
    num = int(len(df.columns) / 2)
    ticker = df.columns[:num]
    realized_r = pd.DataFrame(columns=ticker)
    strategy = pd.DataFrame(columns=ticker)
    for n in range(num):
        stra = []
        rr = []
        for i in range((ln - 1), (len(df) - 1)):
            short = np.mean(df[ticker[n]][(i - sn + 1):(i + 1)])
            long = np.mean(df[ticker[n]][(i - ln + 1):(i + 1)])
            r = (short - long) / long
            if r > thre:
                stra.append(1)
                rr.append(df['{}_r'.format(ticker[n])][i + 1])
            elif r < -thre:
                stra.append(-1)
                rr.append(-df['{}_r'.format(ticker[n])][i + 1])
            else:
                stra.append(0)
                rr.append(0)
        realized_r[ticker[n]] = rr
        strategy[ticker[n]] = stra
    return strategy, realized_r


def EMA(df, sn, ln, thre):
    num = int(len(df.columns) / 2)
    ticker = df.columns[:num]
    realized_r = pd.DataFrame(columns=ticker)
    strategy = pd.DataFrame(columns=ticker)
    beta_s = 2 / (sn + 1)
    beta_l = 2 / (ln + 1)
    for n in range(num):
        stra = []
        rr = []
        for i in range((ln - 1), (len(df) - 1)):
            short = 0
            long = 0
            multiplier_s = beta_s
            multiplier_l = beta_l
            for d in range(i, i - sn, -1):
                short += multiplier_s * df[ticker[n]][d]
                multiplier_s *= (1 - beta_s)
            for d in range(i, i - ln, -1):
                long += multiplier_l * df[ticker[n]][d]
                multiplier_l *= (1 - beta_l)
            r = (short - long) / long
            if r > thre:
                stra.append(1)
                rr.append(df['{}_r'.format(ticker[n])][i + 1])
            elif r < -thre:
                stra.append(-1)
                rr.append(-df['{}_r'.format(ticker[n])][i + 1])
            else:
                stra.append(0)
                rr.append(0)
        realized_r[ticker[n]] = rr
        strategy[ticker[n]] = stra
    return strategy, realized_r


def channel_breakouts(df, h_length, k, L):
    num = int(len(df.columns) / 2)
    ticker = df.columns[:num]
    realized_r = pd.DataFrame(columns=ticker)
    strategy = pd.DataFrame(columns=ticker)
    start_i = max(h_length, L)
    for n in range(num):
        stra = []
        rr = []
        for i in range(start_i, (len(df) - 1)):
            B = k * np.std(df['{}_r'.format(ticker[n])][(i - L + 1):(i + 1)])
            maximum = max(df[ticker[n]][(i - h_length):i])
            minimum = min(df[ticker[n]][(i - h_length):i])
            if i == start_i or (i > start_i and stra[-1] == 0):
                if df[ticker[n]][i] > (1 + B) * maximum:
                    stra.append(1)
                    rr.append(df['{}_r'.format(ticker[n])][i + 1])
                elif df[ticker[n]][i] < (1 - B) * minimum:
                    stra.append(-1)
                    rr.append(-df['{}_r'.format(ticker[n])][i + 1])
                else:
                    stra.append(0)
                    rr.append(0)
            elif stra[-1] == 1:
                if df[ticker[n]][i] < (1 - B) * minimum:
                    stra.append(0)
                    rr.append(0)
                else:
                    stra.append(1)
                    rr.append(df['{}_r'.format(ticker[n])][i + 1])
            else:
                if df[ticker[n]][i] > (1 + B) * maximum:
                    stra.append(0)
                    rr.append(0)
                else:
                    stra.append(-1)
                    rr.append(-df['{}_r'.format(ticker[n])][i + 1])
        realized_r[ticker[n]] = rr
        strategy[ticker[n]] = stra
    return strategy, realized_r


def metrics(strategy, realized_r):
    w = [i for i in realized_r if i > 0]
    l = [i for i in realized_r if i < 0]
    ratio = np.mean(w) / abs(np.mean(l)) * 100
    roundtrip_trade = 0
    for i in range(len(strategy)):
        if i == 0:
            if strategy[i] != 0:
                roundtrip_trade += 0.5
        else:
            if abs(strategy[i] - strategy[i - 1]) == 1:
                roundtrip_trade += 0.5
            elif abs(strategy[i] - strategy[i - 1]) == 2:
                roundtrip_trade += 1
    accumulated_return = np.cumsum(realized_r) + np.ones(len(realized_r))
    total_return = np.sum(realized_r) * 100
    BH_return = (np.prod(realized_r + np.ones(len(realized_r))) - 1) * 100
    winning_trades = len(w) / len(realized_r) * 100
    max_drawdown = 0
    for i in range(len(realized_r)):
        maximum = max(accumulated_return[:(i + 1)])
        if maximum - accumulated_return[i] > max_drawdown:
            max_drawdown = maximum - accumulated_return[i]
    return [roundtrip_trade, total_return, BH_return, winning_trades, ratio, max_drawdown * 100]


def ar1_model(df):
    num = int(len(df.columns) / 2)
    ticker = df.columns[:num]
    coef_ = {}
    res_ = {}
    for n in range(num):
        r = df[['{}_r'.format(ticker[n])]].dropna()
        x = r.iloc[0:(len(r)-1), :].reset_index(drop=True)
        y = r.iloc[1:len(r), :].reset_index(drop=True)
        x = sm.add_constant(x)
        model = sm.OLS(y, x).fit()
        coef = model.params
        res = model.resid
        coef_[ticker[n]] = coef.tolist()
        res_[ticker[n]] = res.tolist()
    return coef_, res_


def bootstrap(df, *args, boot_size=10):
    num = int(len(df.columns) / 2)
    ticker = df.columns[:num]
    boot_result = pd.DataFrame(columns=metrics_list, index = range(1, boot_size*num+1))
    coef_, res_ = ar1_model(df)
    for i in range(boot_size):
        boot_df = df
        np.random.seed(i)
        for n in range(num):
            coef = coef_[ticker[n]]
            res = res_[ticker[n]]
            sample_res = np.random.choice(res, size=len(res))
            sample_r = np.zeros(len(res) + 2)
            sample_r[1] = df['{}_r'.format(ticker[n])].tolist()[-(len(res)+1)]
            sample_p = np.zeros(len(res) + 2)
            sample_p[0] = df[ticker[n]].tolist()[-(len(res)+2)]
            sample_p[1] = df[ticker[n]].tolist()[-(len(res)+2)]
            for j in range(len(res)):
                rj_1 = coef[0] + coef[1] * sample_r[1 + j] + sample_res[j]
                pj_1 = sample_p[1 + j] * math.exp(rj_1)
                sample_r[2 + j] = rj_1
                sample_p[2 + j] = pj_1
            start = len(df)-len(res)
            boot_df['{}_r'.format(ticker[n])][start:] = sample_r[2:]
            boot_df[ticker[n]][start:] = sample_p[2:]
        if args[0] == 'FR':
            s, r = filter_rules(boot_df, args[1], args[2])
        elif args[0] == 'SMA':
            s, r = SMA(boot_df, args[1], args[2], args[3])
        elif args[0] == 'EMA':
            s, r = EMA(boot_df, args[1], args[2], args[3])
        else:
            s, r = channel_breakouts(boot_df, args[1], args[2], args[3])
        for n in range(num):
            boot_result.iloc[(i*num+n), :] = [ticker[n]] + metrics(s[ticker[n]], r[ticker[n]])
    return boot_result


