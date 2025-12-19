#!/usr/bin/env python3
"""
Preprocess transactions dataset for downstream modelling and graph/node construction.
This module reads a raw transactions CSV, computes a set of categorical/derived
features, clusters merchants, buckets amounts and times, encodes categorical
fields into integer ids, and writes both a processed CSV and a pickle file of
encoders/metadata.
Primary features produced
- merchant_cluster: integer cluster id for each merchant (KMeans over merchant-level stats)
- merchant_id: dense integer id per merchant (enumeration of merchants seen in the file)
- user_id: dense integer id per card holder (enumeration of cc_num)
- amt_bin: quantile-based bucket id computed on log(1 + amt)
- time_bucket: integer bucket of the transaction hour (hour // time_bucket_hours)
- category_small_id: integer id for category after collapsing low-frequency categories into 'other'
- gender_id, city_id, job_id: integer ids obtained by enumerating unique values in the file
- age, age_bucket: age (years) computed from dob and bucketed by configured edges
Expected input columns (at minimum)
- 'merchant' (string/identifier)
- 'amt' (numeric)
- 'category' (string)
- 'trans_date_trans_time' (parsable datetime string)
- 'cc_num' (card identifier)
- 'dob' (date of birth; parsable datetime string)
Optional but used if present:
- 'merch_lat', 'merch_long' (floats used for merchant clustering)
- 'gender', 'city', 'job' (categorical fields encoded to ids)
- 'is_fraud' (if present, the script currently filters to rows with is_fraud == 1)
Command-line usage
- The script is invokable from the command line with arguments:
    --input               Path to raw CSV (default 'data/raw/transactions.csv')
    --out                 Path to write processed CSV (default 'data/preprocessed/processed.csv')
    --encoders            Path to write encoders pickle (default 'data/encoders/encoders.pkl')
    --k_merchant          Number of merchant clusters for KMeans (default 60)
    --amt_bins            Number of quantile bins for log-amount (default 10)
    --time_bucket_hours   Size of each time bucket in hours (default 8)
    --top_k_cat           Number of top categories to keep; others collapsed to 'other' (default 14)
Saved outputs
- Processed CSV: original rows augmented with the derived columns listed above.
- Encoders pickle (dict) containing:
    - 'merchant_cluster_map': mapping merchant -> cluster id
    - 'merchant_meta': {'scaler': fitted StandardScaler, 'kmeans': fitted KMeans}
    - 'merchant_id_map', 'user_id_map': mappings to dense ids
    - 'amt_edges': list of log-amount quantile edges
    - 'n_amt_bins', 'time_bucket_hours'
    - 'top_cats': list of kept category names
    - 'gender', 'city', 'job', 'age_bucket': maps used to encode corresponding fields
Important implementation notes and assumptions
- Merchant-level clustering: merchant statistics are aggregated using mean/std/count
    of amt, merchant location mean, and the modal category. The modal category is
    factorized and used as an input feature. Missing amt_std values are set to 0.0.
    StandardScaler is fit on these features prior to KMeans. KMeans is created with
    random_state=0 and n_init=20 for reproducibility.
- Amount binning: log1p is applied to amounts and quantile edges are computed.
    Edges are uniqued and searchsorted is used to assign bins; bins are clipped
    to [0, n_amt_bins-1].
- Time bucketing: trans_date_trans_time is parsed to a pandas.Timestamp; the
    hour is extracted and bucketed by integer division with bucket_hours.
- Category collapsing: top_k categories by frequency are kept; remaining entries
    are labeled 'other'. A factorize is used to produce integer ids.
- DOB -> age: age is computed as the integer number of years between a fixed
    reference date ('2025-11-17') and the dob. Age buckets are defined by
    [0,18,25,35,45,60,150] with right=False (left-inclusive).
- If the input contains an 'is_fraud' column, the script currently filters to
    rows where is_fraud == 1 before computing encoders and features.
- Missing values: many intermediate steps call .fillna(0) or otherwise
    substitute defaults; downstream consumers should be aware of these defaults.
- When mapping merchants to clusters, merchants unseen in the merchant_cluster_map
    are filled with the integer value equal to k_merchant (i.e., an out-of-range
    cluster id used as a sentinel).
Dependencies
- numpy, pandas, scikit-learn
Performance and scaling
- Merchant clustering operates on merchant-level aggregated rows; for extremely
    large numbers of distinct merchants, memory/time for KMeans may become
    significant. Consider sampling merchants or increasing compute resources.
- Reading/writing of CSVs is done with pandas and may be improved using chunking
    or more efficient on-disk formats (Parquet) for very large datasets.
Example CLI
    python scripts/preprocess_transactions_improved.py \
        --input data/raw/transactions.csv \
        --out data/preprocessed/processed.csv \
        --encoders data/encoders/encoders.pkl \
        --k_merchant 60 --amt_bins 10 --time_bucket_hours 8 --top_k_cat 14
"""
"""
Preprocess transactions: merchant clustering, log-amt quantile bins,
time buckets, produce encoders.pkl and a preprocessed csv.

Usage:
  python scripts/preprocess_transactions_improved.py --input data/raw/transactions.csv --out data/preprocessed/processed.csv
"""

import os, argparse, pickle
import numpy as np
import pandas as pd


def build_merchant_clusters(df, K=60):
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    # build merchant-level stats
    m_stats = df.groupby('merchant').agg({
        'amt': ['mean', 'std', 'count'],
        'category': lambda s: s.mode().iloc[0] if len(s)>0 else 'UNK',
        'merch_lat': 'mean',
        'merch_long': 'mean'
    }).reset_index()
    m_stats.columns = ['merchant', 'amt_mean', 'amt_std', 'cnt', 'top_cat', 'merch_lat', 'merch_long']
    # replace nan std
    m_stats['amt_std'] = m_stats['amt_std'].fillna(0.0)
    # factorize top_cat
    m_stats['top_cat_idx'] = pd.factorize(m_stats['top_cat'])[0]
    Xm = m_stats[['amt_mean','amt_std','cnt','top_cat_idx','merch_lat','merch_long']].fillna(0).values
    scaler = StandardScaler()
    Xm_s = scaler.fit_transform(Xm)
    km = KMeans(n_clusters=K, random_state=0, n_init=20).fit(Xm_s)
    m_stats['merchant_cluster'] = km.labels_.astype(int)
    merchant_cluster_map = dict(zip(m_stats['merchant'], m_stats['merchant_cluster']))
    # save scaler and kmeans to enc map
    return merchant_cluster_map, {'scaler': scaler, 'kmeans': km}

def compute_amt_bins(df, n_amt_bins=10):
    df['log_amt'] = np.log1p(df['amt'].astype(float))
    edges = np.quantile(df['log_amt'], np.linspace(0,1,n_amt_bins+1))
    # ensure monotonic edges
    edges = np.unique(edges)
    df['amt_bin'] = np.searchsorted(edges, df['log_amt'], side='right') - 1
    return edges, n_amt_bins

def compute_time_buckets(df, bucket_hours=3):
    df['ts'] = pd.to_datetime(df['trans_date_trans_time'])
    df['hour'] = df['ts'].dt.hour
    df['time_bucket'] = (df['hour'] // bucket_hours).astype(int)
    return bucket_hours

def collapse_categories(df, top_k=14):
    top = df['category'].value_counts().index[:top_k]
    df['category_small'] = df['category'].where(df['category'].isin(top), 'other')
    df['category_small_id'] = pd.factorize(df['category_small'])[0]
    return top

def main(args):
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df = pd.read_csv(args.input)
    if 'is_fraud' in df.columns:
        df = df[df['is_fraud'] == 1]
    
    # merchant clusters
    merchant_map, merchant_meta = build_merchant_clusters(df, K=args.k_merchant)
    df['merchant_cluster'] = df['merchant'].map(merchant_map).fillna(args.k_merchant).astype(int)

    # merchant id
    merchant_id_map = {m: i for i, m in enumerate(df['merchant'].unique())}
    df['merchant_id'] = df['merchant'].map(merchant_id_map)

    # user id
    user_id_map = {u: i for i, u in enumerate(df['cc_num'].unique())}
    df['user_id'] = df['cc_num'].map(user_id_map)

    # collapse categories
    top_cats = collapse_categories(df, top_k=args.top_k_cat)

    # amount bins
    amt_edges, _ = compute_amt_bins(df, n_amt_bins=args.amt_bins)
    df['amt_bin'] = df['amt_bin'].clip(0, args.amt_bins-1)

    # time buckets
    compute_time_buckets(df, bucket_hours=args.time_bucket_hours)

    # Y features
    # Gender
    gender_map = {g: i for i, g in enumerate(df['gender'].unique())}
    df['gender_id'] = df['gender'].map(gender_map)

    # City
    city_map = {c: i for i, c in enumerate(df['city'].unique())}
    df['city_id'] = df['city'].map(city_map)

    # Job
    job_map = {j: i for i, j in enumerate(df['job'].unique())}
    df['job_id'] = df['job'].map(job_map)

    # Age
    df['dob'] = pd.to_datetime(df['dob'])
    df['age'] = (pd.to_datetime('2025-11-17') - df['dob']).dt.days // 365
    age_bins = [0, 18, 25, 35, 45, 60, 150]
    df['age_bucket'] = pd.cut(df['age'], bins=age_bins, labels=False, right=False).fillna(len(age_bins)-2).astype(int)
    age_bucket_map = {i: i for i in range(len(age_bins) - 1)}

    # Save encoders
    encoders = {
        'merchant_cluster_map': merchant_map,
        'merchant_meta': merchant_meta,
        'merchant_id_map': merchant_id_map,
        'user_id_map': user_id_map,
        'amt_edges': amt_edges.tolist(),
        'top_cats': list(top_cats),
        'n_amt_bins': args.amt_bins,
        'time_bucket_hours': args.time_bucket_hours,
        'gender': gender_map,
        'city': city_map,
        'job': job_map,
        'age_bucket': age_bucket_map
    }
    enc_path = args.encoders or 'data/encoders/encoders.pkl'
    os.makedirs(os.path.dirname(enc_path), exist_ok=True)
    with open(enc_path, 'wb') as f:
        pickle.dump(encoders, f)

    # Save processed csv
    df.to_csv(args.out, index=False)
    print("Saved processed data to:", args.out)
    print("Saved encoders to:", enc_path)
    # quick stats
    print("Unique node combos (approx):")
    node_fields = ['merchant_id', 'amt_bin','merchant_cluster','category_small_id','city_id','time_bucket']
    combos = df[node_fields].dropna().astype(int).apply(tuple, axis=1).value_counts()
    print("unique_node_combos:", len(combos))
    print("top 20 node combos:\n", combos.head(20))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='data/raw/transactions.csv')
    p.add_argument('--out', default='data/preprocessed/processed.csv')
    p.add_argument('--encoders', default='data/encoders/encoders.pkl')
    p.add_argument('--k_merchant', type=int, default=60)
    p.add_argument('--amt_bins', type=int, default=10)
    p.add_argument('--time_bucket_hours', type=int, default=8)
    p.add_argument('--top_k_cat', type=int, default=14)
    args = p.parse_args()
    main(args)
