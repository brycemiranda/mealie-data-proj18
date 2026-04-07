#!/usr/bin/env python3
import os, json
import pandas as pd
import boto3
import psycopg2
from io import BytesIO
from datetime import datetime

BUCKET      = os.environ['BUCKET_NAME']
S3_ENDPOINT = os.environ['S3_ENDPOINT']
AWS_KEY     = os.environ['AWS_ACCESS_KEY_ID']
AWS_SECRET  = os.environ['AWS_SECRET_ACCESS_KEY']
PG_HOST     = os.environ.get('POSTGRES_HOST', 'postgres')
PG_USER     = os.environ['POSTGRES_USER']
PG_PASS     = os.environ['POSTGRES_PASSWORD']
PG_DB       = os.environ['POSTGRES_DB']

def s3():
    return boto3.client('s3', endpoint_url=S3_ENDPOINT,
                        aws_access_key_id=AWS_KEY,
                        aws_secret_access_key=AWS_SECRET)

def pg():
    return psycopg2.connect(host=PG_HOST, user=PG_USER,
                            password=PG_PASS, dbname=PG_DB)

def fetch_production_events():
    print("Fetching production events from PostgreSQL...")
    conn = pg()
    df = pd.read_sql("""
        SELECT user_id, recipe_id, event_type, rating, weight, timestamp
        FROM mealie_events
        WHERE weight != 0.0
        ORDER BY timestamp ASC
    """, conn)
    conn.close()
    print(f"  ✓ {len(df):,} events fetched")
    return df

def fetch_foodcom_base():
    print("Loading Food.com base interactions from object storage...")
    obj = s3().get_object(Bucket=BUCKET,
                          Key='processed/interactions_clean.parquet')
    df = pd.read_parquet(BytesIO(obj['Body'].read()))
    df['source'] = 'foodcom'
    df['user_id'] = df['user_id'].astype(str)
    df['recipe_id'] = df['recipe_id'].astype(str)
    print(f"  ✓ {len(df):,} Food.com interactions loaded")
    return df

def candidate_selection(df):
    print("Applying candidate selection...")
    df = df[df['weight'] != 0.0]
    user_counts = df.groupby('user_id').size()
    valid = user_counts[user_counts >= 3].index
    df = df[df['user_id'].isin(valid)]
    print(f"  ✓ {len(df):,} events kept")
    print(f"  ✓ {df['user_id'].nunique():,} unique users")
    return df

def chronological_split(df, ratio=0.8):
    print(f"Chronological split ({int(ratio*100)}/{int((1-ratio)*100)})...")
    df = df.sort_values('timestamp')
    cut = int(len(df) * ratio)
    train, val = df.iloc[:cut], df.iloc[cut:]
    print(f"  Train: {len(train):,} rows")
    print(f"  Val:   {len(val):,} rows")
    return train, val

def upload_versioned(client, train, val, version):
    def up(df, key):
        buf = BytesIO()
        df.to_parquet(buf, index=False); buf.seek(0)
        client.put_object(Bucket=BUCKET, Key=key, Body=buf.getvalue())
        print(f"  ✓ Uploaded {key}")

    up(train, f'datasets/{version}/train.parquet')
    up(val,   f'datasets/{version}/val.parquet')

    meta = {
        'version': version,
        'train_rows': len(train),
        'val_rows': len(val),
        'unique_users_train': int(train['user_id'].nunique()),
        'created_at': datetime.utcnow().isoformat(),
        'split_method': 'chronological',
    }
    client.put_object(Bucket=BUCKET,
                      Key=f'datasets/{version}/meta.json',
                      Body=json.dumps(meta, indent=2))
    print(f"  ✓ Metadata written")
    print(f"  {json.dumps(meta, indent=2)}")

def main():
    version = f"v2_{datetime.today().strftime('%Y-%m-%d')}"
    print(f"=== Batch Pipeline | version: {version} ===\n")

    client = s3()

    # Get production events from PostgreSQL
    prod = fetch_production_events()
    prod['timestamp'] = pd.to_datetime(prod['timestamp'])
    prod['source'] = 'production'

    # Get Food.com base
    foodcom = fetch_foodcom_base()
    foodcom['timestamp'] = pd.to_datetime('2020-01-01')

    # Combine
    combined = pd.concat([foodcom, prod], ignore_index=True)

    # Candidate selection
    selected = candidate_selection(combined)

    # Chronological split
    train, val = chronological_split(selected)

    # Upload versioned datasets
    upload_versioned(client, train, val, version)

    print(f"\n✅ Batch pipeline complete! Version: {version}")

if __name__ == '__main__':
    main()
