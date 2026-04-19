#!/usr/bin/env python3
"""
nightly_eval.py
Data quality checks at 3 points:
1. Ingestion quality — checks raw data in object storage
2. Training set quality — checks versioned train/val datasets
3. Live inference drift — checks production events in PostgreSQL
Results logged to MLflow.
"""
import os, json, sys
import boto3
import pandas as pd
import numpy as np
import psycopg2
import mlflow
from io import BytesIO
from datetime import datetime, timedelta

# Config from env vars (set by Mahima via K8s secrets)
MINIO_ENDPOINT  = os.environ.get('MINIO_ENDPOINT', 'http://129.114.26.176:30900')
MINIO_ACCESS    = os.environ.get('MINIO_ACCESS_KEY', 'minioadmin')
MINIO_SECRET    = os.environ.get('MINIO_SECRET_KEY', 'minioadmin123')
BUCKET          = os.environ.get('BUCKET_NAME', 'training-data')

MLFLOW_URL      = os.environ.get('MLFLOW_TRACKING_URI', 'http://129.114.26.176:30500')

PG_HOST         = os.environ.get('POSTGRES_HOST', 'postgres.platform.svc.cluster.local')
PG_USER         = os.environ.get('POSTGRES_USER', 'mealie')
PG_PASS         = os.environ.get('POSTGRES_PASSWORD', 'mealie')
PG_DB           = os.environ.get('POSTGRES_DB', 'mealie')

#Expected values (from what we built) 
EXPECTED_RECIPE_COUNT       = 231636
EXPECTED_INTERACTION_COUNT  = 1091512
EXPECTED_TAGS               = 552
WEIGHT_VALUES               = {1.0, 0.7, -0.5, -1.0, 0.4, -0.3}
DISMISS_RATE_THRESHOLD      = 0.5   # alert if >50% of events are dismissals
NEW_TAG_THRESHOLD           = 0.1   # alert if >10% of tags are unseen

failed_checks = []

def s3():
    return boto3.client('s3',
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS,
        aws_secret_access_key=MINIO_SECRET)

def pg():
    return psycopg2.connect(
        host=PG_HOST, user=PG_USER,
        password=PG_PASS, dbname=PG_DB)

def load_parquet(key):
    obj = s3().get_object(Bucket=BUCKET, Key=key)
    return pd.read_parquet(BytesIO(obj['Body'].read()))

def check(name, passed, value, expected, warn=False):
    status = "PASS" if passed else ("WARN" if warn else "FAIL")
    print(f"  [{status}] {name}: {value} (expected {expected})")
    if not passed and not warn:
        failed_checks.append(name)
    return passed

#  Check 1: Ingestion Quality 
def check_ingestion():
    print("\n=== CHECK 1: Ingestion Quality ===")
    metrics = {}

    # recipes
    try:
        recipes = load_parquet('processed/recipes_clean.parquet')
        count = len(recipes)
        metrics['recipe_count'] = count
        check('recipe_count', count >= EXPECTED_RECIPE_COUNT * 0.95,
              count, f">= {int(EXPECTED_RECIPE_COUNT * 0.95)}")

        null_pct = recipes[['id','name','tags','minutes']].isnull().mean().max()
        metrics['recipe_null_pct'] = float(null_pct)
        check('recipe_nulls', null_pct < 0.01, f"{null_pct:.2%}", "< 1%")

        has_tags = recipes['tags'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False)
        tag_coverage = has_tags.mean()
        metrics['recipe_tag_coverage'] = float(tag_coverage)
        check('recipe_tag_coverage', tag_coverage > 0.9, f"{tag_coverage:.2%}", "> 90%")

    except Exception as e:
        print(f"  [FAIL] Could not load recipes: {e}")
        failed_checks.append('recipe_load')
        metrics['recipe_error'] = str(e)

    # interactions
    try:
        interactions = load_parquet('processed/interactions_clean.parquet')
        count = len(interactions)
        metrics['interaction_count'] = count
        check('interaction_count', count >= EXPECTED_INTERACTION_COUNT * 0.95,
              count, f">= {int(EXPECTED_INTERACTION_COUNT * 0.95)}")

        null_pct = interactions[['user_id','recipe_id','rating','weight']].isnull().mean().max()
        metrics['interaction_null_pct'] = float(null_pct)
        check('interaction_nulls', null_pct < 0.01, f"{null_pct:.2%}", "< 1%")

        rating_dist = interactions['rating'].value_counts(normalize=True)
        metrics['rating_distribution'] = rating_dist.to_dict()
        no_single_dominant = (rating_dist.max() < 0.6)
        check('rating_distribution', no_single_dominant,
              f"max={rating_dist.max():.2%}", "no rating > 60%")

        invalid_weights = ~interactions['weight'].isin(WEIGHT_VALUES)
        metrics['invalid_weight_count'] = int(invalid_weights.sum())
        check('weight_values', invalid_weights.sum() == 0,
              invalid_weights.sum(), "0 invalid weights")

    except Exception as e:
        print(f"  [FAIL] Could not load interactions: {e}")
        failed_checks.append('interaction_load')
        metrics['interaction_error'] = str(e)

    return metrics

#  Check 2: Training Set Quality 
def check_training_set():
    print("\n=== CHECK 2: Training Set Quality ===")
    metrics = {}

    try:
        # find latest dataset version
        response = s3().list_objects_v2(Bucket=BUCKET, Prefix='datasets/', Delimiter='/')
        versions = [p['Prefix'] for p in response.get('CommonPrefixes', [])]
        if not versions:
            print("  [FAIL] No versioned datasets found")
            failed_checks.append('no_datasets')
            return metrics

        latest = sorted(versions)[-1]
        print(f"  Using latest version: {latest}")
        metrics['dataset_version'] = latest

        train = load_parquet(f'{latest}train.parquet')
        val   = load_parquet(f'{latest}val.parquet')

        metrics['train_rows'] = len(train)
        metrics['val_rows']   = len(val)

        # row counts reasonable
        check('train_not_empty', len(train) > 1000, len(train), "> 1000")
        check('val_not_empty',   len(val)   > 100,  len(val),   "> 100")

        # train/val ratio roughly 80/20
        total = len(train) + len(val)
        train_ratio = len(train) / total
        metrics['train_ratio'] = float(train_ratio)
        check('train_val_ratio', 0.75 <= train_ratio <= 0.85,
              f"{train_ratio:.2%}", "75-85%")

        # no overlap between train and val users
        if 'user_id' in train.columns and 'user_id' in val.columns:
            train_users = set(train['user_id'].astype(str))
            val_users   = set(val['user_id'].astype(str))
            overlap = len(train_users & val_users)
            metrics['user_overlap'] = overlap
            check('no_user_overlap', overlap == 0, overlap, "0 overlapping users", warn=True)

        # weight balance  not too many negatives
        if 'weight' in train.columns:
            pos = (train['weight'] > 0).mean()
            neg = (train['weight'] < 0).mean()
            metrics['positive_weight_ratio'] = float(pos)
            metrics['negative_weight_ratio'] = float(neg)
            check('weight_balance', pos > 0.3, f"pos={pos:.2%}", "> 30% positive")

        # chronological  check timestamps if available
        if 'timestamp' in train.columns and 'timestamp' in val.columns:
            train_max = pd.to_datetime(train['timestamp']).max()
            val_min   = pd.to_datetime(val['timestamp']).min()
            no_leakage = train_max <= val_min
            metrics['train_max_time'] = str(train_max)
            metrics['val_min_time']   = str(val_min)
            check('no_temporal_leakage', no_leakage,
                  f"train_max={train_max}", f"<= val_min={val_min}")

    except Exception as e:
        print(f"  [FAIL] Training set check error: {e}")
        failed_checks.append('training_set_load')
        metrics['training_set_error'] = str(e)

    return metrics

#  Check 3: Live Inference Drift 
def check_inference_drift():
    print("\n=== CHECK 3: Live Inference Drift ===")
    metrics = {}

    try:
        conn = pg()

        # get events from last 24 hours
        df = pd.read_sql("""
            SELECT user_id, recipe_id, event_type, weight, timestamp
            FROM mealie_events
            WHERE timestamp >= NOW() - INTERVAL '24 hours'
            ORDER BY timestamp ASC
        """, conn)

        metrics['events_last_24h'] = len(df)
        print(f"  Events in last 24h: {len(df)}")

        if len(df) == 0:
            print("  [WARN] No events in last 24 hours")
            return metrics

        # dismissal rate drift
        dismiss_rate = (df['event_type'] == 'dismiss').mean()
        metrics['dismiss_rate'] = float(dismiss_rate)
        check('dismiss_rate', dismiss_rate < DISMISS_RATE_THRESHOLD,
              f"{dismiss_rate:.2%}", f"< {DISMISS_RATE_THRESHOLD:.0%}", warn=True)

        # weight distribution drift
        avg_weight = df['weight'].mean()
        metrics['avg_weight_24h'] = float(avg_weight)
        check('avg_weight', avg_weight > -0.2,
              f"{avg_weight:.3f}", "> -0.2 (not too negative)", warn=True)

        # event type distribution
        event_dist = df['event_type'].value_counts(normalize=True).to_dict()
        metrics['event_distribution'] = event_dist
        print(f"  Event distribution: {event_dist}")

        # unique users active
        metrics['unique_users_24h'] = int(df['user_id'].nunique())

        # check for unseen tags in recent recipe interactions
        try:
            tag_vectors = pd.read_sql(
                "SELECT tag FROM tag_vectors", conn)
            known_tags = set(tag_vectors['tag'].tolist())

            recent_recipes = load_parquet('processed/recipes_clean.parquet')
            recent_recipes = recent_recipes[
                recent_recipes['id'].astype(str).isin(df['recipe_id'].astype(str))]

            if len(recent_recipes) > 0:
                all_recent_tags = {t for tags in recent_recipes['tags']
                                   for t in (tags if isinstance(tags, list) else [])}
                unseen = all_recent_tags - known_tags
                unseen_rate = len(unseen) / max(len(all_recent_tags), 1)
                metrics['unseen_tag_count'] = len(unseen)
                metrics['unseen_tag_rate']  = float(unseen_rate)
                check('unseen_tags', unseen_rate < NEW_TAG_THRESHOLD,
                      f"{unseen_rate:.2%}", f"< {NEW_TAG_THRESHOLD:.0%}", warn=True)
                if unseen:
                    print(f"  Unseen tags: {list(unseen)[:5]}")

        except Exception as e:
            print(f"  [WARN] Could not check tag drift: {e}")

        conn.close()

    except Exception as e:
        print(f"  [FAIL] Drift check error: {e}")
        failed_checks.append('drift_check')
        metrics['drift_error'] = str(e)

    return metrics

#  Main 
def main():
    print(f"=== Nightly Eval | {datetime.utcnow().isoformat()} ===")

    mlflow.set_tracking_uri(MLFLOW_URL)
    mlflow.set_experiment("nightly-data-eval")

    with mlflow.start_run(run_name=f"eval_{datetime.utcnow().strftime('%Y%m%d')}"):

        m1 = check_ingestion()
        m2 = check_training_set()
        m3 = check_inference_drift()

        # log all metrics to MLflow
        all_metrics = {**m1, **m2, **m3}
        for k, v in all_metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, v)
            else:
                mlflow.log_param(k, str(v)[:250])

        mlflow.log_param('failed_checks', str(failed_checks))
        mlflow.log_param('total_failed', len(failed_checks))
        mlflow.log_metric('checks_passed',
                          1 if len(failed_checks) == 0 else 0)

        # save report
        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'failed_checks': failed_checks,
            'metrics': {k: v for k, v in all_metrics.items()
                       if isinstance(v, (int, float, str))}
        }
        with open('/tmp/eval_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact('/tmp/eval_report.json')

    print(f"\n{'='*40}")
    if failed_checks:
        print(f"❌ {len(failed_checks)} checks FAILED: {failed_checks}")
        sys.exit(1)
    else:
        print("✅ All checks passed!")
        sys.exit(0)

if __name__ == '__main__':
    main()
