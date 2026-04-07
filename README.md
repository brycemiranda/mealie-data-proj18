# Mealie Data Pipeline — Team Bias & Variance (proj18)

Data member implementation for the personalized recipe recommender integrated into Mealie.
Role: Bryce Miranda (bm3986) — Data

## Project Overview
A "Recommended for You" panel that ranks recipes in a user's Mealie library by predicted
preference using ALS (Alternating Least Squares) collaborative filtering.

## Infrastructure
- VM: node-mealie-data-proj18 on KVM@TACC (m1.medium)
- Object Storage: mealie-data-proj18 on CHI@TACC
- Database: PostgreSQL 15 (Docker container with persistent volume)

## Components

### ingestion/
Reproducible pipeline that downloads the Food.com dataset from Kaggle, cleans recipes
and interactions, generates synthetic data, performs a chronological 80/20 split, and
uploads versioned datasets to Chameleon object storage.

### feature_service/
FastAPI service (port 8000) that assembles real-time ALS input for the serving layer.
Fetches user taste vectors from PostgreSQL and returns input JSON matching als_input.json.

### generator/
Simulates Mealie users making requests and logging interactions (ratings, saves, dismissals)
to PostgreSQL. Used to populate production feedback for retraining.

### batch/
Monthly batch pipeline that fetches production events from PostgreSQL, combines with
Food.com base interactions, applies candidate selection, and uploads a new versioned
dataset to object storage using a chronological split to avoid leakage.

## How to Run

Start persistent services:
    docker compose up -d postgres feature_service

Run ingestion pipeline (once):
    docker compose --profile run_ingestion up ingestion

Run data generator:
    docker compose --profile run_generator up generator

Run batch pipeline (monthly):
    docker compose --profile run_batch up batch

## External Dataset
Food.com Recipes and User Interactions
https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions
231,637 recipes and 1,048,576 interactions scraped from Food.com (2000-2018).
Collected by Majumder, Li, Ni, McAuley (EMNLP 2019). License: CC BY-SA 3.0.

## Note
.env file is gitignored — contains database credentials and S3 keys.
All code assisted by Claude Sonnet 4.6.
