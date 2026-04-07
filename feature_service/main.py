#!/usr/bin/env python3
import os, json
import psycopg2
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="Mealie Feature Service")

PG_HOST = os.environ.get('POSTGRES_HOST', 'postgres')
PG_USER = os.environ['POSTGRES_USER']
PG_PASS = os.environ['POSTGRES_PASSWORD']
PG_DB   = os.environ['POSTGRES_DB']
DIM     = 50

WEIGHT_MAP = {5: 1.0, 4: 0.7, 3: 0.0, 2: -0.5, 1: -1.0}

def pg():
    return psycopg2.connect(host=PG_HOST, user=PG_USER,
                            password=PG_PASS, dbname=PG_DB)

def get_user_vector(user_id: str):
    conn = pg(); cur = conn.cursor()
    cur.execute("SELECT vector FROM user_vectors WHERE user_id = %s", (user_id,))
    row = cur.fetchone()
    cur.close(); conn.close()
    if row:
        return row[0] if isinstance(row[0], list) else json.loads(row[0])
    return [0.0] * DIM

class Recipe(BaseModel):
    recipe_id: str
    name: str
    tags: List[str]
    minutes: Optional[int] = 30
    calories: Optional[float] = 300.0

class FeaturesRequest(BaseModel):
    user_id: str
    library_recipes: List[Recipe]
    top_n: Optional[int] = 10

class EventRequest(BaseModel):
    user_id: str
    recipe_id: str
    event_type: str
    rating: Optional[int] = None
    weight: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/features")
def get_features(req: FeaturesRequest):
    user_vector = get_user_vector(req.user_id)
    library = [
        {"recipe_id": r.recipe_id, "name": r.name,
         "tags": r.tags, "minutes": r.minutes,
         "calories": r.calories}
        for r in req.library_recipes
    ]
    return {
        "user_id":         req.user_id,
        "user_vector":     user_vector,
        "library_recipes": library,
        "top_n":           req.top_n
    }

@app.post("/log_event")
def log_event(req: EventRequest):
    conn = pg(); cur = conn.cursor()
    cur.execute("""INSERT INTO mealie_events
                   (user_id, recipe_id, event_type, rating, weight)
                   VALUES (%s, %s, %s, %s, %s)""",
                (req.user_id, req.recipe_id, req.event_type,
                 req.rating, req.weight))
    conn.commit(); cur.close(); conn.close()
    return {"status": "logged"}
