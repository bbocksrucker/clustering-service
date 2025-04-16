from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import os
import uuid
import pandas as pd
from k_means_constrained import KMeansConstrained
import requests
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

class ClusterRequest(BaseModel):
    import_id: str
    amount: int
    deviation: float

@app.post("/api/cluster")
def cluster_stops(req: ClusterRequest):
    # 1. Hole alle route_stops + locations mit Koordinaten
    url = f"{SUPABASE_URL}/rest/v1/route_stop?import_id=eq.{req.import_id}&select=id,location_id,location:locations(id,latitude,longitude)"
    res = requests.get(url, headers={
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}"
    })
    
    if not res.ok:
        raise HTTPException(status_code=500, detail="❌ Fehler beim Laden der Daten aus Supabase")

    data = res.json()

    df = pd.DataFrame([
        {
            "id": entry["id"],
            "lat": entry["location"]["latitude"],
            "lon": entry["location"]["longitude"]
        }
        for entry in data
        if entry["location"] and entry["location"]["latitude"] and entry["location"]["longitude"]
    ])

    if df.empty:
        raise HTTPException(status_code=400, detail="⚠️ Keine gültigen Koordinaten gefunden.")

    # 2. Clustering durchführen
    X = df[["lon", "lat"]]
    clf = KMeansConstrained(
        n_clusters=req.amount,
        size_min=int((len(X) / req.amount) * (1 - req.deviation)),
        size_max=int((len(X) / req.amount) * (1 + req.deviation)),
        random_state=42
    )
    clf.fit(X)

    df["cluster"] = clf.labels_

    # 3. Neue cluster_ids generieren
    cluster_ids = {label: str(uuid.uuid4()) for label in df["cluster"].unique()}
    df["cluster_id"] = df["cluster"].map(cluster_ids)

    # 4. Updates vorbereiten
    updates = []
    for _, row in df.iterrows():
        updates.append({
            "id": row["id"],
            "cluster_id": row["cluster_id"]
        })

    # 5. Updates ausführen
    for update in updates:
        update_url = f"{SUPABASE_URL}/rest/v1/route_stop?id=eq.{update['id']}"
        update_res = requests.patch(update_url, headers={
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "return=representation"
        }, json={ "cluster_id": update["cluster_id"] })

        if not update_res.ok:
            print(f"❌ Fehler beim Update von {update['id']}: {update_res.text}")

    return {
        "message": "✅ Clustering abgeschlossen",
        "import_id": req.import_id,
        "clusters": list(cluster_ids.values())
    }
