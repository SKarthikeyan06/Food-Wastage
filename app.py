"""
FoodBridge AI v2 - Complete Flask Backend
Role-based auth + Formula+ML hybrid prediction + Supabase
"""

from flask import Flask, request, jsonify, session, send_from_directory
from flask_cors import CORS
from datetime import datetime
import os, hashlib, csv, io, requests, uuid, math

# ML imports
try:
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
except Exception:
    pd = None
    np = None
    RandomForestRegressor = None
    train_test_split = None

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "foodbridge-secret-key-v2")
CORS(app, supports_credentials=True)

# ── Supabase REST API ─────────────────────────────────────────────────
# Declare Supabase credentials here (not via environment or request)
SUPABASE_URL = 'https://lrcemdihdncmtfxqqgew.supabase.co'
SUPABASE_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImxyY2VtZGloZG5jbXRmeHFxZ2V3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzA4NjE1MTQsImV4cCI6MjA4NjQzNzUxNH0.NmiOJZCvx0gvM7X5hJJoiaIrjK4OxmJDolZY93uSXkA'

sb = None
DB = False

# ML model globals
ML_MODEL = None
ML_TRAINED = False
ML_TRAIN_ROWS = 0


def build_ml_dataset(limit=5000):
    """Build ML dataset from Supabase tables or local CSV fallback.
    Returns X (2-col) and y arrays or (None,0) on failure."""
    rows = []
    # Try DB sources first
    if DB:
        try:
            # 1. Live confirmed events
            ev = db_request("GET", "food_events", select="invited_guests,quantity_prepared,actual_wastage", limit=limit) or []
            for r in ev:
                try:
                    q = float(r.get("quantity_prepared") or 0)
                    g = int(r.get("invited_guests") or 0)
                    w = float(r.get("actual_wastage") or 0)
                    if q > 0 and w >= 0:
                        rows.append((g, q, w))
                except Exception:
                    continue

            # 2. TN dataset
            tn_limit = max(0, limit - len(rows))
            if tn_limit > 0:
                tn = db_request("GET", "tn_food_surplus_dataset", select="invited_guests,quantity_prepared_kg,wastage_kg", limit=tn_limit) or []
                for r in tn:
                    try:
                        g = int(r.get("invited_guests") or 0)
                        q = float(r.get("quantity_prepared_kg") or 0)
                        w = float(r.get("wastage_kg") or 0)
                        if q > 0 and w >= 0:
                            rows.append((g, q, w))
                    except Exception:
                        continue

            # 3. Original dataset (The 1782 rows)
            orig_limit = max(0, limit - len(rows))
            if orig_limit > 0:
                orig = db_request("GET", "original_food_dataset", select="number_of_guests,quantity_of_food,wastage_food_amount", limit=orig_limit) or []
                for r in orig:
                    try:
                        g = int(r.get("number_of_guests") or 0)
                        q = float(r.get("quantity_of_food") or 0)
                        w = float(r.get("wastage_food_amount") or 0)
                        if q > 0 and w >= 0:
                            rows.append((g, q, w))
                    except Exception:
                        continue

        except Exception as e:
            print(f"ML build db error: {e}")

    # Fallback to local CSV if pandas available
    if (not rows) and pd is not None:
        try:
            path = "D:/DataSet/food_wastage_data.csv"
            if os.path.exists(path):
                df = pd.read_csv(path)
                for _, r in df.iterrows():
                    try:
                        # Map correct columns for local CSV
                        g = int(r.get("Number of Guests") or 0)
                        q = float(r.get("Quantity of Food") or 0)
                        w = float(r.get("Wastage Food Amount") or 0)
                        if q > 0:
                            rows.append((g, q, w))
                    except Exception:
                        continue
        except Exception as e:
            print(f"ML build csv error: {e}")

    if not rows:
        return None, 0

    # prepare arrays
    try:
        arr = np.array(rows)
        X = arr[:, :2].astype(float)
        y = arr[:, 2].astype(float)
        return (X, y), len(rows)
    except Exception as e:
        print(f"ML build array error: {e}")
        return None, 0


def train_ml_model(force=False, limit=10000):
    """Train or retrain the ML model. Returns rows used."""
    global ML_MODEL, ML_TRAINED, ML_TRAIN_ROWS
    if RandomForestRegressor is None or pd is None:
        print("ML libs not available")
        return 0

    ds, count = build_ml_dataset(limit=limit)
    if not ds or count < 10:
        print(f"Not enough ML rows: {count}")
        return 0

    X, y = ds
    try:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        ML_MODEL = model
        ML_TRAINED = True
        ML_TRAIN_ROWS = count
        print(f"ML model trained on {count} rows")
        return count
    except Exception as e:
        print(f"ML train error: {e}")
        return 0


def ml_predict(guests, quantity_kg):
    """Return predicted wastage (kg) from ML model, or None."""
    if not ML_TRAINED or ML_MODEL is None:
        return None
    try:
        Xp = np.array([[float(guests), float(quantity_kg)]])
        pred = ML_MODEL.predict(Xp)
        return float(pred[0])
    except Exception as e:
        print(f"ML predict error: {e}")
        return None


# ── Demo in-memory notifications when DB is unavailable
DEMO_NOTIFICATIONS = [
    {"id":"d1","title":"Welcome","message":"Welcome to FoodBridge AI demo","to_role":"all","is_read":False,"created_at":datetime.utcnow().isoformat()},
]

# ── Gemini AI fallback prediction ─────────────────────────────────
GEMINI_API_KEY = "AIzaSyDfa_HuIcwfQ9PXdbKkvT9rOcBIHKZuLSk"
_gemini_cache  = {}  # in-memory cache to avoid hitting rate limits

def gemini_predict(guests, event_type, food_type, prep_method, season, pricing):
    """Call Gemini API as Global AI fallback when formula confidence is low."""
    print("\t\tCOMING INSIDE GEMINI")
    if not GEMINI_API_KEY:
        return None, "no_key"

    # Return cached result to avoid repeated API calls (rate limit safety)
    cache_key = f"{event_type}|{food_type}|{round(guests/50)*50}|{prep_method}|{season}|{pricing}"
    if cache_key in _gemini_cache:
        print("[Gemini] Cache hit — reusing previous result")
        return _gemini_cache[cache_key], "gemini_cached"

    try:
        prompt = (
            f"You are a food waste prediction expert for Indian events. "
            f"Predict the food wastage for this event:\n"
            f"- Event type: {event_type}\n"
            f"- Food type: {food_type}\n"
            f"- Number of guests: {guests}\n"
            f"- Preparation method: {prep_method}\n"
            f"- Season: {season}\n"
            f"- Pricing level: {pricing}\n\n"
            f"Reply ONLY with valid JSON in this exact format (no extra text):\n"
            f'{{"wastage_kg": 12.5, "optimal_kg": 85.0, "confidence": 75, "reasoning": "Brief reason here"}}'
        )
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
        body = {"contents": [{"parts": [{"text": prompt}]}]}
        r = requests.post(url, json=body, timeout=10)
        print(f"[Gemini] Response status: {r.status_code}")
        if r.status_code == 429:
            print("[Gemini] Rate limit (429) — falling back to formula")
            return None, "rate_limited"
        if r.status_code != 200:
            print(f"[Gemini] Error body: {r.text[:300]}")
            return None, "error"
        import json, re
        text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
        print(f"[Gemini] Raw response: {text}")
        # extract JSON from the response (handles markdown code blocks too)
        m = re.search(r'\{[^{}]+\}', text, re.DOTALL)
        if m:
            data = json.loads(m.group())
            _gemini_cache[cache_key] = data   # cache for future identical requests
            return data, "gemini"
        print("[Gemini] Could not extract JSON from response")
    except Exception as e:
        print(f"[Gemini] Error: {e}")
    return None, "error"

# ── Haversine distance (km) between two lat/lng points ───────────────────
def haversine_km(lat1, lng1, lat2, lng2):
    R = 6371  # Earth radius km
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlng/2)**2
    return round(R * 2 * math.asin(math.sqrt(a)), 1)

def find_nearest_ngos(city, lat=None, lng=None, surplus_kg=0, max_results=3):
    """Fetch NGOs from DB and sort by Haversine distance or city match."""
    if not DB:
        # Demo fallback
        demo = [
            {"name": "Annadhanam Trust", "city": city, "distance_km": 2.3, "phone": "9876500001"},
            {"name": "Hope Shelter",     "city": city, "distance_km": 4.1, "phone": "9876500002"},
            {"name": "City Food Bank",   "city": city, "distance_km": 6.2, "phone": "9876500003"},
        ]
        return [n for n in demo if n["distance_km"] <= max(10, surplus_kg * 0.3)][:max_results]

    try:
        ngos_raw = db_request("GET", "ngos", select="id,name,city,phone,email,capacity_kg,latitude,longitude") or []
        results = []
        for ngo in ngos_raw:
            ngo_lat = ngo.get("latitude")
            ngo_lng = ngo.get("longitude")
            ngo_cap = float(ngo.get("capacity_kg") or 50)

            # Skip if NGO capacity can't handle the surplus
            if ngo_cap < surplus_kg * 0.3:
                continue

            if lat is not None and lng is not None and ngo_lat and ngo_lng:
                dist = haversine_km(float(lat), float(lng), float(ngo_lat), float(ngo_lng))
            elif ngo.get("city", "").lower() == (city or "").lower():
                dist = round(1.0 + len(results) * 1.5, 1)  # city-matched fallback distance
            else:
                continue  # no location data and different city, skip

            results.append({**ngo, "distance_km": dist})

        results.sort(key=lambda x: x["distance_km"])
        return results[:max_results]
    except Exception as e:
        print(f"[NGO Finder] Error: {e}")
        return []

def get_supabase_credentials():
    """Get declared Supabase credentials"""
    return SUPABASE_URL, SUPABASE_KEY

def get_supabase_headers():
    """Get authorization headers for Supabase REST API"""
    url, key = get_supabase_credentials()
    if not url or not key:
        return None, None, None
    
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }
    return url, key, headers

def supabase_table(table_name):
    """Build Supabase REST API table URL"""
    url, _ = get_supabase_credentials()
    if not url:
        return None
    return f"{url}/rest/v1/{table_name}"

def db_request(method, table_name, data=None, filters=None, select="*", limit=None):
    """
    Helper function to make Supabase REST API requests
    method: 'GET', 'POST', 'PATCH', 'DELETE'
    """
    global DB
    
    url, key, headers = get_supabase_headers()
    if not url or not headers:
        DB = False
        return None
    
    # Build URL
    api_url = supabase_table(table_name)
    if not api_url:
        DB = False
        return None
    
    # Add query parameters
    params = []
    if select != "*":
        params.append(f"select={select}")
    if limit:
        params.append(f"limit={limit}")
    if filters:
        for key_filter, value_filter in filters.items():
            if value_filter is None:
                params.append(f"{key_filter}=is.null")
            elif isinstance(value_filter, bool):
                params.append(f"{key_filter}=eq.{str(value_filter).lower()}")
            else:
                params.append(f"{key_filter}=eq.{value_filter}")
    
    if params:
        api_url += "?" + "&".join(params)
    
    # Supabase PostgREST: To get result data back on POST/PATCH, we need a special header
    if method in ["POST", "PATCH"]:
        headers["Prefer"] = "return=representation"

    try:
        if method == "GET":
            response = requests.get(api_url, headers=headers)
        elif method == "POST":
            response = requests.post(api_url, headers=headers, json=data)
        elif method == "PATCH":
            response = requests.patch(api_url, headers=headers, json=data)
        elif method == "DELETE":
            response = requests.delete(api_url, headers=headers)
        else:
            return None
        
        if response.status_code in [200, 201]:
            DB = True
            return response.json() if response.text else []
        else:
            print(f"[DB] Error {response.status_code}: {response.text}")
            # do not disable DB entirely on a single failed query (e.g. 401 due to RLS)
            # just return None so caller can fallback to demo or handle error
            return None
    except Exception as e:
        print(f"[DB] Request error: {e}")
        DB = False
        return None

# Test initial connection with declared credentials
try:
    if SUPABASE_URL and SUPABASE_KEY:
        # Test connection with a simple endpoint
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json"
        }
        
        # Try different test endpoints
        test_urls = [
            f"{SUPABASE_URL}/rest/v1/",  # Schema endpoint (no RLS)
            f"{SUPABASE_URL}/rest/v1/food_events?limit=1",  # Food events table
        ]
        
        connection_ok = False
        for test_url in test_urls:
            try:
                response = requests.get(test_url, headers=headers, timeout=5)
                if response.status_code in [200, 401]:
                    # 401 means key exists but no permission to users table
                    # Continue with demo mode but note the issue
                    connection_ok = True
                    if response.status_code == 200:
                        print(f"[INIT] Supabase connected successfully")
                        DB = True
                    else:
                        print(f"[INIT] Supabase key valid but limited access (401) - using demo mode")
                        DB = False
                    break
            except Exception as e:
                continue
        
        if not connection_ok:
            print("[INIT] Supabase connection test failed - running in demo mode")
            print("[INIT] Make sure:")
            print("      1. SUPABASE_URL is correct")
            print("      2. SUPABASE_KEY is valid (check expiry in .env)")
            print("      3. Row Level Security (RLS) doesn't block anonymous access")
            DB = False
    else:
        print("[INIT] Supabase credentials not set")
        DB = False
except Exception as e:
    print(f"[INIT] Connection test error: {e}")
    print("[INIT] Continuing in demo mode...")
    DB = False

# ── Helper for Supabase UUID compatibility ────────────────────
def clean_uuid(uid):
    """Return uid if it's a valid UUID string, else None. 
    Prevents Supabase 400 errors for 'demo-donor' style strings."""
    try:
        if not uid: return None
        # if it already looks like a UUID, confirm it
        uuid.UUID(str(uid))
        return str(uid)
    except Exception:
        return None

# ── Password hashing ─────────────────────────────────────────
def hash_pw(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

# ============================================================
# HYBRID PREDICTION ENGINE
# Formula + ML + Live data from Supabase
# ============================================================

# ICMR Per capita (grams per person)
PER_CAPITA = {
    "Rice/Biryani": 175, "Curry/Dal": 110, "Vegetables": 90,
    "Meat/Chicken": 135, "Sweets": 65,  "Snacks": 75, "Bread/Roti": 120,
    "Meat": 135, "Fruits": 100, "Baked Goods": 90, "Dairy Products": 80
}

# Base waste % (from food science research + your CSV avg 6.9% + TN dataset)
WASTE_BASE = {
    "Wedding": 0.25, "Corporate": 0.15, "Birthday": 0.22,
    "Social Gathering": 0.18, "Temple Festival": 0.20, "College Event": 0.17,
    "Other": 0.18
}

ATTENDANCE = {
    "Wedding": 0.87, "Corporate": 0.92, "Birthday": 0.82,
    "Social Gathering": 0.78, "Temple Festival": 0.75,
    "College Event": 0.85, "Other": 0.82
}

BUFFER = {
    "Wedding": 1.10, "Corporate": 1.05, "Birthday": 1.06,
    "Social Gathering": 1.05, "Temple Festival": 1.08,
    "College Event": 1.07, "Other": 1.05
}

METHOD_W  = {"Buffet": 1.15, "Sit-down Dinner": 1.00, "Finger Food": 0.85, "Thali": 0.95}
PRICING_W = {"High": 1.30, "Moderate": 1.00, "Low": 0.80}
SEASON_W  = {"Summer": 1.05, "Winter": 0.98, "All Seasons": 1.00}
STORAGE_W = {"Room Temperature": 1.10, "Refrigerated": 1.00}
LOCATION_W= {"Urban": 1.05, "Suburban": 1.00, "Rural": 0.95}


def formula_predict(guests, event_type, food_type, prep_method,
                    season, storage, location, pricing, qty_prepared=None):
    """Pure formula prediction using ICMR + research standards"""
    per_cap = PER_CAPITA.get(food_type, 120) / 1000  # kg
    att     = ATTENDANCE.get(event_type, 0.85)
    buf     = BUFFER.get(event_type, 1.06)

    season_att = {"Summer": 0.95, "Winter": 1.00, "All Seasons": 1.00}.get(season, 1.0)
    actual_expected = int(guests * att * season_att)
    optimal_qty = round(actual_expected * per_cap * buf, 1)

    if qty_prepared:
        q = qty_prepared
    else:
        q = optimal_qty

    waste_factor = (WASTE_BASE.get(event_type, 0.18)
                  * METHOD_W.get(prep_method, 1.00)
                  * PRICING_W.get(pricing, 1.00)
                  * SEASON_W.get(season, 1.00)
                  * STORAGE_W.get(storage, 1.00)
                  * LOCATION_W.get(location, 1.00))

    predicted_waste = round(q * waste_factor, 1)
    return optimal_qty, predicted_waste, actual_expected


def get_historical_factor(event_type, food_type, prep_method):
    """Pull real data from Supabase to improve formula"""
    if not DB:
        return None, 0
    try:
        # Try live app data first
        r1 = db_request("GET", "food_events", select="quantity_prepared,actual_wastage",
                       filters={"event_type": event_type, "food_type": food_type}, limit=100)

        # Also try TN dataset
        r2 = db_request("GET", "tn_food_surplus_dataset", select="quantity_prepared_kg,wastage_kg",
                       filters={"event_type": event_type, "food_type": food_type}, limit=100)

        rows = []
        for r in (r1 or []):
            if r.get("quantity_prepared") and r.get("actual_wastage"):
                rows.append(r["actual_wastage"] / r["quantity_prepared"])
        for r in (r2 or []):
            if r.get("quantity_prepared_kg") and r.get("wastage_kg"):
                rows.append(r["wastage_kg"] / r["quantity_prepared_kg"])

        if len(rows) < 5:
            return None, len(rows)

        return sum(rows) / len(rows), len(rows)
    except Exception as e:
        print(f"Historical fetch error: {e}")
        return None, 0


def hybrid_predict(guests, event_type, food_type, prep_method,
                   season, storage, location, pricing, qty_prepared=None):
    """
    Hybrid: Formula + Historical data from Supabase
    Confidence increases as real data grows!
    """
    # Get formula baseline
    optimal_qty, formula_waste, expected_guests = formula_predict(
        guests, event_type, food_type, prep_method,
        season, storage, location, pricing, qty_prepared
    )

    # Get historical factor from DB
    hist_factor, data_count = get_historical_factor(event_type, food_type, prep_method)

    # Try ML prediction if available
    ml_pred = None
    try:
        # determine quantity to predict on
        q_for_ml = qty_prepared if qty_prepared else optimal_qty
        ml_pred = ml_predict(guests, q_for_ml)
    except Exception:
        ml_pred = None

    q = qty_prepared if qty_prepared else optimal_qty

    if hist_factor and data_count >= 5:
        # Blend formula + historical
        blend = min(data_count / 200.0, 0.80)  # max 80% trust to historical
        hist_waste = round(q * hist_factor, 1)
        # if ML prediction exists, blend with ML as well (favor ML when available)
        if ml_pred is not None:
            # ml_pred is wastage in kg for q_for_ml; scale if q differs
            ml_waste = round(ml_pred * (q / q_for_ml), 1) if q_for_ml and q_for_ml>0 else round(ml_pred,1)
            # combine formula, historical, and ML
            final_waste = round((formula_waste * (1 - blend) + hist_waste * (blend*0.6) + ml_waste * (blend*0.4)), 1)
        else:
            final_waste = round(formula_waste * (1 - blend) + hist_waste * blend, 1)

        if data_count < 20:
            mode, confidence = "Formula+", 68
        elif data_count < 50:
            mode, confidence = "Hybrid", 75
        elif data_count < 200:
            mode, confidence = "Hybrid ML", 84
        else:
            mode, confidence = "ML Dominant", 92
    else:
        final_waste = formula_waste
        mode = "Formula"
        confidence = 65 + min(data_count * 2, 15)

        # ── Tier 3: Gemini AI fallback when confidence is low ──
        if confidence < 70 and GEMINI_API_KEY:
            ai_data, source = gemini_predict(guests, event_type, food_type, prep_method, season, pricing)
            if ai_data and ai_data.get("wastage_kg"):
                final_waste = round(float(ai_data["wastage_kg"]), 1)
                optimal_qty = round(float(ai_data.get("optimal_kg", optimal_qty)), 1)
                mode        = "Gemini AI"
                confidence  = int(ai_data.get("confidence", 78))

    # Risk level
    waste_pct = final_waste / max(q, 1) * 100
    if waste_pct > 20:   risk = "HIGH"
    elif waste_pct > 12: risk = "MEDIUM"
    else:                risk = "LOW"

    savings = max(0, round(q - optimal_qty, 1)) if qty_prepared else 0

    if qty_prepared and qty_prepared > optimal_qty * 1.1:
        rec = f"You're preparing {qty_prepared}kg. AI recommends {optimal_qty}kg - reducing by {savings}kg prevents approximately {round(savings*0.7,1)}kg waste."
    elif qty_prepared and qty_prepared < optimal_qty * 0.9:
        rec = f"You might under-prepare! AI recommends {optimal_qty}kg for {guests} guests. Consider adding {round(optimal_qty-qty_prepared,1)}kg more."
    else:
        rec = f"Prepare {optimal_qty}kg for {expected_guests} expected guests. Predicted waste: {final_waste}kg ({waste_pct:.1f}%)."

    return {
        "optimal_quantity_kg":    optimal_qty,
        "predicted_wastage_kg":   final_waste,
        "wastage_pct":            round(waste_pct, 1),
        "expected_guests":        expected_guests,
        "risk_level":             risk,
        "confidence":             confidence,
        "prediction_mode":        mode,
        "data_points_used":       data_count,
        "savings_if_optimal_kg":  savings,
        "recommendation":         rec,
        "people_can_feed":        int(final_waste * 2.5),
        "co2_save_kg":            round(final_waste * 0.5, 1),
    }


# ============================================================
# AUTH ROUTES
# ============================================================

@app.route("/api/register", methods=["POST"])
def register():
    d = request.json
    required = ["email","password","full_name","role"]
    for f in required:
        if not d.get(f): return jsonify({"error": f"Missing: {f}"}), 400

    if d["role"] not in ["donor","ngo","individual","admin"]:
        return jsonify({"error": "Invalid role"}), 400

    if DB:
        try:
            # Check if email exists
            existing = db_request("GET", "users", select="id", filters={"email": d["email"]})
            if existing and len(existing) > 0:
                return jsonify({"error": "Email already registered"}), 409

            # Insert new user
            user_data = {
                "email":         d["email"],
                "password_hash": hash_pw(d["password"]),
                "full_name":     d["full_name"],
                "phone":         d.get("phone",""),
                "role":          d["role"],
                "city":          d.get("city",""),
                "organization":  d.get("organization",""),
            }
            result = db_request("POST", "users", data=user_data)
            if result and len(result) > 0:
                user = result[0]
                session["user_id"]   = user["id"]
                session["user_role"] = user["role"]
                session["user_name"] = user["full_name"]
                return jsonify({"status":"registered","user": {
                    "id": user["id"], "name": user["full_name"],
                    "role": user["role"], "email": user["email"]
                }})
            else:
                # DB returned None – likely an RLS permission issue
                # Check server logs for the actual [DB] Error line printed above
                print(f"[REGISTER] db_request returned None for user_data={user_data}")
                return jsonify({
                    "error": "Failed to create user. Check server logs for DB error.",
                    "hint": "Run rls_fix.sql in Supabase SQL Editor to fix permissions."
                }), 500
        except Exception as e:
            print(f"[REGISTER] Exception: {e}")
            return jsonify({"error": str(e)}), 500

    # Demo mode
    demo_id = "demo-" + d["role"] + "-001"
    session["user_id"] = demo_id
    session["user_role"] = d["role"]
    session["user_name"] = d["full_name"]
    return jsonify({"status":"registered (demo)","user":{
        "id": demo_id, "name": d["full_name"], "role": d["role"], "email": d["email"]
    }})


@app.route("/api/login", methods=["POST"])
def login():
    d = request.json
    if not d.get("email") or not d.get("password"):
        return jsonify({"error": "Email and password required"}), 400

    if DB:
        try:
            result = db_request("GET", "users", select="*", 
                              filters={"email": d["email"], "password_hash": hash_pw(d["password"]), "is_active": True})
            if result and len(result) > 0:
                user = result[0]
                session["user_id"]   = user["id"]
                session["user_role"] = user["role"]
                session["user_name"] = user["full_name"]
                return jsonify({"status":"logged in","user":{
                    "id": user["id"], "name": user["full_name"],
                    "role": user["role"], "email": user["email"]
                }})
            # if DB connected but user not found, fall back to demo below
        except Exception as e:
            # log error and continue to demo fallback
            print(f"Login DB error: {e}")

    # Demo mode accounts
    demo_users = {
        "donor@demo.com":  {"role":"donor",  "name":"Demo Donor",  "id":"demo-d"},
        "ngo@demo.com":    {"role":"ngo",    "name":"Demo NGO",    "id":"demo-n"},
        "admin@demo.com":  {"role":"admin",  "name":"Admin",       "id":"demo-a"},
        "user@demo.com":   {"role":"individual","name":"Demo User","id":"demo-u"},
    }
    u = demo_users.get(d["email"])
    if u and d["password"] == "demo123":
        session["user_id"]   = u["id"]
        session["user_role"] = u["role"]
        session["user_name"] = u["name"]
        return jsonify({"status":"logged in","user":{**u, "email": d["email"]}})
    return jsonify({"error": "Invalid credentials (demo: use donor@demo.com / demo123)"}), 401


@app.route("/api/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"status": "logged out"})


@app.route("/api/me", methods=["GET"])
def me():
    if not session.get("user_id"):
        return jsonify({"error": "Not logged in"}), 401
    return jsonify({
        "user_id":   session["user_id"],
        "role":      session["user_role"],
        "name":      session["user_name"],
    })


# ============================================================
# PREDICTION ROUTE (Donor)
# ============================================================

@app.route("/api/predict", methods=["POST"])
def predict():
    print(f"\n[PREDICT] Request received. DB status={DB}, User={session.get('user_id')}")
    if not session.get("user_id"):
        return jsonify({"error": "Login required"}), 401

    d = request.json
    if not d.get("guests") or not d.get("food_type"):
        return jsonify({"error": "Missing: guests, food_type"}), 400

    print(f"[PREDICT] Input: guests={d.get('guests')}, food={d.get('food_type')}, event={d.get('event_type')}")
    
    result = hybrid_predict(
        guests        = int(d["guests"]),
        event_type    = d.get("event_type","Wedding"),
        food_type     = d.get("food_type","Rice/Biryani"),
        prep_method   = d.get("prep_method","Buffet"),
        season        = d.get("season","All Seasons"),
        storage       = d.get("storage","Refrigerated"),
        location      = d.get("location_type","Urban"),
        pricing       = d.get("pricing","Moderate"),
        qty_prepared  = float(d["qty_prepared"]) if d.get("qty_prepared") else None,
    )

    event_id = None
    if DB:
        try:
            print(f"[PREDICT] Attempting to save to food_events table...")
            event_data = {
                "user_id":           clean_uuid(session.get("user_id")),
                "organizer_name":    d.get("organizer_name",""),
                "event_type":        d.get("event_type",""),
                "food_type":         d.get("food_type",""),
                "invited_guests":    int(d["guests"]),
                "preparation_method":d.get("prep_method",""),
                "season":            d.get("season",""),
                "storage_condition": d.get("storage",""),
                "location_type":     d.get("location_type",""),
                "city":              d.get("city",""),
                "pricing_level":     d.get("pricing",""),
                "quantity_prepared": float(d["qty_prepared"]) if d.get("qty_prepared") else result["optimal_quantity_kg"],
                "predicted_wastage": result["predicted_wastage_kg"],
                "optimal_quantity":  result["optimal_quantity_kg"],
                "risk_level":        result["risk_level"],
                "confidence":        result["confidence"],
                "prediction_mode":   result["prediction_mode"],
                "event_date":        d.get("event_date"),
                "latitude":          d.get("lat"),
                "longitude":         d.get("lng"),
                "created_at":        datetime.utcnow().isoformat(),
            }
            ins = db_request("POST", "food_events", data=event_data)
            event_id = ins[0]["id"] if ins and len(ins) > 0 else None
            print(f"[PREDICT] OK Saved successfully. Event ID: {event_id}")
        except Exception as e:
            print(f"[PREDICT] ERROR Save error: {e}")
    else:
        print(f"[PREDICT] WARNING DB disabled - running in demo mode")

    return jsonify({**result, "event_id": event_id})


# ============================================================
# CONFIRM ACTUAL (Updates training data!)
# ============================================================

@app.route("/api/confirm_actual", methods=["POST"])
def confirm_actual():
    if not session.get("user_id"):
        return jsonify({"error": "Login required"}), 401

    d = request.json
    event_id       = d.get("event_id")
    actual_wastage = float(d.get("actual_wastage", 0))
    surplus_safe   = float(d.get("surplus_safe", actual_wastage * 0.8))
    food_safe      = d.get("food_safe", "yes")

    if DB and event_id:
        try:
            update_data = {
                "actual_guests":   d.get("actual_guests"),
                "actual_wastage":  actual_wastage,
                "surplus_safe_kg": surplus_safe,
                "food_safe":       food_safe,
                "was_accurate":    d.get("was_accurate") == "yes",
                "available_until": d.get("available_until"),
                "confirmed_at":    datetime.utcnow().isoformat(),
            }
            # Need to build update URL manually with filter
            url, key, headers = get_supabase_headers()
            if url and headers:
                api_url = f"{url}/rest/v1/food_events?id=eq.{event_id}"
                requests.patch(api_url, headers=headers, json=update_data)
        except Exception as e:
            print(f"Confirm error: {e}")

    matched_ngos_raw = []
    matched_ngos     = []  # list of display strings for frontend
    donor_city = ""
    donor_lat  = None
    donor_lng  = None

    # Fetch event location from DB if we have event_id
    if DB and event_id:
        try:
            ev = db_request("GET", "food_events", select="city,latitude,longitude", filters={"id": event_id})
            if ev and len(ev) > 0:
                donor_city = ev[0].get("city", "")
                donor_lat  = ev[0].get("latitude")
                donor_lng  = ev[0].get("longitude")
        except Exception:
            pass

    donor_city = donor_city or d.get("city", "")
    donor_lat  = donor_lat  or d.get("lat")
    donor_lng  = donor_lng  or d.get("lng")

    if food_safe != "no" and surplus_safe >= 5:
        matched_ngos_raw = find_nearest_ngos(
            city       = donor_city,
            lat        = donor_lat,
            lng        = donor_lng,
            surplus_kg = surplus_safe,
        )
        matched_ngos = [
            f"{n['name']} ({n['distance_km']} km)" for n in matched_ngos_raw
        ]

    # ── Notification Persistence ─────────────────────────────────
    if DB and food_safe != "no" and surplus_safe >= 5:
        try:
            if matched_ngos_raw:
                # Per-NGO targeted notifications
                for ngo in matched_ngos_raw:
                    notif_data = {
                        "title":      "URGENT: Food Surplus Alert",
                        "message":    (
                            f"{surplus_safe}kg safe food available in {donor_city or 'your area'}. "
                            f"Pickup from donor (Event #{event_id}). Distance: {ngo['distance_km']} km. "
                            f"Contact: {ngo.get('phone','N/A')}"
                        ),
                        "to_role":    "ngo",
                        "to_user":    None,
                        "is_read":    False,
                        "created_at": datetime.utcnow().isoformat(),
                    }
                    db_request("POST", "notifications", data=notif_data)
            else:
                # Broadcast to all NGOs when none are found in DB yet
                notif_data = {
                    "title":      "URGENT: Food Surplus Available",
                    "message":    (
                        f"{surplus_safe}kg of safe food available for pickup in {donor_city or 'your area'}. "
                        f"Event #{event_id}. Contact the donor immediately!"
                    ),
                    "to_role":    "ngo",
                    "to_user":    None,
                    "is_read":    False,
                    "created_at": datetime.utcnow().isoformat(),
                }
                db_request("POST", "notifications", data=notif_data)
                print(f"[NOTIFY] No NGOs found — broadcast notification created for all NGOs")
        except Exception as e:
            print(f"Notification persistence error: {e}")

    return jsonify({
        "status":        "Training data updated",
        "actual_wastage": actual_wastage,
        "surplus_safe":   surplus_safe,
        "matched_ngos":   matched_ngos,
        "alert_type":     "HARD" if matched_ngos else "NONE",
        "message":        f"Hard alert sent to {len(matched_ngos)} nearest NGOs!" if matched_ngos else "No surplus to donate.",
        "ngo_details":    matched_ngos_raw,
    })


# ============================================================
# MY ACTIVITY & IMPACT (Dynamic User Dashboard)
# ============================================================

@app.route("/api/my/events", methods=["GET"])
def my_events():
    """Return all food events for the logged-in user."""
    if not session.get("user_id"):
        return jsonify({"error": "Login required"}), 401

    uid = clean_uuid(session["user_id"])
    if DB and uid:
        try:
            events = db_request("GET", "food_events",
                select="id,event_type,food_type,invited_guests,actual_guests,quantity_prepared,"
                       "predicted_wastage,actual_wastage,surplus_safe_kg,optimal_quantity,"
                       "risk_level,confidence,prediction_mode,created_at,confirmed_at,event_date,city",
                filters={"user_id": uid},
                limit=50) or []
            return jsonify(events)
        except Exception as e:
            print(f"[MyEvents] Error: {e}")

    # Demo fallback
    return jsonify([
        {"id": 1, "event_type": "Wedding",   "food_type": "Rice/Biryani", "invited_guests": 300,
         "actual_guests": 265, "quantity_prepared": 90, "predicted_wastage": 12.5, "actual_wastage": 11.0,
         "surplus_safe_kg": 8.8, "optimal_quantity": 82.5, "risk_level": "MEDIUM", "confidence": 78,
         "prediction_mode": "Hybrid", "created_at": datetime.utcnow().isoformat(), "confirmed_at": None,
         "event_date": None, "city": "Chennai"},
        {"id": 2, "event_type": "Corporate", "food_type": "Curry/Dal",    "invited_guests": 150,
         "actual_guests": 140, "quantity_prepared": 40, "predicted_wastage": 5.2, "actual_wastage": None,
         "surplus_safe_kg": None, "optimal_quantity": 38.0, "risk_level": "LOW", "confidence": 82,
         "prediction_mode": "Hybrid ML", "created_at": datetime.utcnow().isoformat(), "confirmed_at": None,
         "event_date": None, "city": "Chennai"},
    ])


@app.route("/api/my/impact", methods=["GET"])
def my_impact():
    """Return aggregated impact stats for the logged-in user."""
    if not session.get("user_id"):
        return jsonify({"error": "Login required"}), 401

    uid = clean_uuid(session["user_id"])
    if DB and uid:
        try:
            events = db_request("GET", "food_events",
                select="invited_guests,actual_wastage,surplus_safe_kg,was_accurate,predicted_wastage",
                filters={"user_id": uid},
                limit=500) or []

            total_events       = len(events)
            confirmed          = [e for e in events if e.get("actual_wastage") is not None]
            total_food_saved   = sum(e.get("surplus_safe_kg") or 0 for e in events)
            total_wastage      = sum(e.get("actual_wastage")  or 0 for e in confirmed)
            total_guests       = sum(e.get("invited_guests")  or 0 for e in events)
            people_fed         = round(total_food_saved * 2.5)
            co2_saved          = round(total_food_saved * 0.5, 1)
            accurate_count     = sum(1 for e in confirmed if e.get("was_accurate"))
            accuracy_pct       = round(accurate_count / max(len(confirmed), 1) * 100, 1)

            return jsonify({
                "total_events":      total_events,
                "confirmed_events":  len(confirmed),
                "total_food_saved_kg": round(total_food_saved, 1),
                "total_wastage_kg":  round(total_wastage, 1),
                "people_fed":        people_fed,
                "co2_saved_kg":      co2_saved,
                "total_guests":      total_guests,
                "ai_accuracy_pct":   accuracy_pct,
                "trees_saved":       round(co2_saved / 21, 1),
            })
        except Exception as e:
            print(f"[MyImpact] Error: {e}")

    # Demo fallback
    return jsonify({
        "total_events": 2, "confirmed_events": 1,
        "total_food_saved_kg": 8.8, "total_wastage_kg": 11.0,
        "people_fed": 22, "co2_saved_kg": 4.4,
        "total_guests": 450, "ai_accuracy_pct": 88.0, "trees_saved": 0.2,
    })


# ============================================================
# NGO ROUTES
# ============================================================

@app.route("/api/report_needy", methods=["POST"])
def report_needy():
    """Public endpoint — NO LOGIN required.
    Anyone can report a needy person / orphan spotted on the roadside.
    This creates a URGENT notification visible to all NGOs."""
    d = request.json or {}
    location  = (d.get("location") or "").strip()
    city      = (d.get("city") or "").strip()
    desc      = (d.get("description") or "").strip()
    contact   = (d.get("contact") or "").strip()
    lat       = d.get("lat")
    lng       = d.get("lng")

    if not desc:
        return jsonify({"error": "Please describe the situation"}), 400
    if not city and not location:
        return jsonify({"error": "Please provide a city or location"}), 400

    # Build notification for all NGOs
    msg = (
        f"NEEDY PERSON SPOTTED\n"
        f"Location: {location or city}\n"
        f"City: {city}\n"
        f"Description: {desc}\n"
        f"Reporter contact: {contact or 'Anonymous'}"
    )

    notif_id = None
    if DB:
        try:
            notif_data = {
                "title":      "Roadside Help Needed",
                "message":    msg,
                "to_role":    "ngo",
                "to_user":    None,
                "is_read":    False,
                "created_at": datetime.utcnow().isoformat(),
            }
            res = db_request("POST", "notifications", data=notif_data)
            if res and len(res) > 0:
                notif_id = res[0].get("id")

            # Also save as guest_feedback for record keeping
            feedback = {
                "name":       contact or "Anonymous",
                "contact":    contact,
                "city":       city,
                "type":       "needy_report",
                "message":    msg,
                "created_at": datetime.utcnow().isoformat(),
            }
            db_request("POST", "guest_feedback", data=feedback)
        except Exception as e:
            print(f"[ReportNeedy] DB error: {e}")

    # Find nearest NGOs to show the reporter
    nearby = find_nearest_ngos(city=city, lat=lat, lng=lng, surplus_kg=0, max_results=5)

    return jsonify({
        "status":  "Alert sent to all NGOs!",
        "notif_id": notif_id,
        "nearby_ngos": [
            {"name": n["name"], "city": n.get("city",""), "distance_km": n["distance_km"], "phone": n.get("phone","")}
            for n in nearby
        ],
        "message": f"Thank you for reporting! Alert sent to {len(nearby)} nearby NGOs."
    })


@app.route("/api/nearest_ngos", methods=["GET"])
def api_nearest_ngos():
    """Return nearest NGOs for any logged-in user's dashboard."""
    city = request.args.get("city", "")
    lat  = request.args.get("lat")
    lng  = request.args.get("lng")

    try:
        lat = float(lat) if lat else None
        lng = float(lng) if lng else None
    except (ValueError, TypeError):
        lat, lng = None, None

    # If user is logged in, use their city from session
    if not city and session.get("user_city"):
        city = session["user_city"]

    ngos = find_nearest_ngos(city=city, lat=lat, lng=lng, surplus_kg=0, max_results=5)
    return jsonify([
        {"name": n["name"], "city": n.get("city",""), "distance_km": n["distance_km"],
         "phone": n.get("phone",""), "email": n.get("email","")}
        for n in ngos
    ])


@app.route("/api/guest/feedback", methods=["POST"])
def guest_feedback():
    """Allow guest users to send feedback / alert NGOs without login"""
    d = request.json or {}
    message = (d.get("message") or "").strip()
    if not message:
        return jsonify({"error": "Message required"}), 400

    saved_id = None
    if not DB:
        return jsonify({"error": "Database not connected"}), 500

    try:
        feedback_data = {
            "name":       d.get("name", "Guest"),
            "contact":    d.get("contact", ""),
            "city":       d.get("city", ""),
            "type":       d.get("type", "guest"),
            "message":    message,
            "created_at": datetime.utcnow().isoformat(),
        }
        res = db_request("POST", "guest_feedback", data=feedback_data)
        if res and len(res) > 0:
            saved_id = res[0].get("id")
    except Exception as e:
        print(f"Guest feedback save error: {e}")
        return jsonify({"error": str(e)}), 500

    return jsonify({"status": "received", "id": saved_id})


# ── Notifications API ───────────────────────────────────────
@app.route("/api/notifications", methods=["GET"])
def list_notifications():
    if not session.get("user_id"):
        return jsonify({"error": "Login required"}), 401

    role = session.get("user_role")
    uid  = session.get("user_id")

    if DB:
        try:
            rows = db_request("GET", "notifications", select="*") or []
            # filter by role or explicit user
            filtered = []
            for r in rows:
                tr = r.get("to_role") or r.get("to") or "all"
                tu = r.get("to_user")
                if role == "admin" or tr == "all" or tr == role or tu == uid:
                    filtered.append(r)
            # sort by created_at desc if present
            filtered.sort(key=lambda x: x.get("created_at") or "", reverse=True)
            return jsonify(filtered)
        except Exception as e:
            print(f"Notifications fetch error: {e}")

    # Demo fallback
    visible = [n for n in DEMO_NOTIFICATIONS if n.get("to_role","all") in ("all", role)]
    return jsonify(visible)


@app.route("/api/notifications/send", methods=["POST"])
def send_notification():
    if not require_admin():
        return jsonify({"error": "Admin only"}), 403
    d = request.json or {}
    msg = (d.get("message") or "").strip()
    to_role = d.get("to_role") or "all"
    if not msg:
        return jsonify({"error": "Message required"}), 400

    notif = {
        "title": d.get("title", "Admin message"),
        "message": msg,
        "to_role": to_role,
        "is_read": False,
        "created_at": datetime.utcnow().isoformat(),
    }

    if DB:
        try:
            res = db_request("POST", "notifications", data=notif)
            return jsonify(res[0] if res and len(res)>0 else notif)
        except Exception as e:
            print(f"Notification send error: {e}")
            return jsonify({"error": str(e)}), 500

    # demo
    notif["id"] = f"d{len(DEMO_NOTIFICATIONS)+1}"
    DEMO_NOTIFICATIONS.insert(0, notif)
    return jsonify(notif)


@app.route("/api/notifications/mark_read", methods=["POST"])
def mark_notifications_read():
    if not session.get("user_id"):
        return jsonify({"error": "Login required"}), 401
    d = request.json or {}
    nid = d.get("id")
    all_flag = d.get("all")

    if DB:
        try:
            if all_flag:
                # Mark all for this role/user as read
                role = session.get("user_role")
                uid = session.get("user_id")
                # Patch notifications where to_role=role OR to_role=all OR to_user=uid
                # Supabase REST doesn't support complex OR easily here; fetch then patch each
                rows = db_request("GET", "notifications", select="id,to_role,to_user") or []
                for r in rows:
                    tr = r.get("to_role") or "all"
                    tu = r.get("to_user")
                    if role == "admin" or tr == "all" or tr == role or tu == uid:
                        db_request("PATCH", "notifications", data={"is_read": True}, filters={"id": r.get("id")})
                return jsonify({"status": "ok"})
            elif nid:
                db_request("PATCH", "notifications", data={"is_read": True}, filters={"id": nid})
                return jsonify({"status": "ok"})
        except Exception as e:
            print(f"Mark read error: {e}")
            return jsonify({"error": str(e)}), 500

    # Demo fallback
    if all_flag:
        for n in DEMO_NOTIFICATIONS:
            n["is_read"] = True
        return jsonify({"status": "ok"})
    if nid:
        for n in DEMO_NOTIFICATIONS:
            if n.get("id") == nid:
                n["is_read"] = True
                return jsonify({"status": "ok"})
    return jsonify({"error": "Not found"}), 404



@app.route("/api/ngo/alerts", methods=["GET"])
def ngo_alerts():
    if not session.get("user_id"):
        return jsonify({"error": "Login required"}), 401

    if DB:
        try:
            result = db_request("GET", "food_events", 
                              select="id,event_type,food_type,surplus_safe_kg,city,risk_level,available_until,organizer_name,food_safe",
                              filters={"food_safe": "yes"}, limit=20)
            return jsonify(result or [])
        except Exception as e:
            print(f"Alerts error: {e}")

    return jsonify([
        {"id":1,"event_type":"Wedding","food_type":"Rice/Biryani","surplus_safe_kg":35,"city":"Chennai","risk_level":"HIGH","alert_type":"HARD","organizer_name":"Sri Murugan Catering"},
        {"id":2,"event_type":"Corporate","food_type":"Vegetables","surplus_safe_kg":15,"city":"Madurai","risk_level":"MEDIUM","alert_type":"SOFT","organizer_name":"TCS Cafeteria"},
    ])


@app.route("/api/ngo/register", methods=["POST"])
def ngo_register():
    if not session.get("user_id"):
        return jsonify({"error": "Login required"}), 401

    d = request.json
    if DB:
        try:
            ngo_data = {
                "user_id":    clean_uuid(session.get("user_id")),
                "name":       d.get("name",""),
                "contact":    d.get("contact",""),
                "phone":      d.get("phone",""),
                "email":      d.get("email",""),
                "city":       d.get("city",""),
                "area":       d.get("area",""),
                "capacity_kg":float(d.get("capacity",50)),
                "food_types": d.get("food_types","all"),
                "hours":      d.get("hours","8 AM - 8 PM"),
            }
            result = db_request("POST", "ngos", data=ngo_data)
            return jsonify({"status":"NGO registered!", "id": result[0]["id"] if result and len(result) > 0 else None})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"status":"NGO registered (demo)!"})


# ============================================================
# ADMIN ROUTES
# ============================================================

def require_admin():
    if session.get("user_role") != "admin":
        return False
    return True


@app.route('/api/admin/train_model', methods=['POST'])
def admin_train_model():
    if not require_admin():
        return jsonify({"error":"Admin only"}), 403
    
    d = request.json or {}
    limit = int(d.get("limit", 10000))
    cnt = train_ml_model(force=True, limit=limit)
    
    if cnt and cnt>0:
        return jsonify({"status":"trained","rows":cnt})
    return jsonify({"status":"not_trained","rows":cnt}), 500


@app.route("/api/admin/stats", methods=["GET"])
def admin_stats():
    if not require_admin():
        return jsonify({"error": "Admin only"}), 403

    if DB:
        try:
            users    = db_request("GET", "users", select="id,role") or []
            events   = db_request("GET", "food_events", select="actual_wastage,surplus_safe_kg,was_accurate,prediction_mode") or []
            tn_rows  = db_request("GET", "tn_food_surplus_dataset", select="id") or []
            orig_rows= db_request("GET", "original_food_dataset", select="id") or []

            confirmed = [e for e in events if e.get("actual_wastage")]
            total_saved = sum(e["surplus_safe_kg"] or 0 for e in events if e.get("surplus_safe_kg"))
            accuracy = 0
            if confirmed:
                accuracy = len([e for e in confirmed if e.get("was_accurate")]) / len(confirmed) * 100

            return jsonify({
                "total_users":       len(users),
                "donors":            len([u for u in users if u.get("role")=="donor"]),
                "ngos":              len([u for u in users if u.get("role")=="ngo"]),
                "individuals":       len([u for u in users if u.get("role")=="individual"]),
                "total_events":      len(events),
                "confirmed_events":  len(confirmed),
                "total_food_saved":  round(total_saved, 1),
                "people_fed":        int(total_saved * 2.5),
                "co2_saved":         round(total_saved * 0.5, 1),
                "ai_accuracy":       round(accuracy, 1),
                "tn_dataset_rows":   len(tn_rows),
                "original_dataset":  len(orig_rows),
                "total_training_data": len(tn_rows) + len(orig_rows) + len(confirmed),
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({
        "total_users":4,"donors":2,"ngos":1,"individuals":1,
        "total_events":12,"confirmed_events":8,"total_food_saved":320.5,
        "people_fed":801,"co2_saved":160.2,"ai_accuracy":78.5,
        "tn_dataset_rows":5000,"original_dataset":1782,"total_training_data":6790,
    })


@app.route("/api/admin/upload_dataset", methods=["POST"])
def upload_dataset():
    if not require_admin():
        return jsonify({"error": "Admin only"}), 403

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    dataset_type = request.form.get("dataset_type", "tn")  # tn or original

    content = file.read().decode("utf-8-sig")
    reader  = csv.DictReader(io.StringIO(content))
    rows    = list(reader)

    if not rows:
        return jsonify({"error": "Empty CSV"}), 400

    if not DB:
        return jsonify({"status": "Demo mode - would insert rows", "count": len(rows)})

    inserted = 0
    errors   = 0
    batch    = []

    try:
        for row in rows:
            try:
                if dataset_type == "tn":
                    # TamilNadu_FoodSurplus_Dataset.csv
                    batch.append({
                        "event_id_ref":           row.get("Event_ID",""),
                        "date":                   row.get("Date") or None,
                        "city":                   row.get("City",""),
                        "event_type":             row.get("Event_Type",""),
                        "food_type":              row.get("Food_Type",""),
                        "invited_guests":         int(float(row.get("Invited_Guests",0) or 0)),
                        "actual_guests":          int(float(row.get("Actual_Guests",0) or 0)),
                        "attendance_rate_pct":    float(row.get("Attendance_Rate_Pct",0) or 0),
                        "preparation_method":     row.get("Preparation_Method",""),
                        "season":                 row.get("Season",""),
                        "storage_condition":      row.get("Storage_Condition",""),
                        "location_type":          row.get("Location_Type",""),
                        "pricing_level":          row.get("Pricing_Level",""),
                        "quantity_prepared_kg":   float(row.get("Quantity_Prepared_kg",0) or 0),
                        "optimal_quantity_kg":    float(row.get("Optimal_Quantity_kg",0) or 0),
                        "wastage_kg":             float(row.get("Wastage_kg",0) or 0),
                        "wastage_pct":            float(row.get("Wastage_Pct",0) or 0),
                        "surplus_safe_to_donate": float(row.get("Surplus_Safe_to_Donate_kg",0) or 0),
                        "donated_to_ngo":         row.get("Donated_to_NGO",""),
                        "co2_saved_kg":           float(row.get("CO2_Saved_kg",0) or 0),
                        "people_fed":             int(float(row.get("People_Fed",0) or 0)),
                        "uploaded_by":            session["user_id"],
                        "uploaded_at":            datetime.utcnow().isoformat(),
                    })
                else:
                    # Original food_wastage_data.csv
                    batch.append({
                        "type_of_food":           row.get("Type of Food",""),
                        "number_of_guests":       int(float(row.get("Number of Guests",0) or 0)),
                        "event_type":             row.get("Event Type",""),
                        "quantity_of_food":       float(row.get("Quantity of Food",0) or 0),
                        "storage_conditions":     row.get("Storage Conditions",""),
                        "purchase_history":       row.get("Purchase History",""),
                        "seasonality":            row.get("Seasonality",""),
                        "preparation_method":     row.get("Preparation Method",""),
                        "geographical_location":  row.get("Geographical Location",""),
                        "pricing":                row.get("Pricing",""),
                        "wastage_food_amount":    float(row.get("Wastage Food Amount",0) or 0),
                        "uploaded_by":            clean_uuid(session.get("user_id")),
                        "uploaded_at":            datetime.utcnow().isoformat(),
                    })

                # Insert in batches of 100
                if len(batch) >= 100:
                    table = "tn_food_surplus_dataset" if dataset_type=="tn" else "original_food_dataset"
                    db_request("POST", table, data=batch)
                    inserted += len(batch)
                    batch = []

            except Exception as row_err:
                errors += 1

        # Insert remaining
        if batch:
            table = "tn_food_surplus_dataset" if dataset_type=="tn" else "original_food_dataset"
            db_request("POST", table, data=batch)
            inserted += len(batch)

    except Exception as e:
        return jsonify({"error": str(e), "inserted": inserted}), 500

    return jsonify({
        "status":   "Dataset uploaded",
        "inserted": inserted,
        "errors":   errors,
        "table":    "tn_food_surplus_dataset" if dataset_type=="tn" else "original_food_dataset",
        "message":  f"{inserted} rows added to Supabase. AI will use this data for future predictions!"
    })


@app.route("/api/admin/users", methods=["GET"])
def admin_users():
    if not require_admin():
        return jsonify({"error": "Admin only"}), 403
    if DB:
        try:
            result = db_request("GET", "users", select="id,email,full_name,role,city,organization,created_at,is_active")
            return jsonify(result or [])
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify([
        {"id":"1","email":"donor@demo.com","full_name":"Demo Donor","role":"donor","city":"Chennai","is_active":True},
        {"id":"2","email":"ngo@demo.com","full_name":"Hope NGO","role":"ngo","city":"Madurai","is_active":True},
    ])


@app.route("/api/admin/events", methods=["GET"])
def admin_events():
    if not require_admin():
        return jsonify({"error": "Admin only"}), 403
    if DB:
        try:
            result = db_request("GET", "food_events", select="*", limit=100)
            return jsonify(result or [])
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify([])





@app.route("/api/admin/training_stats", methods=["GET"])
def training_stats():
    if not require_admin():
        return jsonify({"error": "Admin only"}), 403
    if DB:
        try:
            tn   = db_request("GET", "tn_food_surplus_dataset", select="event_type,wastage_pct") or []
            orig = db_request("GET", "original_food_dataset", select="event_type,wastage_food_amount,quantity_of_food") or []
            live = db_request("GET", "food_events", select="event_type,actual_wastage,quantity_prepared,prediction_mode", 
                            filters={"actual_wastage": None}) or []
            return jsonify({
                "tn_dataset":       len(tn),
                "original_dataset": len(orig),
                "live_data":        len(live),
                "total":            len(tn) + len(orig) + len(live),
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"tn_dataset":5000,"original_dataset":1782,"live_data":8,"total":6790})


# ============================================================
# DROPDOWN CONFIG ROUTES (Dynamic options from admin)
# ============================================================

@app.route("/api/config/food-types", methods=["GET"])
def get_food_types():
    """Get available food types for dropdown"""
    if DB:
        try:
            print("[DROPDOWN] Fetching food types from DB...")
            result = db_request("GET", "config_food_types", select="id,name")
            print(f"[DROPDOWN] Food types result: {result}")
            if result:
                foods = [r["name"] for r in result]
                print(f"[DROPDOWN] OK Returning {len(foods)} food types from DB")
                return jsonify(foods)
        except Exception as e:
            print(f"[DROPDOWN] ERROR Food types fetch error: {e}")
    
    # Fallback defaults
    fallback = ["Rice/Biryani","Curry/Dal","Vegetables","Meat/Chicken","Sweets","Snacks","Bread/Roti","Fruits","Baked Goods","Dairy Products"]
    print(f"[DROPDOWN] Using fallback food types (DB={DB})")
    return jsonify(fallback)

@app.route("/api/config/event-types", methods=["GET"])
def get_event_types():
    """Get available event types for dropdown"""
    if DB:
        try:
            print("[DROPDOWN] Fetching event types from DB...")
            result = db_request("GET", "config_event_types", select="id,name")
            print(f"[DROPDOWN] Event types result: {result}")
            if result:
                events = [r["name"] for r in result]
                print(f"[DROPDOWN] OK Returning {len(events)} event types from DB")
                return jsonify(events)
        except Exception as e:
            print(f"[DROPDOWN] ERROR Event types fetch error: {e}")
    
    # Fallback defaults
    fallback = ["Wedding","Corporate","Birthday","Social Gathering","Temple Festival","College Event","Other"]
    print(f"[DROPDOWN] Using fallback event types (DB={DB})")
    return jsonify(fallback)

@app.route("/api/admin/config/food-types", methods=["GET", "POST"])
def admin_food_types():
    """Admin: Get or add food types"""
    if not require_admin():
        return jsonify({"error": "Admin only"}), 403
    
    if request.method == "GET":
        if DB:
            try:
                result = db_request("GET", "config_food_types", select="id,name")
                return jsonify(result or [])
            except Exception as e:
                print(f"Admin food types error: {e}")
        return jsonify([
            {"id":1,"name":"Rice/Biryani"},
            {"id":2,"name":"Curry/Dal"},
        ])
    
    # POST: Add new food type
    d = request.json
    if not d.get("name"):
        return jsonify({"error":"Name required"}), 400
    
    if DB:
        try:
            result = db_request("POST", "config_food_types", data={"name": d["name"]})
            return jsonify({"status":"Added","id":result[0]["id"] if result and len(result) > 0 else None})
        except Exception as e:
            return jsonify({"error":str(e)}), 500
    
    return jsonify({"status":"Added (demo)"})

@app.route("/api/admin/config/event-types", methods=["GET", "POST"])
def admin_event_types():
    """Admin: Get or add event types"""
    if not require_admin():
        return jsonify({"error": "Admin only"}), 403
    
    if request.method == "GET":
        if DB:
            try:
                result = db_request("GET", "config_event_types", select="id,name")
                return jsonify(result or [])
            except Exception as e:
                print(f"Admin event types error: {e}")
        return jsonify([
            {"id":1,"name":"Wedding"},
            {"id":2,"name":"Corporate"},
        ])
    
    # POST: Add new event type
    d = request.json
    if not d.get("name"):
        return jsonify({"error":"Name required"}), 400
    
    if DB:
        try:
            result = db_request("POST", "config_event_types", data={"name": d["name"]})
            return jsonify({"status":"Added","id":result[0]["id"] if result and len(result) > 0 else None})
        except Exception as e:
            return jsonify({"error":str(e)}), 500
    
    return jsonify({"status":"Added (demo)"})

@app.route("/", methods=["GET"])
def index():
    # Serve frontend if available
    try:
        return send_from_directory('.', 'index.html')
    except Exception:
        return jsonify({"message":"FoodBridge AI Running","status":"ok"})


if __name__ == "__main__":
    print("\nFoodBridge AI")
    print("http://localhost:5000")
    print("Demo accounts: donor@demo.com | ngo@demo.com | admin@demo.com | user@demo.com")
    print("Password for all: demo123")
    app.run(debug=True, host="0.0.0.0", port=5000)
