'''
We don't have real GPS/sensor data, so we fake it. The trick is making it realistic — not just random noise.
We do this by giving each driver a personality (archetype) and generating their behavior based on that personality.

'''

import numpy as np
import pandas as pd
import os

np.random.seed(42)
os.makedirs("data",exist_ok=True)

n_trips=500
n_drivers=50

driver_ids=[f"DRIVER_{i:03d}" for i in range(n_drivers)]

# Randomly assign each driver a driving personality.
# 40% are safe, 40% are moderate, 20% are risky 
archetypes=np.random.choice(["safe","moderate","risky"],size=n_drivers,p=[0.4,0.4,0.2])
driver_archetype=dict(zip(driver_ids,archetypes)) # maps driver_id to their archetype

records=[] # we'll collect each trip as a dictionary and build a table at the end

for trip_id in range(n_trips):
    # Pick a random driver for this trip
    driver_id=np.random.choice(driver_ids)
    archetype=driver_archetype[driver_id]

    # How long and far was the trip?
    trip_distance_km=np.random.uniform(2,80)
    trip_duration_min=trip_distance_km/np.random.uniform(0.4,1.2)
    # When did they drive? Affects risk (night driving is more dangerous)
    time_of_day=np.random.choice(["morning","afternoon","evening","night"], p=[0.25,0.30,0.25,0.20])

    road_type=np.random.choice(["urban","suburban","highway"], p=[0.35,0.40,0.25])


    '''
    Now generate sensor readings based on the driver's archetype.
    poisson() is good for counts (like "how many times did they brake hard").
    exponential() is good for things with a long tail (like phone use — most people use it a little, but some use it a LOT).
    normal() is good for things clustered around an average (like max speed).
    '''
    if archetype=="safe":
        hard_brakes=np.random.poisson(0.5)
        hard_accel=np.random.poisson(0.4)
        sharp_corners=np.random.poisson(0.3)
        phone_use_sec=np.random.exponential(5)
        speeding_pct=np.random.beta(1,12)*100
        max_speed_kmh=np.random.normal(95,10)
        smooth_score=np.random.normal(88,5)
    
    elif archetype=="moderate":
        hard_brakes=np.random.poisson(2.5)
        hard_accel=np.random.poisson(2.0)
        sharp_corners=np.random.poisson(1.5)
        phone_use_sec=np.random.exponential(25)
        speeding_pct=np.random.beta(2,6)*100
        max_speed_kmh=np.random.normal(115,15)
        smooth_score=np.random.normal(68,8)
    
    else:
        hard_brakes=np.random.poisson(6)
        hard_accel=np.random.poisson(5)
        sharp_corners=np.random.poisson(4)
        phone_use_sec=np.random.exponential(70)
        speeding_pct=np.random.beta(4,4)*100
        max_speed_kmh=np.random.normal(138,20)
        smooth_score=np.random.normal(45,12)
     # Night driving makes everyone a little worse, harder to see, more fatigued, more distracted
    if time_of_day=="night":
        hard_brakes=int(hard_brakes*1.3)
        phone_use_sec=phone_use_sec*1.2
    
    # Clip everything to physically realistic ranges so we don't get
    # values like -3 hard brakes or 500 km/h max speed
    max_speed_kmh=np.clip(max_speed_kmh,40,220)
    smooth_score=np.clip(smooth_score,0,100)
    speeding_pct=np.clip(speeding_pct,0,100)
    phone_use_sec=np.clip(phone_use_sec,0,600)
    hard_brakes=max(0,int(hard_brakes))
    hard_accel=max(0,int(hard_accel))
    sharp_corners=max(0,int(sharp_corners))


    # Here we are placing the trip start near a real US city so the map looks meaningful.
    # We add a small random offset so trips aren't all stacked on the same point.
    city_centers=[(42.36, -71.06), (40.71, -74.00), (41.88, -87.63),
        (37.77, -122.42), (34.05, -118.24),]
    
    base_lat,base_lon=city_centers[np.random.randint(len(city_centers))]
    start_lat = base_lat + np.random.uniform(-0.3, 0.3)
    start_lon = base_lon + np.random.uniform(-0.3, 0.3)

    records.append({
        "trip_id": f"TRIP_{trip_id:04d}", "driver_id": driver_id,
        "archetype": archetype, "trip_distance_km": round(trip_distance_km, 2),
        "trip_duration_min": round(trip_duration_min, 1), "time_of_day": time_of_day,
        "road_type": road_type, "hard_brakes": hard_brakes, "hard_accel": hard_accel,
        "sharp_corners": sharp_corners, "phone_use_sec": round(phone_use_sec, 1),
        "speeding_pct": round(speeding_pct, 2), "max_speed_kmh": round(max_speed_kmh, 1),
        "smooth_score": round(smooth_score, 1), "start_lat": round(start_lat, 5),
        "start_lon": round(start_lon, 5),
    })

df = pd.DataFrame(records)
df.to_csv("data/trips.csv", index=False)
print(f"Generated {len(df)} trips for {n_drivers} drivers -> data/trips.csv")
print(df[["archetype","hard_brakes","phone_use_sec","speeding_pct","smooth_score"]].groupby("archetype").mean().round(2))


