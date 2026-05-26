import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report,accuracy_score
import warnings
from lightgbm import LGBMClassifier
warnings.filterwarnings("ignore")

df=pd.read_csv("data/trips.csv")
print(f"Loaded {len(df)} trips\n")

#Feature Enginnering
'''
Hard events per km: a driver who had 5 hard brakes on a 2km trip is way more dangerous than one who had 5 hard brakes on a 100km highway trip.
Normalizing by distance makes the comparison fair.
'''

df["events_per_km"]=((df["hard_brakes"]+df["sharp_corners"])/df["trip_distance_km"].clip(lower=0.1))#clip avoids divind by 0

#60 seconds on a 5-min trip is very different from 60 seconds on a 90-min trip.
df["phone_per_min"]=df["phone_use_sec"]/df["trip_duration_min"].clip(lower=0.1)

#converting the text categories to numbers
df["is_night"]=(df["time_of_day"]=="night").astype(int) #1 if night, 0 otherwise

df["is_urban"]  = (df["road_type"] == "urban").astype(int)

# Average speed: distance / time gives us another useful signal
df["avg_speed_kmh"] = (df["trip_distance_km"] / (df["trip_duration_min"] / 60)).clip(0, 200)

FEATURES = [
    "hard_brakes", "hard_accel", "sharp_corners",
    "phone_use_sec", "speeding_pct", "max_speed_kmh",
    "smooth_score", "events_per_km", "phone_per_min",
    "is_night", "is_urban", "trip_distance_km", "avg_speed_kmh",
]

#Rule based Risk Score
'''
Before we use ML, we need to compute a simple human-readable score (0-100). This is how 
telematics companies explain risk to non-technical users
'''

def compute_risk_score(row):
    score=min(row["hard_brakes"] * 6, 25)       # up to 25 points for hard braking
    score += min(row["hard_accel"] * 4, 15)        # up to 15 for aggressive acceleration
    score += min(row["sharp_corners"] * 4, 12)     # up to 12 for sharp corners
    score += min(row["phone_use_sec"] / 10, 20)    # up to 20 for phone distraction
    score += min(row["speeding_pct"] * 0.4, 15)    # up to 15 for speeding
    score += max(0, (row["max_speed_kmh"] - 120) * 0.15)  # penalty for going very fast
    score += (100 - row["smooth_score"]) * 0.10    # low smoothness adds to risk
    score += row["is_night"] * 3                   # small penalty for night driving
    return round(min(score, 100), 1)


df["risk_score"]=df.apply(compute_risk_score, axis=1)

#Train the classifier
#the labelencoder converts text labels to numbers
le=LabelEncoder()
df["label"]=le.fit_transform(df["archetype"])

X=df[FEATURES]
y=df["label"]


#Split into training and test sets
#Here we train on 75% of the dtaa and evaluate on the remaining 25% the modelhas never seen.
#startify=y makes sure each risk class is proportionally represented in both sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)


#Defining the models

#Random Forest
rf=RandomForestClassifier(n_estimators=150,max_depth=8,random_state=42)

#LightGBM-much faster than Randomforest on large datasets
lgbm=LGBMClassifier(n_estimators=200,num_leaves=31,random_state=42,verbose=-1)

#Stacking Ensemble
'''RF and LightGBM each make predictions, then a Logistic Regression "meta-model" learns the best way to combine their votes.
Often outperforms either model alone because they make different kinds of errors. 
cv=5 means the base models are trained on 5 folds to avoid leaking training data  into the meta-model — this keeps the stacking honest.
'''
stacking=StackingClassifier(
    estimators=[
        ("rf",RandomForestClassifier(n_estimators=150,max_depth=8,random_state=42)),
        ("lgbm", LGBMClassifier(n_estimators=200, num_leaves=31, random_state=42, verbose=-1)),
    ],
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5,
    n_jobs=-1 # use all CPU cores to speed up the 5-fold training
)

models={"Random Forest":rf,"LightGBM":lgbm,"Stacking Ensemble":stacking,}

#Train and compare all 3 models
print("Training and evaluating all models...\n")
print("─" * 60)

results={}

for name,model in models.items():
    print(f"{name}\n")

    #train on the training set
    model.fit(X_train,y_train)

    #Predict on the test set
    y_pred=model.predict(X_test)

    # cross_val_score splits training data into 5 folds and trains/evaluates
    # 5 times — gives a more reliable accuracy estimate than a single train/test split
    cv_scores=cross_val_score(model, X_train, y_train, cv=5, scoring="f1_weighted")

    test_acc=accuracy_score(y_test, y_pred)
    cv_mean =cv_scores.mean()
    cv_std=cv_scores.std()


    results[name]={"test_accuracy": round(test_acc, 4),"cv_f1_mean":round(cv_mean, 4),"cv_f1_std":round(cv_std, 4),}

    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print(f"  Cross-val F1 (5-fold): {cv_mean:.4f} ± {cv_std:.4f}")


#Model comparison
print("\n" + "═" * 60)
print("MODEL COMPARISON")
print("═" * 60)
print(f"  {'Model':<25} {'Test Acc':>10} {'CV F1':>10} {'CV Std':>10}")
print("  " + "─" * 55)
for name, metrics in results.items():
    print(f"  {name:<25} {metrics['test_accuracy']:>10.4f} {metrics['cv_f1_mean']:>10.4f} {metrics['cv_f1_std']:>10.4f}")

# Identify and announce the winner
best_model_name = max(results, key=lambda k: results[k]["cv_f1_mean"])
print(f"\n  Best model by CV F1: {best_model_name}")
print("═" * 60)


#Use the best model for final predictions
# We pick the winner and use it to score every trip in the full dataset
best_model = models[best_model_name]
print(f"\nUsing {best_model_name} for final predictions...")

df["predicted_class_id"] = best_model.predict(X)
df["predicted_label"]    = le.inverse_transform(df["predicted_class_id"])

# predict_proba gives confidence per class, not just a hard label.
proba = best_model.predict_proba(X)
for i, cls in enumerate(le.classes_):
    df[f"prob_{cls}"] = proba[:, i].round(3)

#Feature importances for Random forest and lightBGN only
# Stacking doesn't expose importances directly, so we read from LightGBM.
print("\n Feature importances according to LightBGM")
lgbm_importances = pd.Series(
    lgbm.feature_importances_, index=FEATURES
).sort_values(ascending=False)

for feat, imp in lgbm_importances.head(8).items():
    bar = "█" * int((imp / lgbm_importances.max()) * 30)
    print(f"  {feat:<22} {bar} {imp}")

#Why a trip was flagged as risky
def flag_trip(row):
    flags=[]
    if row["hard_brakes"] >= 4:flags.append("excessive braking")
    if row["phone_use_sec"] >= 60:flags.append("phone distraction")
    if row["speeding_pct"] >= 20:flags.append("frequent speeding")
    if row["max_speed_kmh"] >= 130:flags.append("high max speed")
    if row["sharp_corners"] >= 4:flags.append("aggressive cornering")
    if row["is_night"] and row["predicted_label"] != "safe":
        flags.append("night risk")
    return "; ".join(flags) if flags else "none"

df["risk_flags"]=df.apply(flag_trip,axis=1)

#Driver-level summary
driver_summary = df.groupby("driver_id").agg(
    total_trips=("trip_id", "count"),
    avg_risk_score=("risk_score", "mean"),
    pct_risky_trips=("predicted_label", lambda x: (x == "risky").mean() * 100),
    pct_safe_trips=("predicted_label", lambda x: (x == "safe").mean() * 100),
    avg_hard_brakes=("hard_brakes", "mean"),
    avg_phone_use_sec=("phone_use_sec", "mean"),
    avg_speeding_pct=("speeding_pct", "mean"),
    total_km=("trip_distance_km", "sum"),
).round(2).reset_index()

driver_summary["driver_tier"] = pd.cut(
    driver_summary["avg_risk_score"],
    bins=[0, 30, 55, 100],
    labels=["Safe", "Moderate", "Risky"]
)
driver_summary = driver_summary.sort_values("avg_risk_score", ascending=False)

df.to_csv("data/trips_scored.csv", index=False)
driver_summary.to_csv("data/driver_summary.csv", index=False)


print(f"\n Driver Tier Distribution")
print(driver_summary["driver_tier"].value_counts().to_string())
print(f"\nSaved to data/trips_scored.csv")
print(f"Saved to  data/driver_summary.csv")


          
