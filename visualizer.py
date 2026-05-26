import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import folium
from folium.plugins import MarkerCluster
import os

os.makedirs("output", exist_ok=True)

df=pd.read_csv("data/trips_scored.csv")
drivers=pd.read_csv("data/driver_summary.csv")

# Color scheme: green = safe, orange = moderate, red = risky.
COLORS={"safe": "#2ecc71", "moderate": "#f39c12", "risky": "#e74c3c"}

# Create a 2x3 grid of charts — one figure, six panels.
# figsize is in inches; 16x10 gives enough room for labels to breathe.
fig,axes=plt.subplots(2, 3, figsize=(16, 10))
fig.patch.set_facecolor("#f8f9fa")
fig.suptitle("Driver Behavior Analysis Dashboard", fontsize=18, fontweight="bold", y=1.01)

#How many trips fell into each category
ax = axes[0, 0]
counts = df["predicted_label"].value_counts()[["safe", "moderate", "risky"]]
bars = ax.bar(counts.index, counts.values,
              color=[COLORS[l] for l in counts.index], edgecolor="white", linewidth=1.5)
ax.set_title("Trip classifications", fontweight="bold")
ax.set_ylabel("Number of trips")
# Add the exact count on top of each bar so you don't have to read the axis
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
            str(val), ha="center", fontsize=11, fontweight="bold")
ax.set_facecolor("#f8f9fa")
ax.spines[["top","right"]].set_visible(False)


#What does the risk score distribution look like?
ax = axes[0, 1]
for label, color in COLORS.items():
    subset = df[df["predicted_label"] == label]["risk_score"]
    ax.hist(subset, bins=20, alpha=0.7, color=color, label=label.capitalize(), edgecolor="white")
ax.set_title("Risk score distribution", fontweight="bold")
ax.set_xlabel("Risk score (0–100)")
ax.set_ylabel("Trips")
ax.legend()
ax.set_facecolor("#f8f9fa")
ax.spines[["top","right"]].set_visible(False)

#Does phone use correlate with hard braking?
ax = axes[0, 2]
for label, color in COLORS.items():
    sub = df[df["predicted_label"] == label]
    ax.scatter(sub["phone_use_sec"], sub["hard_brakes"],
               c=color, alpha=0.5, s=25, label=label.capitalize())
ax.set_title("Phone use vs hard braking", fontweight="bold")
ax.set_xlabel("Phone use (seconds)")
ax.set_ylabel("Hard brakes")
ax.legend()
ax.set_facecolor("#f8f9fa")
ax.spines[["top","right"]].set_visible(False)

#Whta proportion of drivers fall into each tier?
ax = axes[1, 1]
tier_counts = drivers["driver_tier"].value_counts()[["Safe", "Moderate", "Risky"]]
wedges, texts, autotexts = ax.pie(
    tier_counts.values, labels=tier_counts.index,
    colors=["#2ecc71","#f39c12","#e74c3c"], autopct="%1.0f%%",
    startangle=90, wedgeprops=dict(edgecolor="white", linewidth=2)
)
for t in autotexts:
    t.set_fontsize(12); t.set_fontweight("bold")
ax.set_title("Driver tier breakdown", fontweight="bold")

#Who are the riskiest individual drivers?
ax = axes[1, 2]
top10 = drivers.head(10)
colors_bar = [COLORS.get(str(t).lower(), "#999") for t in top10["driver_tier"]]
ax.barh(top10["driver_id"], top10["avg_risk_score"], color=colors_bar, edgecolor="white")
ax.set_title("Top 10 riskiest drivers", fontweight="bold")
ax.set_xlabel("Avg risk score")
ax.invert_yaxis()  # highest score at the top
ax.set_facecolor("#f8f9fa")
ax.spines[["top","right"]].set_visible(False)
legend_patches = [mpatches.Patch(color=c, label=l.capitalize()) for l, c in COLORS.items()]
ax.legend(handles=legend_patches, fontsize=9)

# Do drivers speed more on certain road types?
ax = axes[1, 0]
road_types = ["urban", "suburban", "highway"]
for i, road in enumerate(road_types):
    sub = df[df["road_type"] == road]["speeding_pct"]
    ax.boxplot(sub, positions=[i], widths=0.5, patch_artist=True,
               boxprops=dict(facecolor="#4a90d9", alpha=0.6),
               medianprops=dict(color="#2c3e50", linewidth=2),
               whiskerprops=dict(color="#555"), capprops=dict(color="#555"),
               flierprops=dict(marker="o", markersize=3, alpha=0.3))
ax.set_xticks(range(len(road_types)))
ax.set_xticklabels([r.capitalize() for r in road_types])
ax.set_title("Speeding % by road type", fontweight="bold")
ax.set_ylabel("% of trip spent speeding")
ax.set_facecolor("#f8f9fa")
ax.spines[["top","right"]].set_visible(False)



plt.tight_layout()
plt.savefig("output/dashboard.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved -> output/dashboard.png")


#Interactive Risk Map
m = folium.Map(location=[39.5, -98.35], zoom_start=4, tiles="CartoDB positron")

cluster = MarkerCluster(name="Trips").add_to(m)
risk_hex = {"safe": "#2ecc71", "moderate": "#f39c12", "risky": "#e74c3c"}

for _, row in df.iterrows():
    label = row["predicted_label"]
    color= risk_hex[label]
    # Each marker gets a popup — clicking it shows the full trip breakdown.
    # This is what a traffic safety analyst would use to investigate a hotspot.

    popup_html = f"""
    <div style="font-family:sans-serif;font-size:13px;min-width:180px;">
      <b>{row['trip_id']}</b><br>Driver: {row['driver_id']}<br>
      <span style="color:{color};font-weight:bold;">{label.upper()}</span> (score: {row['risk_score']})<br>
      <hr style="margin:4px 0">
      Hard brakes: {row['hard_brakes']}<br>
      Phone use: {row['phone_use_sec']}s<br>
      Speeding: {row['speeding_pct']:.1f}%<br>
      Max speed: {row['max_speed_kmh']} km/h<br>
      <i style="color:#888">{row['risk_flags']}</i>
    </div>"""

    folium.CircleMarker(
        location=[row["start_lat"], row["start_lon"]],
        radius=7, color=color, fill=True, fill_color=color, fill_opacity=0.75,
        popup=folium.Popup(popup_html, max_width=220),
        tooltip=f"{row['trip_id']} — {label} ({row['risk_score']})"
    ).add_to(cluster)

legend_html = """
<div style="position:fixed;bottom:30px;left:30px;z-index:1000;background:white;
     padding:12px 16px;border-radius:8px;border:1px solid #ddd;
     font-family:sans-serif;font-size:13px;line-height:1.8;">
  <b>Trip Risk Level</b><br>
  <span style="color:#2ecc71">●</span> Safe<br>
  <span style="color:#f39c12">●</span> Moderate<br>
  <span style="color:#e74c3c">●</span> Risky
</div>"""
m.get_root().html.add_child(folium.Element(legend_html))
m.save("output/risk_map.html")
print("Saved -> output/risk_map.html")
print("\nDone! Open output/dashboard.png and output/risk_map.html to explore.")





