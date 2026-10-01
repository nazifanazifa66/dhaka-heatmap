import json
import pandas as pd 
import numpy as np
features=pd.read_csv("Dhaka_Heatmap_Grid_Features.csv")
risk=pd.read_csv("dhaka_heat_risk_data.csv")
print("Feature rows:",len(features))
print("Risk rows:",len(risk))

if len(features) != len(risk):
 print("ERROR:Row counts does not match:")
 raise SystemExit
checks=[
 np.allclose(features["NDVI"],risk["NDVI"],
equal_nan=True),
 np.allclose(features["NDBI"],risk["NDBI"],
equal_nan=True),
 np.allclose(features["NDWI"],risk["NDWI"],
equal_nan=True)]

print("NDVI match:",checks[0])
print("NDVI match:",checks[1])
print("NDVI match:",checks[2])
if not all (checks):
 print("ERROR:Data order does not match.")
 raise SystemExit

risk["system:index"]= features["system:index"].astype(str)

with open("Dhaka_Heatmap_500m_Grid.geojson","r",
encoding="utf-8")as f:
 geo=json.load(f)

risk_lookup=risk.set_index("system.index").to_dict("index")

for feature in geo["features"]:
 grid_id=str(feature["properties"]["grid_id"])

 if grid_id in risk_lookup:
  features["properties"].update(risk_lookup[grid_id])

with open("Dhaka_Heatmap_Final.geojson","w",encoding="utf-8")as f:
 json.dump(geo,f)
print("SUCCESS!")
print("Final GeoJSON Grids:",len(geo["features"]))
print("Created:Dhaka_Heatmap_Final.geojson")