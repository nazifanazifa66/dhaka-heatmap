import json
import pandas

geo=json.load(open("Dhaka_Heatmap_500m_Grid.geojson"))
csv=pandas.read_csv("Dhaka_Heatmap_Grid_Features.csv")
geo_ids=set()
for feature in geo["features"]:
 geo_ids.add(str(feature['properties']["grid_id"]))
csv_ids=set(csv["system:index"].astype(str))

print("GeoJSON:",len(geo_ids))
print("CSV:",len(csv_ids))
print("MATCH:",geo_ids==csv_ids)
print("Missing:",len(csv_ids-geo_ids))
print("Extra:",len(geo_ids-csv_ids))