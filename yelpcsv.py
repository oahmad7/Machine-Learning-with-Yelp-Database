#!/usr/bin/env python3

import csv
import json
from collections import defaultdict

objs = []

with open("./business.json") as f:
    # Each line in business.json is a separate json object
    for line in f:
        # Read the line, convert it into a dictionary with json.loads,
        # then convert THAT into a defaultdict so that if we ask for a
        # field that a line doesn't have, it will give an empty string
        # instead of throwing an error
        objs.append(defaultdict(lambda: "", json.loads(line)))

print("All entries read (%d)" % len(objs))

# Get all unique columns
keys = set()
for obj in objs:
    keys.update(obj.keys())

# Split categories into list
for obj in objs:
    # This would be easier, but no guarantee categories always split by ", "
    # and not just ","
    # obj["categories"] = obj["categories"].split(", ")
    cats = obj["categories"]
    # Might be null
    if cats:
        obj["categories"] = [s.strip() for s in cats.split(",")]
    else:
        obj["categories"] = []

# Filter just restaurants
objs = [o for o in objs if "Restaurants" in o["categories"]]

print("Number of restaurants: %d" % len(objs))

# Promote "attributes" fields to full-blown columns
for obj in objs:
    attrs = obj["attributes"]
    if not attrs:
        continue
    for k in attrs.keys():
        keys.add(k)
        obj[k] = attrs[k]
keys.remove("attributes") # Don't want in final CSV

with open("./business.csv", "w") as outfile:
    writer = csv.writer(outfile, delimiter=",")

    # Use keys as column headings
    writer.writerow([k for k in keys])

    for obj in objs:
        # For each object, iterate through the keys and get the value for each
        # If no value for that key, use default defined above
        # It's possible that this value might be an object instead of a single
        # string or number - in this case use json.dumps to convert to a string
        # (which shouldn't affect strings or numbers)
        row = [json.dumps(obj[k]) for k in keys]
        writer.writerow(row)

