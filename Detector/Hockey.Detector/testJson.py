import json
from types import SimpleNamespace

with open("zones.json") as f:
    d = json.loads(f.read(), object_hook=lambda d: SimpleNamespace(**d))
    for i in d:
        print(i.x)
        print(i.y)
