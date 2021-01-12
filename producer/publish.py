import time

import msgpack
import requests

message = msgpack.packb({"date": time.strftime('%H:%M:%S'), "path": "/data/56-00.mp4"}, use_bin_type=True)
resp = requests.post("http://nsqd:4151/pub", params={"topic": "wildflower", "binary": "true", "channel": "pose-detect"}, data=message)
print(resp)
print(resp.text)



for c in range(200):
    resp = requests.post("http://nsqd:4151/pub", params={"topic": "wildflower"}, data=message)
    print(resp)
    print(resp.text)
    time.sleep(1)
