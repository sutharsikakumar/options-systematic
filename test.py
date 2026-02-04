import os, hashlib

path = "options.parquet"
print("size bytes:", os.path.getsize(path))

h = hashlib.md5()
with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
        h.update(chunk)
print("md5:", h.hexdigest())
