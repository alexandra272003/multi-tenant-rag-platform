p = "/code/scripts/verify_postgres.py"
s = open(p).read()
old = 'up4.json()["status"] == "ready")'
new = '(await api.get(f"/documents/{up4.json()[\'id\']}", headers=B)).json()["status"] == "ready")'
assert old in s, "line not found"
open(p, "w").write(s.replace(old, new))
print("patched")
