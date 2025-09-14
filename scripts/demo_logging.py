import hashlib, json, os, time
os.makedirs("reports/artifacts", exist_ok=True)
cfg = {"run_at": time.time(), "params": {"universe":"NIFTY50","mode":"paper"}}
h = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
payload = {"run_id": h, "config_hash": h, "status":"demo"}
open("reports/artifacts/logging_demo.json","w").write(json.dumps(payload, indent=2))
print(payload)
