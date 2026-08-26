# Hot40i local-client demo

The A03 client uses the OpenAI Python SDK against the laptop gateway while model compute stays
on the phone:

```powershell
python -m host.gateway.app
python -m host.local_client_demo --output benchmarks/service/end_to_end.json
```

The pinned demo environment is recorded in `host/local_client_demo.requirements.txt`. The client
configuration is `base_url=http://127.0.0.1:18081/v1`, model `openai/gpt-oss-20b`, a non-secret
local API-key placeholder, no retries, and no request timeout.

Each scenario requests one deterministic argmax token. A one-token cutoff is represented as an
OpenAI Responses `incomplete` result with reason `max_output_tokens`; it is not presented as a
finished Harmony message. Response headers identify the emitted token and a bounded gateway trace
record. The demo resolves that trace and verifies token identity, prompt-token count, decoder
timing, and device-service request counters end to end.
