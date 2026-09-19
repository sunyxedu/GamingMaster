# Jev context injection: live interface probe

Source commit: `7201cf9`. This is an interface check, not a gameplay benchmark.

The tier-2 task builder's complete output, including the bundled Jev guide,
was given to real `codex:gpt-6-astra` at high effort through `ctx.think`.
A separate synthetic fixture asked it to create three independent typed
questions. Its unmodified state/questions were submitted in one `ctx.decide`
request to real Jev. `trace.jsonl.gz` records both requests and responses,
including actual latency, backend version and usage.

- Astra: one call; generated valid Choice, Score and Noul questions and named
  the correct local `.value`, `.probabilities` and `.confidence` fields.
- Jev: one call; reported contradicted preconditions (probability 0.96),
  progress 0.0 on the 0–2 scale, completion probability 0.02. Noul confidence
  was `None`, as documented.
- The fixture describes an unplugged socket and a dark beacon, so these
  judgments are consistent with the supplied evidence.

The probe checker initially assumed `read_fields` would be a flat list;
Astra returned a valid per-question mapping. The checker was corrected to
flatten that metadata. The stored Astra output and actual Jev questions were
not edited, and Astra was not called again.

Verification also ran 17 targeted tests, including the offline tier-2 loop's
actual proposer prompt and archive/manifest capture, the guide's executable
example through the context/HTTP adapter, and the existing live Jev test.
The Markdown resource was verified inside a wheel built from a committed
snapshot of the metaharness package.

Reproduce the targeted checks from the repository root:

```bash
uv run pytest -q tests/metaharness/test_loop_tier2.py tests/metaharness/test_jev_usage.py tests/metaharness/test_models.py
```

New searches automatically receive this context. Existing scored archives
must not be resumed with changed task text. Previous gameplay results predate
this prompt change; no claim of improved gameplay follows from this probe.
