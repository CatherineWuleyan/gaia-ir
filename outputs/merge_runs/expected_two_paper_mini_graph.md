# Pipeline 8.0 expected two-paper mini graph

## Scope

Small subdomain: pruning / parameter compression / storage and accuracy retention.

## Paper nodes

- **Paper A — QMPNN / quaternion GNN compression**: replaces standard message-passing functions with quaternion counterparts; reports substantially fewer parameters with comparable or improved task performance; selected subnetworks can match the baseline.
- **Paper B — pruning and resource utilization**: pruning generally reduces storage and, after fine-tuning, often retains accuracy; gains depend on pruning method, task, and implementation; the tested scope does not establish the Lottery Ticket Hypothesis universally.

## Expected cross-paper edges

1. `A:small subnetworks match baseline performance` ↔ `B:pruning can retain accuracy after fine-tuning` — **related / conditional support**.
2. `A:parameter compression` ↔ `B:storage compression` — **related but not equivalent**; parameter count reduction does not by itself prove end-to-end memory or inference reduction.
3. `A:winning-ticket-style result` ↔ `B:LTH not supported in tested scope` — **scope-limited tension**, not a direct contradiction; the claims use different methods/tasks/evidence scopes.

## Expected synthesis

Compression effects are conditional on method, task, and implementation. Parameter or storage compression should not be promoted to universal inference-memory gains or a universal Lottery Ticket claim.
