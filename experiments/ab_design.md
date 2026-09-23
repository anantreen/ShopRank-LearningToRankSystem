# Experiment Design (Not Executed)

This is a proposed online experiment, not a claim that real customers were exposed.

- **Control:** production BM25 plus the old ranker.
- **Treatment:** hybrid lexical/semantic retrieval plus LambdaMART.
- **Randomization unit:** customer (sticky assignment prevents cross-condition contamination).
- **Primary metric:** search click-through rate (CTR).
- **Secondary metrics:** add-to-cart rate, conversion rate, revenue per search session.
- **Guardrails:** p95 latency, zero-result rate, cancellations, bounce rate.
- **Hypotheses:** H0 says treatment CTR equals control CTR; H1 says treatment CTR is higher.

For baseline rate `p`, detectable absolute lift `delta`, two-sided type-I error `alpha=0.05`, and
power `1-beta=0.80`, a planning approximation per arm is:

`n = 2 * (z_(alpha/2) + z_beta)^2 * p * (1-p) / delta^2`

Example only: with `p=.10`, `delta=.005`, `z_(alpha/2)=1.96`, and `z_beta=.84`, this gives about
56,448 customers per arm. Recalculate using current production traffic and variance; validate the
approximation with an experimentation platform before launch.

