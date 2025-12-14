

## Mathematical formulation of the architecture

### 1. Optimiser (design proposal mechanism)

Let $\mathcal D$ be the design space and $\mathcal S$ the space of feedback summaries (scalar or vector).
The optimiser is an interactive system defined by the maps

$$
\textsf{ask}: \mathcal H_t \rightarrow \mathcal D,
\qquad
\textsf{tell}: (\mathcal H_t, d, s) \rightarrow \mathcal H_{t+1}, \qquad s \in \mathcal S,
$$

where the optimiser history is
$\mathcal H_t = {(d_j, s_j)}_{j \le t}$ with $s_j \in \mathcal S$.

The optimiser observes **only** $(d,s)$ pairs. It has no access to trials, randomness, risk definitions, feasibility, or stopping rules.

---

### 2. Experiment and simulator (per-trial semantics)

Fix a probability triple $(\Omega,\mathcal F,\mathbb P)$, where:

* $\Omega$ is the set of all possible *worlds* (elementary random outcomes),
* $\mathcal F$ is a $\sigma$-algebra defining which subsets of $\Omega$ are measurable events,
* $\mathbb P$ is a probability measure assigning likelihoods to events in $\mathcal F$.

An experiment defines a random variable

$$
Z : \mathcal D \times \Omega \rightarrow \mathcal Z,
\qquad
(d,\omega) \mapsto Z(d,\omega),
$$

where $\mathcal Z$ may be scalar-, vector-, or trajectory-valued.
Optionally, it defines a validity indicator

$$
v : \mathcal D \times \Omega \rightarrow {0,1},
$$

which declares whether a trial outcome is admissible.

A simulator is a concrete procedure that samples $\omega \sim \mathbb P$ (via an RNG tape) and evaluates $Z(d,\omega)$. It encodes system dynamics or physics, not decision logic.

---

### 3. Orchestrator (runs experiments, aggregates risk, performs meta-optimisation and decision-making)

For each design $d$, the orchestrator selects a number of trials $n(d)$ and draws **design-local** worlds

$$
\omega_{d,1}, \dots, \omega_{d,n(d)} \in \Omega.
$$

This produces observed trial outcomes

$$
Z_{d,i} := Z(d,\omega_{d,i}), \qquad i = 1,\dots,n(d).
$$

The orchestrator aggregates evidence via a risk estimator

$$
s(d) := \widehat{\rho}\left(Z_{d,1:n(d)}\right) \in \mathcal S,
$$

and reports only this scalar score to the optimiser via `tell`.
All policies for trial allocation, adaptive sampling, early stopping, and re-evaluation are owned by the orchestrator.

---

### 4. Certificates (orchestrator-owned inference)

A certificate is a function of aggregated trial data,

$$
C\left(Z_{d,1:n}\right) \in \mathcal C,
$$

that provides a decision or guarantee.
For example, an upper confidence bound $\mathrm{UCB}_\delta(d)$ satisfies

$$
\mathbb P\left( \rho(d) \le \mathrm{UCB}_\delta(d) \right) \ge 1 - \delta.
$$

Certificates consume trial aggregates and produce guarantees (feasibility, bounds, stopping decisions); they never define the random variable $Z$ itself.

---

### Architectural invariant (formal)

Experiments define $Z(d,\omega)$,
orchestrators choose how ${Z_{d,i}}$ are sampled and interpreted,
optimisers operate only on summaries $s(d) \in \mathcal S$ (scalar or vector).

This invariant is preserved under time-dependent, vector-valued, and continuous-design extensions.
