## Architecture Design


### 1. Optimiser (design / policy proposal mechanism)

Let the design space be either a static design space $\mathcal D$ or a policy space $\Pi$; denote it generically by $\mathcal X$.
Let $\mathcal S$ be the space of feedback summaries (scalar or vector), and let optimiser feedback live in $\mathbb R^k$.


The optimiser is an interactive system defined by the maps

$$
\textsf{ask} : \mathcal H_n \rightarrow \mathcal X,
\qquad
\textsf{tell} : (\mathcal H_n, x, \mathbf{s}) \rightarrow \mathcal H_{n+1},
\qquad \mathbf{s} \in \mathcal S,
$$

where the optimiser history is

$\mathcal H_n = {(x_j, \mathbf{s}_j)}_{j \le n}$ with $\mathbf{s}_j \in \mathbb R^k$

Here $\mathbf{s}$ denotes the aggregated feedback (scalar or vector) returned to the optimiser for a queried $x$, and the index $j$ denotes the optimiser interaction round (the $j$-th ask/tell pair). $\mathcal H_n$ is thus the history of all previous queries to the optimiser and their aggregated feedback.

The optimiser observes **only aggregated feedback**, $(x,\mathbf{s})$ pairs.
It has no access to trials, randomness, risk definitions, feasibility, certificates, or stopping logic.  



### 2. Experiment and simulator (per-trial semantics)

Fix a sample space $(\Omega, \mathcal F)$ and a time-indexed family of mappings $(\mathbb P_t)_{t\in\mathcal T}$, where:

* $\Omega$ is the set of all possible *worlds* (elementary random outcomes),
* $\mathcal F$ is a $\sigma$-algebra defining which subsets of $\Omega$ are measurable events,
* $\mathbb P_t$ is a probability measure assigning likelihoods to events in $\mathcal F$ at evaluation time (or context) $t$.

An experiment defines a random variable

$$
Z : \mathcal X \times \Omega \times \mathcal T \rightarrow \mathcal Z,
\qquad
(x,\omega,t) \mapsto Z(x,\omega,t),
$$

where $\mathcal Z$ may be scalar-, vector-, or trajectory-valued.

Optionally, it defines a validity indicator

$$
v : \mathcal X \times \Omega \times \mathcal T \rightarrow {0,1}.
$$
which declares whether a trial outcome is admissible.

A simulator is a concrete procedure that, given evaluation time/context $t$, samples $\omega \sim \mathbb P_t$ (via an RNG sequence) and evaluates the sampled $Z(x,\omega,t)$. It encodes system dynamics or physics, not decision logic.
The experiment defines what a trial is; orchestration decides how contexts $t$ are scheduled and how data are aggregated.

### 3. Static designs vs. online policies

* **Static design**: $x = d \in \mathcal D$ is a fixed parameter vector.
* **Online policy**: $x = \pi \in \Pi$ is a measurable map
  $$
  \pi : h_i \rightarrow a_i,
  $$
  inducing closed-loop dynamics
  $$
  x_{i+1} = f(x_i, a_i, \varepsilon_i),
  \qquad
  a_i = \pi(h_i),
  \qquad
  \omega \sim \mathbb P_t.
  $$
where $i$ is within-episode time and $\varepsilon_i(\omega)$ denotes the exogenous noise at time $i$ (as a function of the world $\omega$). The evaluation-time/context index $t$ and the law $\mathbb P_t$ are defined in Section 2.
In both cases, a trial is $Z(x,\omega,t)$ with context $t$ scheduled by orchestration.
Adaptivity is fully contained inside $Z$; it does not affect the architecture boundary.



### 4. Orchestrator (sampling, aggregation, certificates)

For each design or policy $x \in \mathcal X$ and each evaluation time/context $t \in \mathcal T$, the orchestrator draws **design- and context-local worlds**

$$
\omega_{x,t,1}, \omega_{x,t,2}, \dots \sim \mathbb P_t,
$$

with the invariant that this sequence depends only on $(x,t)$, not on exploration order.

Observed trials are

$$
Z_{x,t,k} := Z(x,\omega_{x,t,k},t), \quad k=1,\dots,n(x,t).
$$

Here $k$ indexes the trial number within a fixed $(x,t)$ (i.e. repeated Monte Carlo evaluations at the same design and context).

The orchestrator defines vector-valued risk aggregation

$$
\mathbf{s}(x,t) := \widehat{\boldsymbol{\rho}}\!\left(Z_{x,t,1:n(x,t)}\right) \in \mathbb R^k,
\qquad
\widehat{\boldsymbol{\rho}} := (\widehat{\rho}_1,\dots,\widehat{\rho}_k),
$$

where each component may represent expectation, probability, quantile, CVaR, or any pathwise functional.

The orchestrator:

* schedules contexts $t$,
* allocates trials $n(x,t)$,
* adapts sampling,
* decides stopping,
* meta-optimises,
* invokes certificates
  $$
  C\left(Z_{x,t,1:n(x,t)}\right) \in \mathcal C,
  $$
  such as confidence bounds or feasibility guarantees.

Only aggregated feedback $\mathbf{s}$ is used to update the optimiser.


### 5. Certificates (orchestrator-owned inference)

A certificate is a function of aggregated trial data (typically at a fixed evaluation time/context $t$),

$$
C\left(Z_{x,t,1:n(x,t)}\right) \in \mathcal C,
$$

that provides a decision or guarantee.
For example, for confidence level $\delta \in (0,1)$, an upper confidence bound $\mathrm{UCB}_\delta(x,t)$ satisfies

$$
\mathbb P_t\left( \rho(x,t) \le \mathrm{UCB}_\delta(x,t) \right) \ge 1 - \delta.
$$

Certificates consume trial aggregates and produce guarantees (feasibility, bounds, stopping decisions); they never define the random variable $Z$ itself.


---

### Architectural invariant (fully general)

Experiments define per-trial semantics (random variables) $Z(x,\omega,t)$ (and optionally validity $v(x,\omega,t)$).
Orchestrators decide how contexts $t$ are scheduled and how collections of trials are sampled, aggregated, certified, and turned into decisions.
Optimisers propose designs or policies using only aggregated feedback summaries (scalar or vector).

This invariant holds for scalar or vector designs, static or online policies, scalar or vector objectives, and time-dependent simulations.
