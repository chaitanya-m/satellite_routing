## Architecture Design


### 1. Optimiser (design / policy proposal mechanism)

Let the design space be either a static design space $\mathcal D$ or a policy space $\Pi$; denote it generically by $\mathcal X$.
Let $\mathcal S$ be the space of feedback summaries (scalar or vector), and let optimiser feedback live in $\mathbb R^k$.


The optimiser is an interactive system defined by the maps

$$
\textsf{ask} : \mathcal H_n \rightarrow \mathcal X,
\qquad
\textsf{tell} : (\mathcal H_n, \pi, \mathbf{s}) \rightarrow \mathcal H_{n+1},
$$

where the optimiser history is

$$
\mathcal H_n = {(\pi_j, \mathbf{s}_j)}_{j \le n}
$$ 
with 
$$
\mathbf{s}_j \in \mathbb R^k, \qquad \mathbf{s} \in \mathcal S
$$

Here $\mathbf{s}$ denotes the aggregated feedback (scalar or vector) returned to the optimiser for a queried $\pi$, and the index $j$ denotes the optimiser interaction round (the $j$-th ask/tell pair). $\mathcal H_n$ is thus the history of all previous queries to the optimiser and their aggregated feedback.

The optimiser observes **only aggregated feedback**, $(\pi,\mathbf{s})$ pairs.
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
(\pi,\omega,t) \mapsto Z(\pi,\omega,t),
$$

where $\mathcal Z$ may be scalar-, vector-, or trajectory-valued.

Optionally, it defines a validity indicator

$$
v : \mathcal X \times \Omega \times \mathcal T \rightarrow {0,1}.
$$

which declares whether a trial outcome is admissible.

A simulator is a concrete procedure that, given evaluation time/context $t$, samples $\omega \sim \mathbb P_t$ (via an RNG sequence) and evaluates the sampled $Z(\pi,\omega,t)$. It encodes system dynamics or physics, not decision logic.
The experiment defines what a trial is; orchestration decides how contexts $t$ are scheduled and how data are aggregated.

### 3. Static designs vs. online policies

* **Static design**: $\pi = d \in \mathcal D$ is a fixed parameter vector (a degenerate, non-adaptive policy).
* **Online policy**: $\pi \in \Pi$ is a measurable map $\pi : h_i \rightarrow a_i$, inducing closed-loop dynamics $x_{i+1} = f(x_i, a_i, \varepsilon_i)$ with $a_i = \pi(h_i)$ and $\omega \sim \mathbb P_t$.

where $i$ is within-episode time and $\varepsilon_i(\omega)$ denotes the exogenous noise at time $i$ (as a function of the world $\omega$). The evaluation-time/context index $t$ and the law $\mathbb P_t$ are defined in Section 2.
In both cases, a trial is $Z(\pi,\omega,t)$ with context $t$ scheduled by orchestration.
Adaptivity is fully contained inside $Z$; it does not affect the architecture boundary.



### 4. Orchestrator (sampling, aggregation, certificates)

For each design or policy $\pi \in \mathcal X$ and each evaluation time/context $t \in \mathcal T$, the orchestrator draws **design- and context-local worlds**

$$
\omega_{\pi,t,1}, \omega_{\pi,t,2}, \dots \sim \mathbb P_t,
$$

with the invariant that this sequence depends only on $(\pi,t)$, not on exploration order.

Observed trials are

$$
Z_{\pi,t,k} := Z(\pi,\omega_{\pi,t,k},t), \quad k=1,\dots,n(\pi,t).
$$

Here $k$ indexes the trial number within a fixed $(\pi,t)$ (i.e. repeated Monte Carlo evaluations at the same design and context).

The orchestrator defines vector-valued risk aggregation

$$
\mathbf{s}(\pi,t) := \widehat{\boldsymbol{\rho}}\left(Z_{\pi,t,1:n(\pi,t)}\right) \in \mathbb R^k,
\qquad
\widehat{\boldsymbol{\rho}} := (\widehat{\rho}_1,\dots,\widehat{\rho}_k),
$$

where each component may represent expectation, probability, quantile, CVaR, or any pathwise functional.

The orchestrator:

* schedules contexts $t$,
* allocates trials $n(\pi,t)$,
* adapts sampling,
* decides stopping,
* meta-optimises,
* invokes certificates
  $C\left(Z_{\pi,t,1:n(\pi,t)}\right) \in \mathcal C$
  such as confidence bounds or feasibility guarantees.

Only aggregated feedback $\mathbf{s}$ is used to update the optimiser.


### 5. Certificates (orchestrator-owned inference)

A certificate is a function of aggregated trial data (typically at a fixed evaluation time/context $t$),

$$
C\left(Z_{\pi,t,1:n(\pi,t)}\right) \in \mathcal C,
$$

that provides a decision or guarantee.
For example, for confidence level $\delta \in (0,1)$, an upper confidence bound $\mathrm{UCB}_\delta(\pi,t)$ satisfies

$$
\mathbb P_t\left( \rho(\pi,t) \le \mathrm{UCB}_\delta(\pi,t) \right) \ge 1 - \delta.
$$

Certificates consume trial aggregates and produce guarantees (feasibility, bounds, stopping decisions); they never define the random variable $Z$ itself.


---

### Architectural invariant (fully general)

Experiments define per-trial semantics (random variables) $Z(\pi,\omega,t)$ and optionally a validity indicator $v(\pi,\omega,t)$.
Orchestrators decide how contexts $t$ are scheduled and how collections of trials are sampled, aggregated, certified, and turned into decisions.
Optimisers propose designs or policies using only aggregated feedback summaries (scalar or vector).

This invariant holds for scalar or vector designs, static or online policies, scalar or vector objectives, and time-dependent simulations.
