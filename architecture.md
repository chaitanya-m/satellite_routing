## Architecture Design


### 1. Optimiser (design / policy proposal mechanism)

Let the design space be either a static design space $\mathcal D$ or a policy space $\Pi$; denote it generically by $\mathcal X$.
Let $\mathcal S$ be the space of feedback summaries (scalar or vector), and let optimiser feedback live in $\mathbb R^k$,.


The optimiser is an interactive system defined by the maps

$$
\textsf{ask} : \mathcal H_t \rightarrow \mathcal X,
\qquad
\textsf{tell} : (\mathcal H_t, x, \mathbf{s}) \rightarrow \mathcal H_{t+1},
\qquad s \in \mathcal S,
$$

where the optimiser history is

$\mathcal H_t = {(x_j, \mathbf{s}_j)}_{j \le t}$ with $\mathbf{s}_j \in \mathbb R^k$

Here $\mathbf{s}$ denotes the aggregated feedback vector returned to the optimiser for a queried $x$, and the index $j$ denotes the optimiser interaction round (the $j$-th ask/tell pair). $\mathcal H_t$ is thus the history of all previous queries to the optimiser and their aggregated feedback.

The optimiser observes **only aggregated vector feedback**, $(x,s)$ pairs.
It has no access to trials, randomness, risk definitions, feasibility, certificates, or stopping logic.  



### 2. Experiment and simulator (per-trial semantics)

Fix a probability triple $(\Omega, \mathcal F, \mathbb P)$, where:

* $\Omega$ is the set of all possible *worlds* (elementary random outcomes),
* $\mathcal F$ is a $\sigma$-algebra defining which subsets of $\Omega$ are measurable events,
* $\mathbb P$ is a probability measure assigning likelihoods to events in $\mathcal F$.

An experiment defines a random variable

$$
Z : \mathcal X \times \Omega \rightarrow \mathcal Z,
\qquad
(x,\omega) \mapsto Z(x,\omega),
$$

where $\mathcal Z$ may be scalar-, vector-, or trajectory-valued.

Optionally, it defines a validity indicator

$$
v : \mathcal X \times \Omega \rightarrow {0,1}.
$$
which declares whether a trial outcome is admissible.

A simulator is a concrete procedure that samples $\omega \sim \mathbb P$ (via an RNG sequence) and evaluates the sampled $Z(x,\omega)$. It encodes system dynamics or physics, not decision logic.
The experiment defines what a trial is, nothing more.

### 3. Static designs vs. online policies

* **Static design**: $x = d \in \mathcal D$ is a fixed parameter vector.
* **Online policy**: $x = \pi \in \Pi$ is a measurable map
  $$
  \pi : h_t \rightarrow a_t,
  $$
  inducing closed-loop dynamics
  $$
  x_{t+1} = f(x_t, a_t, \varepsilon_t),
  \qquad
  a_t = \pi(h_t),
  \qquad
  \varepsilon_t(\omega) \sim \mathbb P.
  $$
where $\varepsilon$ is noise.
In both cases, a trial is simply $Z(x,\omega)$.
Adaptivity is fully contained inside $Z$; it does not affect the architecture.



### 4. Orchestrator (sampling, aggregation, certificates)

For each design or policy $x \in \mathcal X$, the orchestrator draws **design-local worlds**

$$
\omega_{x,1}, \omega_{x,2}, \dots \sim \mathbb P,
$$

with the invariant that this sequence depends only on $x$, not on exploration order.

Observed trials are

$$
Z_{x,i} := Z(x,\omega_{x,i}), \quad i=1,\dots,n(x).
$$

The orchestrator defines vector-valued risk aggregation

$$
\mathbf{s}(x) := \widehat{\boldsymbol{\rho}}\!\left(Z_{x,1:n(x)}\right) \in \mathbb R^k,
\qquad
\widehat{\boldsymbol{\rho}} := (\widehat{\rho}_1,\dots,\widehat{\rho}_k),
$$

where each component may represent expectation, probability, quantile, CVaR, or any pathwise functional.

The orchestrator:

* allocates trials $n(x)$,
* adapts sampling,
* decides stopping,
* meta-optimises,
* invokes certificates
  $$
  C\left(Z_{x,1:n}\right) \in \mathcal C,
  $$
  such as confidence bounds or feasibility guarantees.

Only $\widehat{\boldsymbol{\rho}}(x)$ is used to update the optimiser.


### 5. Certificates (orchestrator-owned inference)

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



### Architectural invariant (fully general)

Experiments define random variables $Z(x,\omega)$.
Orchestrators decide how collections of trials are sampled, aggregated, certified, and make decisions.
Optimisers propose designs or policies using only vector-valued aggregated feedback.

This invariant holds for scalar or vector designs, static or online policies, scalar or vector objectives, and time-dependent simulations.

