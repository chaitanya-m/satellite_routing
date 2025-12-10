## Interface design (DesignProblem / DesignOptimiser)

The optimisation layer is minimal and domain-agnostic:

* **DesignProblem**
  * `sample_one_design() -> Any`: produce a single candidate design (type is up to the problem; could be a float, a dataclass, etc.).
  * `evaluate(design: Any) -> float`: return a scalar score; higher is better (negate losses/costs if needed).
  * Problems may optionally expose extra utilities (e.g., `perturb`, `encode_vector`/`decode_vector`) that specific optimisers can exploit, but they’re not required by the interface.

* **DesignOptimiser**
  * `propose_candidate(problem) -> Any`: ask the optimiser for the next design to evaluate.
  * `record_result(design, score) -> None`: feed back the evaluation of a design so the optimiser can update its internal model.
  * `current_best() -> Optional[(design, score)]`: return the best design/score seen so far.
  * `supports(problem) -> bool`: structural check to decide if the optimiser can operate on a given problem (e.g., does the problem implement the utilities the optimiser needs).

* **Drivers**
  * `run_optimisation` performs a fixed budget of single-candidate evaluations using the above methods.
  * `run_optimisation_parallel` overlaps evaluations via an executor while still interacting with optimisers one design at a time.

 To add a new optimiser (local or from an external library), implement `DesignOptimiser` for it and fill in `supports(problem)` based on the problem methods it requires. To add a new problem, implement `DesignProblem` and any optional utilities that make sense for that domain; `compatible_optimisers(problem, candidates)` will then automatically select which optimisers can run on it.



## Nevergrad adapter loop (how it works)

At a high level, with the Nevergrad adapter in this codebase, the loop looks like this:

1. **Your problem defines the design space and evaluation**  
   - `sample_one_design()` gives a candidate design (here, a float `x`).  
   - `evaluate(design)` returns a scalar score (here, `-(x-3)^2`).

2. **The Nevergrad optimiser maintains an internal model**  
   - In `NevergradScalarOptimiser`, we initialise a Nevergrad optimiser (`OnePlusOne`) over a 1D scalar parameter with given bounds.

3. **Proposal step (`propose_candidate`)**  
   - We call `optimiser.ask()` on the Nevergrad object.  
   - It returns a recommendation: a proposed scalar value based on its current step/perturbation distribution.  
   - This is the candidate design you evaluate next.

4. **Evaluation (your code)**  
   - You pass that candidate to `problem.evaluate(design)`.  
   - You get back a score.

5. **Feedback step (`record_result`)**  
   - We call `optimiser.tell(design, -score)` on the Nevergrad object.  
   - Nevergrad minimises by default, so we negate your score to turn maximisation into minimisation.  
   - Internally, Nevergrad uses that feedback to adjust its step size / distribution, biasing future proposals toward better regions.

6. **Repeat until budget is exhausted**  
   - Each `propose_candidate` gives you a new candidate informed by past feedback.  
   - Each `record_result` feeds back the outcome.  
   - `current_best` tracks the best (design, score) seen so far.

So it’s interactive: Nevergrad proposes candidates based on its adaptive model; your `evaluate` provides the objective values; Nevergrad updates its model accordingly. Unlike random search, which samples blindly, Nevergrad uses the feedback from your evaluations to guide the next steps.
