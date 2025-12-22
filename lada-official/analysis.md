# Psychometric Factor Analysis Report: LegalBench Dataset

**Methodology:** Latent Ability Dirichlet Allocation (LADA)
**Date:** December 2025

---

## 1. Analysis Overview

This report details the psychometric decomposition of the LegalBench dataset (subset $N=1,997$ items). Using a 2-dimensional LADA model, we analyzed item discrimination vectors ($w$) to identify orthogonal cognitive skills.

The analysis followed a **mixed-methods approach**:

1.  **Quantitative Stratification:** Filtering items by high discrimination loadings ($\ge 0.7$) to isolate defining characteristics.
2.  **Qualitative Semantic Review:** Deep reading of "Anchor Items" (loadings $\ge 0.95$) to decode the cognitive operations required for solution.
3.  **Task Distribution Analysis:** Examining how specific legal tasks (e.g., `international_citizenship` vs. `corporate_lobbying`) distribute across factors.

---

## 2. Factor 1: Exclusionary Determination (Negative Evidence Detection)

### 2.1 Quantitative Profile

- **Dominant Answer Pattern:** Strongly correlated with the answer **"No"**.
  - Average loading for "No" items: **0.74**
  - Average loading for "Yes" items: **0.10**
- **Input Complexity:** Inputs are significantly shorter on average (**~572 characters**).
- **Structural Signal:** **81.4%** of high-loading questions contain conditional phrasing (e.g., _"if so, under which conditions?"_).

### 2.2 Task Composition

This factor is heavily saturated by the `international_citizenship` task (**90%** of high loaders), specifically items where the correct legal determination is that a country does **not** have a specific citizenship provision.

### 2.3 Cognitive Process Analysis

Examination of the top anchor items reveals that this dimension measures a specific form of **Verification Search**.

- **Example Anchor:** _"Consider the country of Dominica. Does the country provide for involuntary loss of citizenship by a person one or both of whose parents lose citizenship...?"_ -> **Answer: No.**

To solve this correctly, the model must:

1.  Ingest the query and the knowledge base (statutes/rules).
2.  Perform an **Exhaustive Search** of the legal context.
3.  Match the query conditions against _every_ available provision.
4.  Conclude that **no match exists**.
5.  **Inhibit Hallucination:** The model must resist the urge to generate a plausible-sounding legal mechanism where none exists.

### 2.4 Factor Definition

**Latent Skill:** **Negative Evidence Detection (Absence Verification)**.
This is the capacity to validly conclude that a phenomenon or rule is absent from the context. It represents a "Skeptic" or "Gatekeeper" function.

---

## 3. Factor 2: Inclusionary Determination (Affirmative Rule Extraction)

### 3.1 Quantitative Profile

- **Dominant Answer Pattern:** Strongly correlated with the answer **"Yes"** (and specific categorical labels).
  - Average loading for "Yes" items: **0.90**
  - Average loading for "No" items: **0.26**
- **Input Complexity:** Inputs are massive, averaging **~2,094 characters** (nearly 4x longer than Factor 1).
- **Structural Signal:** Lower conditional frequency (39%), higher reliance on direct text parsing.

### 3.2 Task Composition

This factor is much more diverse, indicating a generalized skill rather than a task-specific artifact. High loaders include:

- `corporate_lobbying` (Interpreting business descriptions to find lobbying intent).
- `abercrombie` (Classifying trademarks into affirmative categories like "fanciful").
- `proa` (Finding Private Rights of Action in statutes).
- `function_of_decision_section` (Identifying section types).

### 3.3 Cognitive Process Analysis

Examination of anchor items reveals this dimension measures **Targeted Information Extraction**.

- **Example Anchor:** _"Official title of bill: To provide for the coverage of medically necessary food... [Long Description]... Is this relevant to the company?"_ -> **Answer: Yes.**

To solve this correctly, the model must:

1.  Ingest a large, noisy context window.
2.  Identify a specific signal (keyword, semantic concept, or rule pattern).
3.  **Map the Signal** to the query.
4.  Terminate the search upon finding positive evidence.

### 3.4 Factor Definition

**Latent Skill:** **Affirmative Rule Comprehension (Presence Verification)**.
This is the capacity to parse complex text and identify that a specific condition _is_ met. It represents a "Hunter" or "Extractor" function.

---

## 4. Synthesis: The Cognitive Asymmetry

The most critical insight from this factor analysis is that **"Yes" and "No" are not inverses** in the context of LLM reasoning. They represent distinct cognitive workloads.

| Feature            | Factor 1 (Absence/F1)                                                   | Factor 2 (Presence/F2)                                  |
| :----------------- | :---------------------------------------------------------------------- | :------------------------------------------------------ |
| **Search Scope**   | **Exhaustive:** Must check _all_ possibilities to confirm zero matches. | **Terminating:** Stops as soon as _one_ match is found. |
| **Failure Mode**   | **False Positive:** Hallucinating a rule that doesn't exist.            | **False Negative:** Missing a rule buried in long text. |
| **Cognitive Bias** | Requires **Conservatism** (defaulting to Null Hypothesis).              | Requires **Responsiveness** (pattern matching).         |

The LADA model successfully disentangled these two skills because they are computationally distinct. A model can be excellent at finding needles in haystacks (High F2) but terrible at knowing when the needle _isn't_ there (Low F1).

---

## 5. Model Behavior Analysis (The Leaderboard)

Applying these factor definitions to the model $\theta$ (ability) scores reveals a significant trend regarding **Instruction Tuning**.

### 5.1 Base Models: The Skeptics

- **Models:** `meta/llama-3-70b`, `meta/llama-2-13b`
- **Profile:** Very High F1 ($\theta > 1.5$), Low/Negative F2.
- **Interpretation:** Base models are trained to predict the next token based on likelihood. Without RLHF (Reinforcement Learning from Human Feedback) pressuring them to be "helpful," they are unbiased toward the null hypothesis. They are excellent at **Exclusionary Determination** because they don't feel compelled to invent answers.

### 5.2 Instruct Models: The Pleasers

- **Models:** `llama-3.1-405b-instruct`, `llama-3.3-70b-instruct`
- **Profile:** Very High F2 ($\theta > 1.7$), Negative F1 ($\theta < -0.5$).
- **Interpretation:** Instruction tuning aligns models to follow user intent and provide helpful answers. In legal contexts, this manifests as a bias toward finding rules. They excel at **Affirmative Extraction** (F2) because they are tuned to retrieve information, but this destroys their ability to perform **Negative Evidence Detection** (F1)—they tend to hallucinate "Yes" answers to be helpful.

### 5.3 Conclusion

The factor analysis demonstrates that **Alignment is a Trade-off**. Enhancing a model's ability to extract and comply (F2) appears to actively degrade its ability to rigorously verify absence (F1). For high-stakes legal tasks, this suggests that **Base models may be safer** for compliance checking (where false positives are dangerous), while **Instruct models are superior** for discovery and research (where missing information is the primary risk).
