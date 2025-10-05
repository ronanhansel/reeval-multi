# Checkpoint 5th October 2025

- Performed MIRT fitting on each scenario from HELM.
- Converted latent factors embedding into relative statistical measures using EAP.
- Performed oblique rotation analysis and selected `geomin_obl` as the most interpretable rotation.

## MIRT Fitting results

Detailed in [mirt-official/calibration.ipynb](mirt-official/calibration.ipynb) section model fit assessment, we see Rasch models can sometimes match MIRT performance on homogeneous datasets e.g. GSM, MATH, but not on heterogeneous datasets e.g. LEGALBENCH. Statistical significance tests are performed in [mirt-official/evaluation.ipynb](mirt-official/evaluation.ipynb) which shows MIRT 2-factor significantly outperforms Rasch models on LEGALBENCH, with 6-factor (best-fit AUC) only marginally better than 2-factor at a risk of complexity and interpretability.

Latent factors can't be used directly because they are only the product of the fitting procedure. Converting them into relative statistical measures using EAP (expected a posteriori) yields a Rasch-like ability estimate per individual.

## Rotation results

Performed all oblique rotations available in `factor_analyzer` package, and selected `geomin_obl` as the most interpretable rotation. Based on [mirt-official/rotational_eval.ipynb](mirt-official/rotational_eval.ipynb), to compare rotations, we look at:

- Top loaders from each ends of each factor.
- Their corresponding scenario categories and answers.
- Characteristics metrics: `question_length`, `average_word_length`, `burstiness`, `perplexity`, `flesch_kincaid_grade`, `sentiment_compound`.

Using oblique rotation, we get mostly similar results.

- Factor 1 strongly reflects affirmative biased questions - (e.g. "Is X true?") vs. negative biased questions + (e.g. "Is X false?")

- Factor 2 is better aligned with `Contextual Scale` - Semantic Precision, Pattern Recognition & Rule Application, High-Fidelity Textual Grounding + Abstract Reasoning, Working Memory & Attention Management, Integration with Pre-existing Knowledge.

### External Validation F1

Although Affirmative Bias is an oversimplification of this Axis, but it's the most easy to find further analyses can yield more meaning. But based on the top loaders, the structure is almost symmetrical, the question categories are balanced, only difference is the number of Yes/No answers. No focused more on the positive end of Factor 1, and Yes focused more on the negative end of Factor 1.

### External Validation F2

Based on [mirt-official/rotational_eval.ipynb](mirt-official/rotational_eval.ipynb) F2 is more difficult to separate between the two ends. Their scores on each metric are very similar, not as polarised as F1. Performing additional NLP analyses in [mirt-official/validation_neg_f2.ipynb](mirt-official/validation_neg_f2.ipynb) yields more meaningful correlation based on

```bash
avg_dep_depth   0.2811
n_entities      0.4070
entity_types    0.4357
n_pronouns      0.4652
n_sentences     0.4662
n_tokens        0.4962
f2_loading      1.0000
```

Further analysis on top loaders' categories, we see negative end contains more in-context reasoning (e.g. "Based on the passage, is X true?") vs. positive end contains more out-of-context knowledge (e.g. Asking about a field of a related company). To verify this, we need external analysis into specialised datasets.

Using `SQuAD`, we performed tests on meta models present in `legalbench`. Collecting responses from those models and aggregated F1 scores (token-based multiple-correct overlapses score) as a measure of a model's aptitude in multi-hop reasoning and in-context analysis. However, the raw results from SQuAD do not yield actual correlation with negative end of F2 as they are still confounded with general ability, i.e. models with higher overall ability also tend to do better in SQuAD, not necessarily needing intended abilities to do well. In other words, general ability can _compensate_ for the possible lack of in-context reasoning.

To mitigate this, I hypothesised that we would need a general aptitude score to offset the raw SQuAD results, aligning with the bi-factor model where `Observed Specific Skill = General + Skill Residual + Error`. Using well-established datasets that intend to measure multidisciplinary language reasoning capability, `mmlu`, as a proxy for general ability, I computed the residuals of SQuAD F1 after regressing out MMLU scores. The residuals should represent the part of SQuAD performance that is not explained by general ability, and thus more likely to reflect in-context reasoning skills. The result shows a significant positive correlation (r=0.41) between the residuals and the negative end of F2, supporting the hypothesis that this factor captures in-context reasoning ability.

To verify this further, I performed the same procedure on another multidisciplinary dataset `commonsense` and other homogeneous datasets `math`, `gsm`, with the latter likely include more language-related reasoning. The results are consistent, with `commonsense` showing a similar positive correlation (r=0.42) with the negative end of F2, while `math` and `gsm` show weaker correlations (r=0.11 and r=0.18 respectively). This pattern suggests that F2 indeed captures a specific ability related to in-context reasoning, as it correlates more strongly with the residual datasets that require such skills.

I also fitted another LOESS model to test for non-linearity but the results are consistent. A summary table can be found in [mirt-official/validation_neg_f2.ipynb](mirt-official/validation_neg_f2.ipynb).

The moderate correlation is not neccessarily a bad sign if my understanding is correct, with similar psychological surveys suggests variations in the residual correlations [here](https://www.frontiersin.org/articles/10.3389/fpsyg.2020.01237/full#B64-ijpsy-11-01237) can sometimes be smaller. The correlation is not expected to be very high because specific factors in bifactor or MIRT frameworks are designed to capture residual, domain-specific variance after accounting for the general ability factor.

## Impact

The fitted MIRT can measure the loading for each item, reflecting the degree to which each item measures each latent trait. This information can be used to design more targeted assessments that focus on specific abilities, rather than relying solely on overall scores. Also, by understanding the multidimensional structure of abilities, we can better interpret model performance across different types of tasks and scenarios. Ultilising compensatory properties of MIRT, we can identify models that may excel in certain areas while compensating for weaknesses in others, leading to a more nuanced understanding of model capabilities. A model can perform well on a specific skill even if it has lower overall ability, as long as it compensates with strengths in other areas.

Since we still haven't accounted for a truly bi-factor model, the latent dimensions of MIRT can be inferred as the _tendencies_ of specific skills, rather than the aptitude of skills themselves. For example, a model with high loading on Factor 1 (Affirmative Bias) may not necessarily be good at all tasks requiring affirmative reasoning, but it indicates a tendency to perform better on such tasks compared to models with lower loadings on this factor if they have similar general aptitude (maybe from aggregated correct answer of IRT theta score). This nuanced understanding can help in selecting and deploying models for specific applications where certain abilities are more critical.
