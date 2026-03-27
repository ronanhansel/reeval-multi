# Pre/Post Stability Diagnosis

## Configuration
- Pre-revision setting: `max`
- Column alignment: `intersection`
- Repeated samples: 50
- Seed: 42
- Post binarization: `post_binary = where(isnan(post_beta), nan, (post_beta > 0.5).astype(float))`

## Matrix Summary
### pre_full_raw
- rows: 143
- cols: 1176
- observed_fraction: 0.168153
- overall_mean: 0.396849
- avg_item_variance: 0.129674
- avg_agent_variance: 0.133906
- zero_variance_item_fraction: 0.059524
### post_beta
- rows: 32
- cols: 1176
- observed_fraction: 0.245509
- overall_mean: 0.244952
- avg_item_variance: 0.017152
- avg_agent_variance: 0.083179
- zero_variance_item_fraction: 0.093537
### post_binary
- rows: 32
- cols: 1176
- observed_fraction: 0.245509
- overall_mean: 0.198398
- avg_item_variance: 0.048591
- avg_agent_variance: 0.111170
- zero_variance_item_fraction: 0.743197
- mean_item_entropy: 0.185365
### pre_full_binary_sensitivity
- rows: 143
- cols: 1176
- observed_fraction: 0.168153
- overall_mean: 0.386944
- avg_item_variance: 0.156567
- avg_agent_variance: 0.140086
- zero_variance_item_fraction: 0.075680
- mean_item_entropy: 0.657537

## Full Comparison
### raw_pre_vs_binary_post
- avg_item_variance_pre: 0.129674
- avg_item_variance_post: 0.048591
- delta_post_minus_pre: -0.081083
- avg_agent_variance_pre: 0.133906
- avg_agent_variance_post: 0.111170
- mean_score_pre: 0.396849
- mean_pass_rate_post: 0.198398
### binary_sensitivity_pre_vs_post
- avg_item_variance_pre_binary: 0.156567
- avg_item_variance_post_binary: 0.048591
- delta_post_minus_pre_binary: -0.107976
- mean_item_entropy_pre_binary: 0.657537
- mean_item_entropy_post_binary: 0.185365
### per_benchmark_avg_item_variance
- pre_full_raw:
  - colbench_backend_programming: 0.134212
  - corebench_hard: 0.140581
  - scicode: 0.015585
  - scienceagentbench: 0.112810
- post_binary:
  - colbench_backend_programming: 0.046107
  - corebench_hard: 0.085185
  - scicode: 0.004310
  - scienceagentbench: 0.070459
- pre_full_binary_sensitivity:
  - colbench_backend_programming: 0.165838
  - corebench_hard: 0.140581
  - scicode: 0.015585
  - scienceagentbench: 0.112810

## Matched Sampling
### raw_pre_vs_binary_post
- Target pre sample size: 32
- Repeats: 50
- overall_mean: sampled pre mean=0.416807, SE=0.002394, post=0.198398, delta(post-pre)=-0.218409, post percentile in pre samples=0.000
- avg_item_variance: sampled pre mean=0.132924, SE=0.002491, post=0.048591, delta(post-pre)=-0.084333, post percentile in pre samples=0.000
- avg_agent_variance: sampled pre mean=0.137929, SE=0.001054, post=0.111170, delta(post-pre)=-0.026759, post percentile in pre samples=0.000
- zero_variance_item_fraction: sampled pre mean=0.147313, SE=0.007450, post=0.743197, delta(post-pre)=0.595884, post percentile in pre samples=1.000
- Benchmark-wise avg item variance:
  - colbench_backend_programming: sampled pre mean=0.138186, SE=0.002919, post=0.046107, delta(post-pre)=-0.092079, post percentile=0.000
  - corebench_hard: sampled pre mean=0.141722, SE=0.002961, post=0.085185, delta(post-pre)=-0.056537, post percentile=0.000
  - scicode: sampled pre mean=0.016047, SE=0.000842, post=0.004310, delta(post-pre)=-0.011736, post percentile=0.020
  - scienceagentbench: sampled pre mean=0.110690, SE=0.000883, post=0.070459, delta(post-pre)=-0.040231, post percentile=0.000

### binary_sensitivity_pre_vs_post
- Target pre sample size: 32
- Repeats: 50
- overall_mean: sampled pre mean=0.406091, SE=0.002469, post=0.198398, delta(post-pre)=-0.207692, post percentile in pre samples=0.000
- avg_item_variance: sampled pre mean=0.160563, SE=0.002723, post=0.048591, delta(post-pre)=-0.111972, post percentile in pre samples=0.000
- avg_agent_variance: sampled pre mean=0.148008, SE=0.001073, post=0.111170, delta(post-pre)=-0.036838, post percentile in pre samples=0.000
- zero_variance_item_fraction: sampled pre mean=0.222687, SE=0.010346, post=0.743197, delta(post-pre)=0.520510, post percentile in pre samples=1.000
- mean_item_entropy: sampled pre mean=0.606797, SE=0.009717, post=0.185365, delta(post-pre)=-0.421432, post percentile in pre samples=0.000
- Benchmark-wise avg item variance:
  - colbench_backend_programming: sampled pre mean=0.170689, SE=0.003192, post=0.046107, delta(post-pre)=-0.124581, post percentile=0.000
  - corebench_hard: sampled pre mean=0.141722, SE=0.002961, post=0.085185, delta(post-pre)=-0.056537, post percentile=0.000
  - scicode: sampled pre mean=0.016047, SE=0.000842, post=0.004310, delta(post-pre)=-0.011736, post percentile=0.020
  - scienceagentbench: sampled pre mean=0.110690, SE=0.000883, post=0.070459, delta(post-pre)=-0.040231, post percentile=0.000
