# Limitations and intended use

- Training-free means there is no learned model fitting at inference, not that parameter selection was data-independent. All labeled development and ablation results use the same 20 DRIVE training images.
- A06 is the primary conference configuration but has lower development macro F1 and IoU than L00. A07 and some sensitivity settings also exceed A06. Do not claim universal superiority or optimality.
- The strong paired results refer to specific matched local-only and region-only controls, not independent validation or all competing methods. Statistical intervals and p-values are conditional on the development sample.
- The adapted L00 comparator is not a verified reproduction of the original APSRG enhancement pipeline. No controlled state-of-the-art comparison is provided here.
- The unlabeled test prediction disagreement is descriptive; it does not measure correctness against ground truth. Independent test accuracy and cross-dataset generalization are not established by this package.
- The historical runtime is a six-image CPU subset with five repeats; power/thermal state was not instrumented. The reported A06 time exceeds L00, and training-free operation alone is not evidence of faster inference. New schedules and hardware can change times.
- Thin/peripheral vessel proxies and segmentation errors have not been clinically validated. This research software is not a medical device and is not intended for diagnosis or clinical decision-making.
- Exact numerical reproduction requires matching input arrays, dependencies and execution conditions. The archived configuration and numerical files are provided as reference evidence; a reviewed local clean-environment run passed with historical package versions. Independent external replication and cross-platform validation remain unverified.
