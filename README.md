# GRUBI
This is the implementation of the proposed method in https://arxiv.org/abs/1907.08023 for modeling and control of a reconfigurable photonic circuit using deep learning.

There are two implementations: one that uses "tensorflow 1.12" located under the folder "TF1", and the other uses "tensorflow 2.3.1" located under the folder "TF2". Each folder has a subfolder "real_valued" that includes the implementation of the main setting, where we can only measure power distribution at the output; while the subfolder "complex_valued" includes the implementation for the fully-quantum setting where we can measure phases at the output indirectly using interferometer measurements.
