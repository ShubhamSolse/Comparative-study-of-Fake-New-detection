# Methodology
This study uses the CRISP-DM framework—a six-step, industry-standard process—to guide our research on detecting fake news with a Fuzzy Deep Hybrid Network (FDHN). CRISP-DM begins with Business Understanding, where we define our goal: automating the identification of deceptive news. Next is Data Understanding, in which we examine and explore the available data to see how it can help us accomplish that goal.

With a solid grasp of our objectives and data, we move to Data Preparation, cleaning and organizing the dataset to ensure it’s ready for modeling. In the Modeling phase, we build and train our FDHN, integrating fuzzy logic to handle uncertainty inherent in fact-checking. Evaluation follows, where we rigorously test the model against benchmarks to confirm its accuracy and reliability. Finally, Deployment plans how to integrate the solution into real-world fact-checking workflows.

The framework’s strength lies in its clear structure and emphasis on thoroughly understanding both the problem and the data before crafting a solution. By following every CRISP-DM phase, our research remains systematic and transparent, ensuring that each step—from defining the challenge to rolling out our model—is thoughtfully executed and grounded in the realities of fake news detection.



1. **Business Understanding**

    *  Define your core research question: Which fuzzy membership functions and hybrid strategies yield the most accurate fake-news detection?
    *  Clarify objectives: Compare Gaussian, Triangular, Trapezoidal, Sigmoid-based, Bell-shaped, and FNN layers, plus BERT and ANN+BERT hybrids, all within the FDHN architecture.
    *  Identify success criteria: Improved accuracy, F1-score, and robustness against uncertainty.

2. **Data Understanding**

    *  Describe the LIAR2 dataset: 23K fact-checked political statements, with
      full speaker descriptions, timestamps, and justifications.
    *  Explore its features: Statement text, speaker history, fuzzy justifications, and metadata.
    *  Note any quality issues: Missing justifications, imbalanced classes, or noisy context that may affect modeling.

3. **Data Preparation**

    * Clean and normalize text: removing unnecessary information, apply tokenization, and standardize date fields.
    * Engineer fuzzy inputs: For each membership function (Gaussian, Triangular, Trapezoidal, Sigmoid, Bell), compute membership degrees for numerical context.
    * Integrate BERT embeddings: Precompute BERT vectors for statements to serve as an extra feature

4. **Modelling**:
Build multiple FDHN variants:

    *   FDHN+Gaussian
    *   FDHN+Triangular
    *   FDHN+Trapezoidal
    *   FDHN+Sigmoid
    *   FDHN+Bell
    *   FDHN+FNN layer
    *   FDHN+BERT
    *   Hybrid ANN+BERT

5. **Evaluation**

    *  Use accuracy, macro-F1, and micro-F1 to compare models on the held-out test set.
    *  Conduct ablation studies to isolate the impact of each membership function, the FNN layer, and BERT features.
    *  Analyse statistical significance and error cases to understand where each approach excels or struggles.

6. **Deployment**: Outline how the best-performing FDHN variant could be packaged as a fact-checking tool.

This CRISP-DM layout ensures you systematically move from defining your fake-news-detection goals to delivering a rigorously evaluated, deployable FDHN comparison.
