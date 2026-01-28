# Fake News Detection using Fuzzy Deep Hybrid Network (FDHN)

## Project Overview

This project implements a comprehensive fake news detection system using a Fuzzy Deep Hybrid Network (FDHN) approach. The research follows the CRISP-DM methodology to systematically develop and evaluate multiple FDHN variants for detecting deceptive political statements.

## Table of Contents

- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Dataset](#dataset)
- [Model Variants](#model-variants)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Structure

```
Fake_News_Detection/
├── README.md                    # This file
├── FDHN_Compare_Thesis.ipynb   # Main Jupyter notebook with implementation
├── combined_model.keras        # Trained model file
├── fake_news_train.csv         # Training dataset
├── fake_news_test.csv          # Test dataset
└── fake_news_valid.csv         # Validation dataset
```

## Methodology

This project follows the **CRISP-DM (Cross-Industry Standard Process for Data Mining)** framework, which provides a structured approach to data mining projects:

### 1. Business Understanding
- **Research Question**: Which fuzzy membership functions and hybrid strategies yield the most accurate fake-news detection?
- **Objective**: Compare various fuzzy membership functions (Gaussian, Triangular, Trapezoidal, Sigmoid-based, Bell-shaped) and hybrid strategies within the FDHN architecture
- **Success Criteria**: Improved accuracy, F1-score, and robustness against uncertainty

### 2. Data Understanding
- **Dataset**: LIAR2 dataset containing 23,000+ fact-checked political statements
- **Features**: Statement text, speaker history, fuzzy justifications, metadata, timestamps
- **Quality Assessment**: Analysis of missing justifications, class imbalance, and noisy context

### 3. Data Preparation
- Text cleaning and normalization
- Tokenization and date field standardization
- Fuzzy input engineering for different membership functions
- BERT embeddings integration

### 4. Modeling
Multiple FDHN variants are implemented and compared:
- FDHN + Gaussian membership function
- FDHN + Triangular membership function
- FDHN + Trapezoidal membership function
- FDHN + Sigmoid membership function
- FDHN + Bell-shaped membership function
- FDHN + FNN (Fuzzy Neural Network) layer
- FDHN + BERT integration
- Hybrid ANN + BERT approach

### 5. Evaluation
- Performance metrics: Accuracy, Macro-F1, Micro-F1
- Ablation studies to isolate impact of each component
- Statistical significance analysis
- Error case analysis

### 6. Deployment
Guidelines for packaging the best-performing FDHN variant as a production-ready fact-checking tool

## Dataset

The project uses the **LIAR2 dataset**, which contains:
- **Size**: 23,000+ fact-checked political statements
- **Features**:
  - Statement text
  - Speaker information and history
  - Publication date
  - Subject categories
  - Context information
  - Truth ratings (6-point scale)
  - Historical fact-checking counts
  - Detailed justifications

### Data Distribution Analysis

The notebook includes comprehensive exploratory data analysis showing:
- **Top subjects**: Facebook fact-checks, health care, elections, taxes, education
- **Most frequent speakers**: Facebook posts, viral images, Donald Trump, bloggers
- **Class distribution**: Analysis of truth rating distribution across the dataset

## Model Variants

### Fuzzy Membership Functions
1. **Gaussian**: Smooth, bell-shaped curves for uncertainty modeling
2. **Triangular**: Simple triangular functions for computational efficiency
3. **Trapezoidal**: Flat-top functions for stable membership regions
4. **Sigmoid**: S-shaped curves for smooth transitions
5. **Bell-shaped**: Generalized bell curves for flexible modeling

### Hybrid Approaches
- **FDHN + BERT**: Integration of pre-trained language models
- **ANN + BERT**: Traditional neural network with BERT embeddings
- **FNN Layer**: Pure fuzzy neural network implementation

## Installation

### Prerequisites
- Python 3.8+
- Jupyter Notebook
- CUDA-compatible GPU (recommended)

### Required Libraries
```bash
pip install pandas numpy matplotlib seaborn
pip install tensorflow keras
pip install transformers torch
pip install scikit-learn
pip install fuzzy-logic
```

### Setup
1. Clone the repository:
```bash
git clone <repository-url>
cd Fake_News_Detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook FDHN_Compare_Thesis.ipynb
```

## Usage

### Training Models
1. Open the main notebook: `FDHN_Compare_Thesis.ipynb`
2. Run the data preparation cells to load and preprocess the dataset
3. Execute the model training sections for each FDHN variant
4. Compare results using the evaluation metrics

### Making Predictions
```python
# Load the trained model
model = tf.keras.models.load_model('combined_model.keras')

# Prepare input data
processed_text = preprocess_statement(input_statement)

# Make prediction
prediction = model.predict(processed_text)
truth_score = interpret_prediction(prediction)
```

### Evaluation
The notebook includes comprehensive evaluation sections with:
- Accuracy comparisons across all model variants
- F1-score analysis (macro and micro)
- Confusion matrices
- Statistical significance tests
- Error analysis and case studies

## Results

The project systematically compares multiple FDHN variants to identify the most effective approach for fake news detection. Key findings include:

- Performance comparison of different fuzzy membership functions
- Impact of BERT integration on detection accuracy
- Effectiveness of hybrid approaches
- Robustness analysis under uncertainty

*Detailed results and performance metrics are available in the notebook.*

## Key Features

- **Comprehensive Methodology**: Follows industry-standard CRISP-DM framework
- **Multiple Model Variants**: Systematic comparison of 8 different FDHN approaches
- **Fuzzy Logic Integration**: Handles uncertainty inherent in fact-checking
- **BERT Integration**: Leverages state-of-the-art language models
- **Thorough Evaluation**: Statistical analysis and ablation studies
- **Real-world Dataset**: Uses professionally fact-checked political statements

## Technical Highlights

- **Uncertainty Handling**: Fuzzy logic components manage ambiguity in fact-checking
- **Hybrid Architecture**: Combines traditional ML with deep learning approaches
- **Feature Engineering**: Comprehensive text processing and metadata integration
- **Scalable Design**: Modular architecture for easy extension and modification

## Future Work

- Integration with real-time news feeds
- Multi-language support
- Enhanced explainability features
- Mobile application development
- API development for third-party integration

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{fdhn_fake_news_detection,
  title={Fake News Detection using Fuzzy Deep Hybrid Network: A Comparative Study},
  author={[Your Name]},
  year={2024},
  note={Thesis Project - DBS Study}
}
```

## Contact

For questions or collaboration opportunities, please contact [shubhamsolse@gmail.com].

---

**Note**: This project is part of a thesis research conducted during Semester 2 of DBS Study program, focusing on advanced machine learning techniques for misinformation detection.
