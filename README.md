# Vec2Summ: Text Summarization via Probabilistic Sentence Embeddings

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Vec2Summ** is a novel approach to text summarization that leverages probabilistic sentence embeddings. Instead of directly generating summaries, Vec2Summ:

1. **Embeds** text into high-dimensional sentence embeddings
2. **Models** the distribution of embeddings using mean and covariance
3. **Samples** new vectors from this learned distribution  
4. **Reconstructs** text from sampled vectors using vec2text
5. **Summarizes** the reconstructed texts using LLMs

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/vec2summ.git
cd vec2summ

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Environment Setup

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your OpenAI API key
OPENAI_API_KEY=your_openai_api_key_here
```

### Basic Usage

```bash
# Run a basic experiment with CSV data
python scripts/run_experiment.py \
    --data_path data/your_dataset.csv \
    --text_column "text" \
    --embedding_type openai \
    --n_samples 10 \
    --evaluate \
    --visualize \
    --generate_summaries \
    --output_dir results/experiment_1
```

## 🎛️ Embedding Types

Vec2Summ supports two embedding types with different trade-offs:

### OpenAI Embeddings
- **Models**: `text-embedding-ada-002`, `text-embedding-3-small`, `text-embedding-3-large`
- ✅ High quality, well-trained embeddings
- ✅ No local model download required
- ❌ Requires OpenAI API key and costs money
- ❌ Internet connection required

### GTR Embeddings (Open Source)
- **Model**: `sentence-transformers/gtr-t5-base`
- ✅ Free and runs locally
- ✅ No API key or internet required (after download)
- ✅ Full control over the model
- ❌ Requires more computational resources
- ❌ Initial model download (~1GB)

### ⚠️ Critical: Embedding-Corrector Pairing

**The vec2text library requires proper pairing between embeddings and correctors:**

- **OpenAI embeddings** → **OpenAI corrector** (same model name)
- **GTR embeddings** → **GTR corrector** (`gtr-base`)

**Mixing different types will cause errors!** Our code automatically handles this pairing.

```python
# ✅ Correct - automatically paired
embeddings, corrector, models = get_embeddings_and_corrector(
    texts, embedding_type="openai", openai_model="text-embedding-ada-002"
)

# ❌ Wrong - manual pairing can lead to errors
embeddings = get_openai_embeddings(texts)
corrector = vec2text.load_pretrained_corrector("gtr-base")  # WRONG!
```

## 📊 Experiment Examples

### Amazon Reviews Summarization

```bash
python scripts/run_experiment.py \
    --data_path data/amazon_reviews.tsv \
    --amazon_reviews \
    --amazon_sample_size 50 \
    --embedding_type openai \
    --n_samples 15 \
    --evaluate \
    --evaluate_coverage \
    --generate_summaries \
    --visualize \
    --output_dir results/amazon_experiment
```

### Custom Dataset with GTR Embeddings

```bash
python scripts/run_experiment.py \
    --data_path data/my_texts.json \
    --text_column "content" \
    --embedding_type gtr \
    --gtr_model "sentence-transformers/gtr-t5-base" \
    --n_samples 20 \
    --max_samples 1000 \
    --evaluate \
    --output_dir results/gtr_experiment
```

### Using Configuration Files

For easier experiment management, you can use YAML configuration files:

```bash
# Run experiment with OpenAI embeddings
python scripts/run_config_experiment.py configs/default.yaml

# Run experiment with GTR embeddings
python scripts/run_config_experiment.py configs/gtr_embeddings.yaml

# Run Amazon reviews experiment
python scripts/run_config_experiment.py configs/amazon_reviews.yaml --openai-api-key YOUR_KEY
```

Example configuration file (`configs/gtr_embeddings.yaml`):
```yaml
data:
  path: "data/raw/sample.csv"
  text_column: "text"
  max_samples: null

model:
  embedding_type: "gtr"  # Use GTR instead of OpenAI
  gtr_model: "sentence-transformers/gtr-t5-base"
  n_samples: 10

evaluation:
  enabled: true
  coverage_eval: true
  generate_summaries: true
  visualize: true

output:
  dir: "results/gtr_experiment"
```

## 🏗️ Project Structure

```
vec2summ/
├── src/vec2summ/           # Main package
│   ├── core/               # Core algorithms
│   │   ├── embeddings.py   # Embedding generation
│   │   ├── distribution.py # Distribution modeling
│   │   ├── reconstruction.py # Text reconstruction
│   │   └── summarization.py # LLM summarization
│   ├── data/               # Data processing
│   │   ├── loader.py       # Data loading utilities
│   │   └── preprocessing.py # Text preprocessing
│   ├── evaluation/         # Evaluation metrics
│   │   └── metrics.py      # Quality metrics & G-Eval
│   ├── utils/              # Utilities
│   │   ├── visualization.py # Plotting functions
│   │   └── logging_config.py # Logging setup
│   └── experiment.py       # Main experiment runner
├── scripts/                # Command-line scripts
├── configs/                # Configuration files
├── data/                   # Data directory
│   ├── raw/                # Raw datasets
│   └── processed/          # Processed data
├── notebooks/              # Jupyter notebooks
├── tests/                  # Unit tests
├── results/                # Experiment outputs
└── docs/                   # Documentation
```

## 🔧 Configuration Options

### Data Parameters
- `--data_path`: Path to your dataset
- `--text_column`: Column containing text data
- `--max_samples`: Maximum number of texts to process
- `--amazon_reviews`: Enable Amazon reviews processing mode

### Model Parameters
- `--embedding_type`: Choose `openai` or `gtr` embeddings
- `--openai_model`: OpenAI embedding model (default: `text-embedding-ada-002`)
- `--gtr_model`: GTR model (default: `sentence-transformers/gtr-t5-base`)
- `--n_samples`: Number of vectors to sample from distribution

#### Embedding Type Comparison

**OpenAI Embeddings** (`--embedding_type openai`):
- ✅ High quality, well-trained embeddings
- ✅ No local model download required
- ✅ Consistent performance across domains
- ❌ Requires OpenAI API key and internet connection
- ❌ Usage costs apply
- ❌ No control over model architecture

**GTR Embeddings** (`--embedding_type gtr`):
- ✅ Free to use (open source)
- ✅ Runs locally (no internet required after download)
- ✅ Full control over the model
- ✅ No usage limits or costs
- ❌ Requires initial model download (~1GB)
- ❌ Requires more computational resources
- ❌ May need fine-tuning for specific domains

### Evaluation Options
- `--evaluate`: Enable reconstruction quality evaluation
- `--evaluate_coverage`: Enable G-Eval coverage assessment
- `--generate_summaries`: Generate and compare summaries
- `--visualize`: Create embedding visualizations

## 📈 Output Files

Each experiment generates:

- `reconstructed_texts.txt`: Text reconstructed from sampled embeddings
- `vec2summ_summary.txt`: Summary from Vec2Summ method
- `direct_llm_summary.txt`: Direct LLM summary for comparison
- `summary_comparison.json`: G-Eval comparison results
- `metrics.json`: Reconstruction quality metrics
- `coverage_metrics.json`: Coverage evaluation results
- `embeddings_visualization.png`: PCA visualization
- `mean_vector.npy` & `covariance_matrix.npy`: Distribution parameters

## 📚 API Reference

### Core Functions

```python
from vec2summ import (
    get_openai_embeddings,
    calculate_distribution_params,
    sample_from_distribution,
    reconstruct_text_from_embeddings,
    generate_vec2summ_summary
)

# Load your texts
texts = ["Your text data here..."]

# Generate embeddings
embeddings = get_openai_embeddings(texts)

# Model distribution
mean, cov = calculate_distribution_params(embeddings)

# Sample and reconstruct
samples = sample_from_distribution(mean, cov, n_samples=10)
reconstructed = reconstruct_text_from_embeddings(samples, corrector)

# Generate summary
summary = generate_vec2summ_summary(reconstructed)
```

### Evaluation

```python
from vec2summ.evaluation import evaluate_reconstruction, evaluate_coverage_geval

# Evaluate reconstruction quality
metrics = evaluate_reconstruction(original_texts, reconstructed_texts, "openai")

# Evaluate coverage using G-Eval
coverage = evaluate_coverage_geval(original_texts, reconstructed_texts)
```

## 🧪 Supported Datasets

- **CSV/TSV files**: Standard tabular data with text columns
- **JSON/JSONL**: Structured data with configurable text fields
- **Amazon Reviews**: Special processing for large review datasets
- **TIFU Dataset**: Reddit "Today I F***ed Up" posts
- **Custom formats**: Extensible data loading system

## 🎯 Evaluation Metrics

### Reconstruction Quality
- **Semantic Similarity**: Cosine similarity between original and reconstructed embeddings
- **Coverage Statistics**: Mean, median, min, max similarity scores

### Summary Quality (G-Eval Framework)
- **Coverage**: How well the summary captures key information
- **Conciseness**: Efficiency of information conveyance
- **Coherence**: Organization and fluency
- **Factual Accuracy**: Consistency with source material

## 🔬 Research Applications

Vec2Summ is particularly useful for:

- **Embedding Space Analysis**: Understanding semantic distributions
- **Lossy Compression Studies**: Information preservation through embeddings
- **Probabilistic Text Generation**: Sampling-based content creation
- **Summarization Research**: Novel approaches to text summarization
- **Representation Learning**: Analyzing learned text representations

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📖 Citation

If you use Vec2Summ in your research, please cite:

```bibtex
@misc{vec2summ2024,
  title={Vec2Summ: Text Summarization via Probabilistic Sentence Embeddings},
  author={Li, Mao},
  year={2025},
  url={https://github.com/yourusername/vec2summ}
}
```

## 🙏 Acknowledgments

- [vec2text](https://github.com/jxmorris12/vec2text) for embedding inversion capabilities
- [OpenAI](https://openai.com/) for embedding models and evaluation LLMs
- [Hugging Face](https://huggingface.co/) for transformer models and infrastructure

## 📞 Support

- 📧 Email: maolee@umich.edu
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/vec2summ/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/yourusername/vec2summ/discussions)

---

**Happy Summarizing! 🎉**
