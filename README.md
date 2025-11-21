# KodeCamp Task Grader

An AI-powered CLI tool for automated grading and evaluation of KodeCamp promotional tasks using LLM-based assessment with customizable rubrics.

## Overview

KodeCamp Task Grader is a sophisticated automated grading system that leverages Large Language Models (LLMs) to evaluate trainee submissions against predefined rubrics. It integrates with Learning Management Systems (LMS), supports multiple document formats (Google Docs, Google Colab notebooks), and provides structured, consistent evaluations.

## Features

- **LLM-Based Evaluation**: Uses LangChain with support for Ollama and Groq models
- **Flexible Rubric System**: Define custom grading criteria with weighted scoring
- **Multi-Scale Scoring**: Supports 0-1, 0-5, 0-10, and percentage-based scales
- **LMS Integration**: Fetch submissions directly from your LMS
- **Document Downloaders**: Built-in support for Google Docs and Google Colab notebooks
- **Structured Output**: YAML-formatted evaluation results with detailed justifications
- **Extensible Architecture**: Factory pattern for easy addition of new document types

## Installation

### Prerequisites

- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Using uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/uchokoro/KodeCamp-task-grader.git
cd KodeCamp-task-grader

# Install dependencies
uv sync

# Install in development mode
uv pip install -e .
```

### Using pip

```bash
# Clone the repository
git clone https://github.com/uchokoro/KodeCamp-task-grader.git
cd KodeCamp-task-grader

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

## Configuration

### Environment Variables

Create a `.env` file in the project root with the following variables:

```env
# LMS Configuration
LMS_BASE_URL=https://your-lms-instance.com/api
LMS_EMAIL=your-email@example.com
LMS_PASSWORD=your-password

# LLM Configuration (if using Groq)
GROQ_API_KEY=your-groq-api-key
```

### LLM Setup

#### Using Ollama (Local)

1. Install [Ollama](https://ollama.ai/)
2. Pull a model:
   ```bash
   ollama pull llama3.2:3b
   ```

#### Using Groq (Cloud)

1. Sign up at [Groq](https://groq.com/)
2. Get your API key
3. Add it to your `.env` file

## Usage

### Basic Workflow

The typical grading workflow involves:

1. **Define a Rubric**: Create evaluation criteria
2. **Fetch Submissions**: Get submissions from LMS or provide manually
3. **Download Documents**: Retrieve Google Docs/Colab notebooks
4. **Evaluate**: Run LLM-based grading
5. **Review Results**: Analyze structured evaluation output

### 1. Defining a Rubric

Create a rubric with weighted criteria:

```python
from task_grader.grading.rubric import Rubric, Criterion

rubric = Rubric(
    task_id="task_001",
    title="Prompt Engineering Assignment",
    description="Evaluate trainee's understanding of prompt engineering principles",
    overall_max_score=100.0,
    min_passing_score=70.0,
    criteria=[
        Criterion(
            id="clarity",
            name="Clarity and Structure",
            description="Prompt is clear, well-structured, and easy to understand",
            weight=2.0,
            scale="0-10"
        ),
        Criterion(
            id="specificity",
            name="Specificity",
            description="Prompt provides specific, detailed instructions",
            weight=2.5,
            scale="0-10"
        ),
        Criterion(
            id="context",
            name="Context Awareness",
            description="Demonstrates understanding of context and constraints",
            weight=1.5,
            scale="0-10"
        ),
        Criterion(
            id="creativity",
            name="Creativity",
            description="Shows innovative thinking and creative problem-solving",
            weight=1.0,
            scale="0-5"
        ),
    ]
)
```

**Available Score Scales:**

- `"0-1"`: Binary (pass/fail)
- `"0-5"`: Five-point scale
- `"0-10"`: Ten-point scale
- `"percentage"`: 0-100 percentage

### 2. Fetching Submissions from LMS

```python
from task_grader.lms.lms_client import LMSClient, SubmissionCategory

# Initialize LMS client from environment variables
client = LMSClient.from_env()

# Or initialize manually
client = LMSClient(
    base_url="https://your-lms.com/api",
    email="your-email@example.com",
    password="your-password"
)

# Login
client.login()

# Fetch submissions for a specific task
submissions = client.get_task_submissions(
    task_id="task_001",
    workspace_slug="kodecamp-cohort-5",
    category=SubmissionCategory.SUBMITTED,  # or GRADED, ALL
    offset=0,
    limit=100
)

# Each submission contains metadata
for submission in submissions:
    print(f"Trainee: {submission.trainee_name}")
    print(f"Submission URLs: {submission.solution_urls}")
    print(f"Status: {submission.submission_status}")
    print(f"Current Score: {submission.score}")

# Logout when done
client.logout()
```

### 3. Downloading Documents

#### Google Docs

```python
from task_grader.docs.google_docs import GoogleDocsDownloader

downloader = GoogleDocsDownloader()

# Download as different formats
filepath = downloader.download_as(
    doc_url="https://docs.google.com/document/d/YOUR_DOC_ID/edit",
    dest_dir="./submissions",
    filename="trainee_submission",
    as_format="txt"  # or "docx", "pdf", "odt"
)

print(f"Downloaded to: {filepath}")
```

#### Google Colab Notebooks

```python
from task_grader.docs.google_colab import GoogleColabDownloader

downloader = GoogleColabDownloader()

filepath = downloader.download_as(
    doc_url="https://colab.research.google.com/drive/YOUR_FILE_ID",
    dest_dir="./submissions",
    filename="trainee_notebook",
    as_format="ipynb"
)

print(f"Downloaded to: {filepath}")
```

#### Using the Factory Pattern

```python
from task_grader.docs.factory import SubmissionDownloaderFactory
from task_grader.docs.google_docs import GoogleDocsDownloader
from task_grader.docs.google_colab import GoogleColabDownloader

# Create and configure factory
factory = SubmissionDownloaderFactory()
factory.register_downloader("google_docs", GoogleDocsDownloader)
factory.register_downloader("google_colab", GoogleColabDownloader)

# Get appropriate downloader
downloader = factory.get_downloader("google_docs")
filepath = downloader.download_as(
    doc_url="https://docs.google.com/document/d/...",
    dest_dir="./submissions"
)
```

### 4. Evaluating Submissions

```python
from pathlib import Path
from task_grader.grading.evaluator import LLMTaskEvaluator

# Initialize evaluator with Ollama
evaluator = LLMTaskEvaluator.from_ollama(
    model_name="llama3.2:3b",
    prompt_template_path="task_grader/grading/prompt_templates/grading_prompt_template.txt",
    temperature=0.0
)

# Load assignment and submission texts
assignment_text = Path("./assignments/task_001.txt").read_text()
submission_text = Path("./submissions/trainee_submission.txt").read_text()

# Evaluate the submission
result = evaluator.evaluate(
    rubric=rubric,
    assignment=assignment_text,
    submission=submission_text,
    trainee_name="John Doe",
    knowledge_area="prompt engineering",
    cohort_specifics="Agentic AI Track, Nov 2025",
    track_name="Agentic AI",
    other_notes=""  # Optional additional constraints
)

# Access results
print(f"Intro: {result.intro}")
print(f"Overall Evaluation: {result.overall_evaluation}")
print(f"Verdict: {result.overall_verdict}")
print(f"Total Score: {result.total_score}/{rubric.overall_max_score}")

# View criterion-specific evaluations
for criterion_eval in result.criteria:
    print(f"\n{criterion_eval.name} [{criterion_eval.id}]:")
    print(f"  Score: {criterion_eval.score} (scale: {criterion_eval.score_scale})")
    print(f"  Justification: {criterion_eval.justification}")

# Raw YAML output for debugging
print(f"\nRaw YAML:\n{result.raw_yaml}")
```

### 5. Complete Example

```python
from pathlib import Path
from task_grader.lms.lms_client import LMSClient, SubmissionCategory
from task_grader.docs.google_docs import GoogleDocsDownloader
from task_grader.grading.rubric import Rubric, Criterion
from task_grader.grading.evaluator import LLMTaskEvaluator

# 1. Define rubric
rubric = Rubric(
    task_id="prompt_eng_001",
    title="Prompt Engineering Fundamentals",
    description="Assessment of basic prompt engineering skills",
    overall_max_score=100.0,
    min_passing_score=70.0,
    criteria=[
        Criterion(
            id="clarity",
            name="Clarity",
            description="Clear and well-structured prompts",
            weight=2.0,
            scale="0-10"
        ),
        Criterion(
            id="effectiveness",
            name="Effectiveness",
            description="Achieves desired outcomes",
            weight=3.0,
            scale="0-10"
        ),
    ]
)

# 2. Fetch submissions from LMS
lms_client = LMSClient.from_env()
lms_client.login()
submissions = lms_client.get_task_submissions(
    task_id="prompt_eng_001",
    workspace_slug="kodecamp-cohort-5",
    category=SubmissionCategory.SUBMITTED
)

# 3. Initialize evaluator
evaluator = LLMTaskEvaluator.from_ollama(
    model_name="llama3.2:3b",
    prompt_template_path="task_grader/grading/prompt_templates/grading_prompt_template.txt"
)

# Load assignment
assignment_text = Path("./assignments/prompt_eng_001.txt").read_text()

# 4. Process each submission
downloader = GoogleDocsDownloader()

for submission in submissions[:5]:  # Process first 5
    print(f"\nEvaluating {submission.trainee_name}...")

    # Download submission
    submission_path = downloader.download_as(
        doc_url=submission.solution_urls[0],
        dest_dir="./temp_submissions",
        filename=f"{submission.trainee_id}_{submission.submission_id}",
        as_format="txt"
    )

    # Read submission
    submission_text = Path(submission_path).read_text()

    # Evaluate
    result = evaluator.evaluate(
        rubric=rubric,
        assignment=assignment_text,
        submission=submission_text,
        trainee_name=submission.trainee_name,
        knowledge_area="prompt engineering",
        cohort_specifics="Agentic AI Track, Nov 2025",
        track_name="Agentic AI"
    )

    # Display results
    print(f"Score: {result.total_score}/{rubric.overall_max_score}")
    print(f"Verdict: {result.overall_verdict}")
    print(f"Evaluation: {result.overall_evaluation}")

lms_client.logout()
```

## Project Structure

```
KodeCamp-task-grader/
├── task_grader/
│   ├── __init__.py
│   ├── cli.py                      # CLI interface (coming soon)
│   ├── agent/                      # Agentic components (future)
│   │   └── __init__.py
│   ├── docs/                       # Document downloaders
│   │   ├── __init__.py
│   │   ├── factory.py              # Factory pattern for downloaders
│   │   ├── generic.py              # Base downloader interface
│   │   ├── google_colab.py         # Google Colab notebook downloader
│   │   └── google_docs.py          # Google Docs downloader
│   ├── grading/                    # Core grading logic
│   │   ├── __init__.py
│   │   ├── evaluator.py            # LLM-based evaluator
│   │   ├── prompt_builder.py       # Prompt construction utilities
│   │   ├── rubric.py               # Rubric definitions
│   │   └── prompt_templates/
│   │       └── grading_prompt_template.txt
│   └── lms/                        # LMS integration
│       ├── __init__.py
│       └── lms_client.py           # LMS API client
├── tests/                          # Unit tests
├── main.py
├── pyproject.toml
├── LICENSE
└── README.md
```

## API Reference

### Core Classes

#### `Rubric`

Defines grading criteria and overall scoring parameters.

**Attributes:**

- `task_id`: Unique identifier for the task
- `title`: Rubric title
- `description`: Detailed rubric description
- `overall_max_score`: Maximum possible score
- `min_passing_score`: Minimum score to pass
- `criteria`: List of `Criterion` objects

#### `Criterion`

Individual grading criterion with weight and scale.

**Attributes:**

- `id`: Unique criterion identifier
- `name`: Human-readable name
- `description`: What this criterion evaluates
- `weight`: Relative weight in scoring
- `scale`: Score scale ("0-1", "0-5", "0-10", "percentage")

#### `LLMTaskEvaluator`

Main evaluation engine using LLMs.

**Methods:**

- `from_ollama(model_name, prompt_template_path, temperature, **kwargs)`: Create evaluator with Ollama
- `evaluate(rubric, assignment, submission, trainee_name, knowledge_area, cohort_specifics, track_name, other_notes)`: Evaluate a submission

#### `EvaluationResult`

Structured evaluation output.

**Attributes:**

- `intro`: One-sentence summary
- `overall_evaluation`: Paragraph summary (3-5 sentences)
- `overall_verdict`: Qualitative verdict (e.g., "excellent", "good", "fail")
- `criteria`: List of `CriterionEvaluation` objects
- `total_score`: Computed numeric score
- `raw_yaml`: Raw YAML response from LLM

#### `LMSClient`

Interface for LMS API interactions.

**Methods:**

- `from_env()`: Create client from environment variables
- `login()`: Authenticate with LMS
- `get_task_submissions(task_id, workspace_slug, category, offset, limit)`: Fetch submissions
- `logout()`: End session

## Development

### Running Tests

```bash
# Using uv
uv run pytest

# Using pytest directly
pytest

# Run specific test file
pytest tests/test_evaluator.py

# With coverage
pytest --cov=task_grader
```

### Code Quality

The project uses:

- **Black**: Code formatting
- **Ruff**: Linting
- **MyPy**: Type checking
- **Pre-commit**: Git hooks for quality checks

```bash
# Format code
black task_grader/

# Run linter
ruff check task_grader/

# Type checking
mypy task_grader/

# Install pre-commit hooks
pre-commit install

# Run all checks
pre-commit run --all-files
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Roadmap

- [ ] Complete CLI implementation with Typer
- [ ] Add support for more document types (PDFs, Word docs)
- [ ] Implement agentic grading workflows with LangGraph
- [ ] Add support for more LLM providers (OpenAI, Anthropic)
- [ ] Batch processing and parallel evaluation
- [ ] Web dashboard for result visualization
- [ ] Export results to CSV/JSON/PDF
- [ ] Integration with more LMS platforms

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Anthony Okoro**

- Email: antoine.okoro@gmail.com
- GitHub: [@uchokoro](https://github.com/uchokoro)

## Acknowledgments

- Built for [KodeCamp](https://kodecamp.com/) bootcamp grading automation
- Uses [LangChain](https://python.langchain.com/) for LLM orchestration
- Supports [Ollama](https://ollama.ai/) for local LLM inference
- Integrates with [Groq](https://groq.com/) for cloud-based LLMs

## Support

For issues, questions, or contributions, please:

- Open an issue on [GitHub](https://github.com/uchokoro/KodeCamp-task-grader/issues)
- Contact the author directly

---

**Note**: This is an early-stage project under active development. APIs and features may change.
