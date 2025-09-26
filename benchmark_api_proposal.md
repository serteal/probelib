# Probelib Benchmark & Monitoring API Proposal

## Overview

This proposal extends probelib to support:
1. **Paper reproduction** - Easy reproduction of activation monitoring papers
2. **Dynamic monitoring protocols** - Beyond static probes (e.g., follow-up questions)
3. **Standardized benchmarks** - Curated test sets with specific model requirements

The design philosophy: **simple cases should be simple, complex cases should be possible**.

## Core Abstractions

### 1. Monitor Protocol - The Detection Strategy

```python
from abc import ABC, abstractmethod
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase
    from ..datasets import DialogueDataset
    from ..types import Dialogue

class Monitor(ABC):
    """Base class for all monitoring strategies."""

    @abstractmethod
    def train(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        train_data: DialogueDataset,
        **kwargs
    ) -> None:
        """
        Train the monitor on labeled data.

        Args:
            model: Model to use for training
            tokenizer: Tokenizer for the model
            train_data: Training dataset with built-in labels
            **kwargs: Additional training arguments
        """
        pass

    @abstractmethod
    def score(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        test_data: DialogueDataset | list[Dialogue]
    ) -> np.ndarray:
        """
        Score test data, returning detection probabilities.

        Args:
            model: Model to use for scoring
            tokenizer: Tokenizer for the model
            test_data: Test data to score (DialogueDataset or raw dialogues)

        Returns:
            Array of detection scores/probabilities
        """
        pass
```

### 2. Static Probe Monitor - Simple Wrapper for Existing Probes

```python
from ..probes import BaseProbe, Logistic
from ..processing import collect_activations
from .. import masks

class ProbeMonitor(Monitor):
    """Simple static probe monitoring (most common case)."""

    def __init__(
        self,
        probe_cls: type[BaseProbe] = Logistic,
        mask: MaskFunction | None = None,
        layer: int | None = None,
        **probe_kwargs
    ):
        """
        Initialize a static probe monitor.

        Args:
            probe_cls: Probe class to use (default: Logistic)
            mask: Mask for activation collection (default: assistant())
            layer: Layer to probe (None = search all layers)
            **probe_kwargs: Additional arguments for probe initialization
        """
        self.probe_cls = probe_cls
        self.mask = mask or masks.assistant()  # sensible default
        self.layer = layer
        self.probe_kwargs = probe_kwargs
        self.probe = None

    def train(self, model, tokenizer, train_data, **kwargs):
        """Train the probe on the training data."""
        # DialogueDataset contains both dialogues and labels
        dialogues = train_data.dialogues
        labels = train_data.labels

        # Collect activations
        layers_to_try = [self.layer] if self.layer else list(range(model.config.num_hidden_layers))
        acts = collect_activations(
            model, tokenizer, dialogues,
            layers=layers_to_try,
            mask=self.mask,
            **kwargs
        )

        # Train probe
        self.probe = self.probe_cls(layer=self.layer, **self.probe_kwargs)
        self.probe.fit(acts, labels)

    def score(self, model, tokenizer, test_data):
        """Score test data using the trained probe."""
        if self.probe is None:
            raise RuntimeError("Monitor must be trained before scoring")

        # Handle both DialogueDataset and raw dialogues
        if isinstance(test_data, DialogueDataset):
            dialogues = test_data.dialogues
        else:
            dialogues = test_data

        acts = collect_activations(
            model, tokenizer, dialogues,
            layers=[self.probe.layer],
            mask=self.mask
        )
        return self.probe.predict_proba(acts)[:, 1]  # positive class probabilities
```

### 3. Dynamic Protocol Monitor - For Complex Monitoring Strategies

```python
class ProtocolMonitor(Monitor):
    """Monitor that can modify dialogues before detection."""

    def __init__(
        self,
        transform_fn: Callable,
        probe_cls: type[BaseProbe] = Logistic,
        mask: MaskFunction | None = None,
        **probe_kwargs
    ):
        """
        Initialize a protocol-based monitor.

        Args:
            transform_fn: Function that transforms dialogues before monitoring
                         (dialogue, model, tokenizer) -> modified_dialogue
            probe_cls: Probe class to use for detection
            mask: Mask for activation collection
            **probe_kwargs: Arguments for probe initialization
        """
        self.transform_fn = transform_fn
        self.probe_cls = probe_cls
        self.mask = mask or masks.assistant()
        self.probe_kwargs = probe_kwargs
        self.probe = None

    def train(self, model, tokenizer, train_data, **kwargs):
        """Train on transformed dialogues."""
        # DialogueDataset contains both dialogues and labels
        dialogues = train_data.dialogues
        labels = train_data.labels

        # Transform training dialogues
        transformed = []
        for dialogue in dialogues:
            transformed.append(self.transform_fn(dialogue, model, tokenizer))

        # Train probe on transformed dialogues
        acts = collect_activations(model, tokenizer, transformed, mask=self.mask)
        self.probe = self.probe_cls(**self.probe_kwargs)
        self.probe.fit(acts, labels)

    def score(self, model, tokenizer, test_data):
        """Score transformed test dialogues."""
        # Extract dialogues
        if isinstance(test_data, DialogueDataset):
            dialogues = test_data.dialogues
        else:
            dialogues = test_data

        # Transform test dialogues
        transformed = []
        for dialogue in dialogues:
            transformed.append(self.transform_fn(dialogue, model, tokenizer))

        acts = collect_activations(model, tokenizer, transformed, mask=self.mask)
        return self.probe.predict_proba(acts)[:, 1]
```

## Benchmark System

### Data Structures

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class TestSet:
    """A test dataset paired with a specific model."""
    name: str
    data: DialogueDataset  # Contains dialogues and labels
    model: str  # Model identifier (HuggingFace path)
    metadata: dict = field(default_factory=dict)

@dataclass
class TrainingConfig:
    """Configuration for training a monitor."""
    data: DialogueDataset  # Contains dialogues and labels
    model: str | None = None  # Optional model override for training
```

### Benchmark Class

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

class Benchmark:
    """Manages evaluation across multiple test sets."""

    def __init__(
        self,
        name: str | None = None,
        test_sets: list[TestSet] | None = None
    ):
        """
        Initialize benchmark from preset or custom test sets.

        Args:
            name: Name of preset benchmark to load
            test_sets: Custom list of test sets
        """
        if name:
            # Load preset benchmark (e.g., from a paper)
            config = BenchmarkRegistry.load(name)
            self.test_sets = config['test_sets']
            self.default_training = config.get('default_training')
            self.name = name
        else:
            self.test_sets = test_sets or []
            self.default_training = None
            self.name = "custom"

    def evaluate(
        self,
        monitor: Monitor | type[Monitor],
        training: TrainingConfig | None = None,
        metrics: list[str | Callable] | None = None,
        **monitor_kwargs
    ) -> BenchmarkResults:
        """
        Evaluate a monitor across all test sets.

        Args:
            monitor: Monitor instance or class
            training: Training configuration (uses default if None)
            metrics: Metrics to compute (uses standard set if None)
            **monitor_kwargs: Args for monitor if class provided

        Returns:
            BenchmarkResults object with scores and metrics
        """
        # Create monitor if class provided
        if isinstance(monitor, type):
            monitor = monitor(**monitor_kwargs)

        # Use default training if not provided
        if training is None:
            if self.default_training is None:
                raise ValueError("No training config provided and no default available")
            training = self.default_training

        # Default metrics
        if metrics is None:
            metrics = ['auroc', 'balanced_accuracy', 'recall@5']

        results = {}

        # Group test sets by model for efficiency
        model_groups = self._group_by_model(self.test_sets)

        for model_name, test_sets in model_groups.items():
            # Load model once per group
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map='auto'
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model.eval()

            # Determine training model
            train_model_name = training.model or model_name

            if train_model_name == model_name:
                # Use same model for training
                monitor.train(model, tokenizer, training.data)
            else:
                # Load different model for training
                train_model = AutoModelForCausalLM.from_pretrained(
                    train_model_name,
                    torch_dtype=torch.bfloat16,
                    device_map='auto'
                )
                train_tokenizer = AutoTokenizer.from_pretrained(train_model_name)
                train_model.eval()
                monitor.train(train_model, train_tokenizer, training.data)

                # Clean up training model
                del train_model
                torch.cuda.empty_cache()

            # Evaluate on all test sets for this model
            for test_set in test_sets:
                scores = monitor.score(model, tokenizer, test_set.data)

                # Compute metrics using test set labels
                test_metrics = {}
                for metric in metrics:
                    if isinstance(metric, str):
                        metric_fn = get_metric_by_name(metric)
                    else:
                        metric_fn = metric
                    test_metrics[metric_fn.__name__] = metric_fn(
                        test_set.data.labels,
                        scores
                    )

                results[test_set.name] = {
                    'metrics': test_metrics,
                    'model': model_name,
                    'scores': scores
                }

            # Clean up model
            del model
            torch.cuda.empty_cache()

        return BenchmarkResults(results, benchmark=self, monitor=monitor)

    def _group_by_model(self, test_sets: list[TestSet]) -> dict[str, list[TestSet]]:
        """Group test sets by required model."""
        groups = {}
        for test_set in test_sets:
            if test_set.model not in groups:
                groups[test_set.model] = []
            groups[test_set.model].append(test_set)
        return groups
```

### Paper Reproduction Registry

```python
class BenchmarkRegistry:
    """Registry for paper reproduction benchmarks."""
    _benchmarks = {}

    @classmethod
    def register_paper(cls, name: str, paper_config: dict):
        """
        Register a paper's benchmark configuration.

        Args:
            name: Unique identifier for the benchmark
            paper_config: Configuration dictionary with:
                - citation: Paper citation
                - datasets: Dict of test dataset configs
                - training_data: Training dataset config
                - default_model: Fallback model if not specified
                - monitor_class: Default monitor class (optional)
        """
        test_sets = []

        for dataset_name, dataset_config in paper_config['datasets'].items():
            # Load dataset based on source
            if 'class' in dataset_config:
                # Use specific dataset class
                dataset_cls = dataset_config['class']
                dataset = dataset_cls(**dataset_config.get('kwargs', {}))
            else:
                # Load from source
                dataset = cls._load_dataset(dataset_config['source'])

            test_sets.append(TestSet(
                name=dataset_name,
                data=dataset,
                model=dataset_config.get('model', paper_config['default_model'])
            ))

        # Load training dataset
        train_config = paper_config['training_data']
        if 'class' in train_config:
            train_data = train_config['class'](**train_config.get('kwargs', {}))
        else:
            train_data = cls._load_dataset(train_config['source'])

        # Store configuration
        cls._benchmarks[name] = {
            'test_sets': test_sets,
            'default_training': TrainingConfig(
                data=train_data,
                model=train_config.get('model')
            ),
            'paper_reference': paper_config.get('citation'),
            'default_monitor': paper_config.get('monitor_class', ProbeMonitor)
        }

    @classmethod
    def load(cls, name: str) -> dict:
        """Load a registered benchmark configuration."""
        if name not in cls._benchmarks:
            raise ValueError(f"Unknown benchmark: {name}")
        return cls._benchmarks[name]

    @classmethod
    def list_benchmarks(cls) -> list[str]:
        """List all available benchmark names."""
        return list(cls._benchmarks.keys())
```

## Usage Examples

### Simple Cases

```python
import probelib as pl
from probelib.benchmarks import Benchmark, ProbeMonitor

# 1. Simplest: Static probe on preset benchmark
benchmark = Benchmark('azaria-mitchell-2023')
results = benchmark.evaluate(ProbeMonitor, layer=12, C=1.0)

# 2. Custom mask with preset benchmark
monitor = ProbeMonitor(
    probe_cls=pl.probes.Logistic,
    mask=pl.masks.assistant() & pl.masks.last_n_tokens(50),
    layer=12,
    C=1.0
)
results = benchmark.evaluate(monitor)

# 3. Custom training data
from probelib.benchmarks import TrainingConfig

train_config = TrainingConfig(
    data=pl.datasets.DolusChatDataset(),  # Has dialogues and labels built-in
    model='meta-llama/Llama-3.2-1B'  # Optional model override
)
benchmark = Benchmark('cadenza')
results = benchmark.evaluate(ProbeMonitor, training=train_config, layer=16)
```

### Complex Cases

```python
from probelib.benchmarks import ProtocolMonitor, Monitor
import torch

# 1. Dynamic monitoring with follow-up questions
def add_truthfulness_check(dialogue, model, tokenizer):
    """Add a follow-up question about truthfulness."""
    from probelib.types import Message

    # Add follow-up question
    augmented = dialogue + [
        Message("user", "Were you being completely truthful in your response?")
    ]

    # Generate response
    inputs = tokenizer.apply_chat_template(augmented, return_tensors='pt')
    inputs = inputs.to(model.device)

    with torch.no_grad():
        output = model.generate(inputs, max_new_tokens=50, do_sample=False)

    response_text = tokenizer.decode(output[0][inputs.shape[1]:], skip_special_tokens=True)

    # Add response to dialogue
    augmented.append(Message("assistant", response_text))
    return augmented

monitor = ProtocolMonitor(
    transform_fn=add_truthfulness_check,
    probe_cls=pl.probes.MLP,
    mask=pl.masks.assistant() & pl.masks.nth_message(-1),  # Last assistant message
    hidden_dim=256
)

benchmark = Benchmark('deception-suite')
results = benchmark.evaluate(monitor)

# 2. Multi-step monitoring protocol
class ConversationalMonitor(Monitor):
    """Custom monitor with multiple rounds of questions."""

    def __init__(self, questions: dict[str, str], probe_cls=pl.probes.Logistic):
        self.questions = questions
        self.probe_cls = probe_cls
        self.probes = {}  # One probe per question

    def _add_question_and_generate(self, dialogue, question, model, tokenizer):
        """Add question and generate response."""
        from probelib.types import Message

        augmented = dialogue + [Message("user", question)]
        inputs = tokenizer.apply_chat_template(augmented, return_tensors='pt')
        inputs = inputs.to(model.device)

        with torch.no_grad():
            output = model.generate(inputs, max_new_tokens=100, do_sample=False)

        response = tokenizer.decode(output[0][inputs.shape[1]:], skip_special_tokens=True)
        augmented.append(Message("assistant", response))
        return augmented

    def train(self, model, tokenizer, train_data, **kwargs):
        """Train separate probe for each question."""
        # DialogueDataset contains both dialogues and labels
        dialogues = train_data.dialogues
        labels = train_data.labels

        for q_name, question in self.questions.items():
            # Augment dialogues with this question
            augmented = []
            for dialogue in dialogues:
                aug = self._add_question_and_generate(dialogue, question, model, tokenizer)
                augmented.append(aug)

            # Train probe for this question
            acts = pl.processing.collect_activations(
                model, tokenizer, augmented,
                layers=[12],  # or search across layers
                mask=pl.masks.assistant() & pl.masks.last_n_tokens(100)
            )
            probe = self.probe_cls()
            probe.fit(acts, labels)
            self.probes[q_name] = probe

    def score(self, model, tokenizer, test_data):
        """Score using ensemble of question-specific probes."""
        import numpy as np

        # Extract dialogues
        if isinstance(test_data, DialogueDataset):
            dialogues = test_data.dialogues
        else:
            dialogues = test_data

        all_scores = []

        for q_name, question in self.questions.items():
            # Augment with question
            augmented = []
            for dialogue in dialogues:
                aug = self._add_question_and_generate(dialogue, question, model, tokenizer)
                augmented.append(aug)

            # Get scores from this question's probe
            acts = pl.processing.collect_activations(
                model, tokenizer, augmented,
                layers=[self.probes[q_name].layer],
                mask=pl.masks.assistant() & pl.masks.last_n_tokens(100)
            )
            scores = self.probes[q_name].predict_proba(acts)[:, 1]
            all_scores.append(scores)

        # Average scores across questions
        return np.mean(all_scores, axis=0)

# Use the conversational monitor
monitor = ConversationalMonitor(
    questions={
        'truthfulness': 'Were you being truthful in your response?',
        'confidence': 'How confident are you in your answer?',
        'clarification': 'Is there anything you should clarify about your response?'
    },
    probe_cls=pl.probes.Logistic
)
results = benchmark.evaluate(monitor)
```

### Paper Reproduction

```python
# Register a paper's benchmark
BenchmarkRegistry.register_paper('azaria-mitchell-2023', {
    'citation': 'Azaria, A., & Mitchell, T. (2023). The Internal State of an LLM...',
    'default_model': 'meta-llama/Llama-2-7b-hf',
    'datasets': {
        'statements': {
            'class': StatementsDataset,  # Custom dataset class (contains labels)
            'kwargs': {'split': 'test'}
        },
        'questions': {
            'class': QuestionsDataset,
            'kwargs': {'split': 'test'}
        },
        'generated': {
            'source': 'azaria/generated-v1',  # HuggingFace dataset
            'model': 'meta-llama/Llama-2-7b-hf'
        }
    },
    'training_data': {
        'class': StatementsDataset,
        'kwargs': {'split': 'train'}
    },
    'monitor_class': ProbeMonitor
})

# Now anyone can reproduce the paper with one line
benchmark = Benchmark('azaria-mitchell-2023')
results = benchmark.evaluate(ProbeMonitor, layer=15)

# Or try their own monitor on the same benchmark
results = benchmark.evaluate(my_custom_monitor)
```

### Cadenza-style Benchmark

```python
# Register Cadenza benchmark with multiple test sets
BenchmarkRegistry.register_paper('cadenza-2024', {
    'citation': 'Cadenza: Eliciting Truthfulness in LLMs...',
    'default_model': 'meta-llama/Llama-3.2-3B-Instruct',
    'datasets': {
        'persona': {
            'class': pl.datasets.PersonaDataset,
            'model': 'meta-llama/Llama-3.2-3B-Instruct'
        },
        'sycophancy': {
            'class': pl.datasets.SycophancyDataset,
            'model': 'meta-llama/Llama-3.2-3B-Instruct'
        },
        'repe_facts': {
            'class': pl.datasets.REPEDataset,
            'kwargs': {'variant': 'facts'},
            'model': 'mistralai/Mistral-7B-v0.1'
        },
        'wildguard': {
            'class': pl.datasets.WildGuardDataset,
            'model': 'meta-llama/Llama-Guard-3-8B'
        }
    },
    'training_data': {
        'class': pl.datasets.DolusChatDataset  # Default training set
    }
})

# User provides their own training strategy
# (Note: CombinedDataset would need to merge dialogues and labels from multiple datasets)
my_training = TrainingConfig(
    data=pl.datasets.CombinedDataset([
        pl.datasets.DolusChatDataset(),
        pl.datasets.TruthfulQADataset(),
        pl.datasets.SandbaggingDataset()
    ])
)

benchmark = Benchmark('cadenza-2024')
results = benchmark.evaluate(
    ProbeMonitor,
    training=my_training,
    layer=20,
    C=0.1
)

# Compare different monitors
monitors = {
    'static': ProbeMonitor(layer=20),
    'dynamic': ProtocolMonitor(transform_fn=my_transform, layer=20),
    'ensemble': ConversationalMonitor(questions=my_questions)
}

all_results = {}
for name, monitor in monitors.items():
    all_results[name] = benchmark.evaluate(monitor, training=my_training)
```

## Implementation Plan

### Phase 1: Core Monitor Abstraction
- Implement `Monitor` base class
- Create `ProbeMonitor` wrapper for existing probes
- Test with current probe infrastructure

### Phase 2: Benchmark Infrastructure
- Implement `TestSet` and `TrainingConfig` dataclasses
- Create `Benchmark` class with model grouping
- Add `BenchmarkResults` with serialization

### Phase 3: Dynamic Monitoring
- Implement `ProtocolMonitor` for dialogue transformation
- Add example transform functions (follow-up questions, etc.)
- Create specialized monitors for common patterns

### Phase 4: Registry & Paper Reproduction
- Build `BenchmarkRegistry` with preset loading
- Add 2-3 paper configurations as proof of concept
- Create helper functions for common paper patterns

### Phase 5: Advanced Features
- Multi-dataset training support (CombinedDataset helper)
- Parallel evaluation across model groups
- Benchmark versioning for reproducibility
- Results comparison and visualization tools

## Benefits

1. **Simple cases stay simple**: One-line evaluation with sensible defaults
2. **Progressive complexity**: More sophisticated monitoring requires more code, but is possible
3. **Clean separation of concerns**: Monitoring strategy vs. evaluation protocol
4. **Paper reproduction**: Preset configurations enable easy reproduction
5. **Flexible training**: Different models/data for training vs. evaluation
6. **Extensible**: New monitor types via inheritance
7. **Efficient**: Model grouping and caching for performance
8. **Existing code works**: ProbeMonitor wraps existing probe infrastructure
9. **Labels included**: DialogueDataset already contains labels, simplifying the API