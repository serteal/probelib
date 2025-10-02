"""Utilities for probe visualization and result export."""

import html
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import torch

from .processing.tokenization import tokenize_dialogues
from .types import Dialogue

if TYPE_CHECKING:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    from .masks import MaskFunction


def _is_notebook() -> bool:
    """Check if running in a Jupyter notebook."""
    try:
        from IPython import get_ipython

        if get_ipython() is not None:
            if "IPKernelApp" in get_ipython().config:
                return True
    except (ImportError, AttributeError):
        pass
    return False


def show_detection_mask_in_html(
    input_ids: torch.Tensor,
    detection_mask: torch.Tensor,
    tokenizer: "PreTrainedTokenizerBase",
):
    """
    Show the decoded tokens painted in red if they are in the detection mask.
    """
    from IPython.display import HTML

    toks = [tokenizer.convert_ids_to_tokens(tok) for tok in input_ids.tolist()]

    # Clean up tokens for display while preserving newlines
    cleaned_toks = []
    for tok in toks:
        # Handle different tokenizer formats
        tok = tok.replace("Ġ", " ")  # GPT-2/LLaMA style
        tok = tok.replace("▁", " ")  # SentencePiece style
        # Replace newline characters with symbol + HTML line break
        tok = tok.replace("Ċ", "↵<br>")  # GPT-2 newline
        tok = tok.replace("\n", "↵<br>")  # Literal newline
        cleaned_toks.append(tok)

    html_parts = []
    assert len(cleaned_toks) == len(detection_mask), (
        f"len(cleaned_toks)={len(cleaned_toks)} != len(detection_mask)={len(detection_mask)}"
    )
    for tok, is_detected in zip(cleaned_toks, detection_mask):
        # HTML-escape the token but preserve the <br> tags we added
        # Split by <br> to handle them separately
        parts = tok.split("<br>")
        escaped_parts = [html.escape(p) for p in parts]
        escaped_tok = "<br>".join(escaped_parts)

        if is_detected:
            html_parts.append(f'<span style="color: red">{escaped_tok}</span>')
        else:
            html_parts.append(escaped_tok)

    return HTML("".join(html_parts))


def show_detection_mask_in_terminal(
    input_ids: torch.Tensor,
    detection_mask: torch.Tensor,
    tokenizer: "PreTrainedTokenizerBase",
    highlight_color: str = "red",
    show_legend: bool = True,
) -> None:
    """
    Show the decoded tokens with ANSI colors in terminal.

    Args:
        input_ids: Token IDs
        detection_mask: Boolean mask indicating selected tokens
        tokenizer: Tokenizer to decode tokens
        highlight_color: Color for highlighted tokens ("red", "green", "blue", "yellow", "magenta", "cyan")
        show_legend: Whether to show a legend explaining the colors
    """
    # ANSI color codes
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
    }
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    color_code = colors.get(highlight_color, colors["red"])

    # Convert tokens
    toks = [tokenizer.convert_ids_to_tokens(tok) for tok in input_ids.tolist()]

    # Clean up tokens for display
    cleaned_toks = []
    for tok in toks:
        # Handle different tokenizer formats
        tok = tok.replace("Ġ", " ")  # GPT-2/LLaMA style
        tok = tok.replace("▁", " ")  # SentencePiece style
        tok = tok.replace(
            "Ċ", "↵\n"
        )  # GPT-2 newline - show symbol and preserve newline
        # Also handle literal newlines
        tok = tok.replace("\n", "↵\n")
        cleaned_toks.append(tok)

    # Build output string
    output_parts = []
    for tok, is_detected in zip(cleaned_toks, detection_mask):
        if is_detected:
            output_parts.append(f"{color_code}{tok}{RESET}")
        else:
            output_parts.append(f"{DIM}{tok}{RESET}")

    # Print legend if requested
    if show_legend:
        print(f"\n{BOLD}Mask Visualization:{RESET}")
        print(f"  {color_code}■{RESET} Selected tokens (mask=True)")
        print(f"  {DIM}■{RESET} Unselected tokens (mask=False)")
        print(f"{DIM}{'─' * 50}{RESET}\n")

    # Print the tokenized text with colors
    print("".join(output_parts))
    print()  # Add newline at end


def visualize_mask(
    dialogue: Dialogue | Sequence[Dialogue],
    mask: "MaskFunction",
    tokenizer: "PreTrainedTokenizerBase",
    device: str | torch.device = "cpu",
    show_legend: bool = True,
    highlight_color: str = "red",
    force_terminal: bool = False,
    **tokenize_kwargs: Any,
) -> None:
    """
    Visualize which tokens are selected by a mask.

    Automatically detects environment (notebook vs terminal) and displays accordingly.
    - In Jupyter notebooks: Shows HTML with colored tokens
    - In terminal: Shows colored text using ANSI codes

    Args:
        dialogue: Single dialogue or list of dialogues to visualize
        mask: Mask function to apply
        tokenizer: Tokenizer to use
        device: Device to place tensors on
        show_legend: Whether to show a legend (terminal only)
        highlight_color: Color for highlighted tokens in terminal ("red", "green", "blue", etc.)
        force_terminal: Force terminal output even in notebook
        **tokenize_kwargs: Additional tokenization arguments

    Examples:
        >>> from probelib import visualize_mask
        >>> from probelib.masks import assistant, contains
        >>>
        >>> dialogue = [
        ...     Message(role="user", content="What is 2+2?"),
        ...     Message(role="assistant", content="The answer is 4.")
        ... ]
        >>>
        >>> # Visualize assistant messages
        >>> visualize_mask(dialogue, assistant(), tokenizer)
        >>>
        >>> # Visualize tokens containing "answer"
        >>> visualize_mask(dialogue, contains("answer"), tokenizer)
        >>>
        >>> # Complex mask
        >>> visualize_mask(dialogue, assistant() & contains("4"), tokenizer)
    """
    from .types import Message

    # Check if dialogue is a single Dialogue (list of Messages) or list of Dialogues
    if dialogue and isinstance(dialogue[0], Message):
        # Single dialogue
        dialogues = [dialogue]
    else:
        # List of dialogues
        dialogues = list(dialogue)

    # Tokenize with the mask
    token_dict = tokenize_dialogues(
        tokenizer=tokenizer,
        dialogues=dialogues,
        mask=mask,
        device=device,
        **tokenize_kwargs,
    )

    # Get the first example for visualization
    input_ids = token_dict["input_ids"][0]
    detection_mask = token_dict["detection_mask"][0]

    # Detect environment and display accordingly
    if _is_notebook() and not force_terminal:
        # In notebook - use HTML display
        from IPython.display import display

        display(show_detection_mask_in_html(input_ids, detection_mask, tokenizer))
    else:
        # In terminal - use ANSI colors
        show_detection_mask_in_terminal(
            input_ids,
            detection_mask,
            tokenizer,
            highlight_color=highlight_color,
            show_legend=show_legend,
        )


def print_metrics(
    metrics: Mapping[str, Mapping[str, tuple[float, float, float] | float]],
    table: bool = False,
    precision: int = 5,
    use_colors: bool = True,
) -> None:
    """Print evaluation metrics in a formatted way with terminal colors.

    Args:
        metrics: Dictionary mapping probe names to metric dictionaries.
            Each metric dict maps metric names to either:
            - (point_estimate, lower_ci, upper_ci) tuples from bootstrap
            - Single float values
        table: If True, print as a tabulated table. If False, print as formatted text.
        precision: Number of decimal places to display.
        use_colors: Whether to use ANSI colors in terminal output.

    Examples:
        >>> metrics = {
        ...     "layer_8": {
        ...         "auroc": (0.95, 0.93, 0.97),
        ...         "recall_at_fpr_0.01": (0.85, 0.82, 0.88)
        ...     },
        ...     "layer_12": {
        ...         "auroc": (0.98, 0.97, 0.99),
        ...         "recall_at_fpr_0.01": (0.92, 0.90, 0.94)
        ...     }
        ... }
        >>> print_metrics(metrics)
        >>> print_metrics(metrics, table=True)
    """
    if not metrics:
        print("No metrics to display.")
        return

    # Check if metrics is a flat dict (single probe) or nested dict (multiple probes)
    # A flat dict has metric values (floats/tuples), nested has dicts as values
    is_flat = False
    if metrics:
        first_value = next(iter(metrics.values()))
        # If first value is a number or tuple (not a dict), it's flat
        if isinstance(first_value, (float, int, tuple)):
            is_flat = True
            # Wrap in a dict for uniform handling
            metrics = {"Metrics": metrics}

    # ANSI color codes
    if use_colors:
        RESET = "\033[0m"
        BOLD = "\033[1m"
        CYAN = "\033[36m"
        GREEN = "\033[32m"
        BLUE = "\033[34m"
        GRAY = "\033[90m"
        BRIGHT_WHITE = "\033[97m"
    else:
        RESET = BOLD = CYAN = GREEN = BLUE = GRAY = BRIGHT_WHITE = ""

    # Format a metric value with colors
    def format_metric(value, precision=precision):
        if isinstance(value, tuple) and len(value) == 3:
            # Bootstrap result with confidence intervals
            point, lower, upper = value
            return f"{GREEN}{point:.{precision}f}{RESET} {GRAY}[{lower:.{precision}f}, {upper:.{precision}f}]{RESET}"
        elif isinstance(value, (float, int)):
            # Single value
            return f"{GREEN}{value:.{precision}f}{RESET}"
        else:
            # Unknown format
            return str(value)

    if table:
        # Use tabulate for table format
        try:
            from tabulate import tabulate
        except ImportError:
            print("tabulate package not found. Install with: pip install tabulate")
            print("Falling back to text format...")
            table = False

    if table:
        # Prepare data for tabulate
        # Get all unique metric names across all probes
        all_metrics = set()
        for probe_metrics in metrics.values():
            all_metrics.update(probe_metrics.keys())
        all_metrics = sorted(all_metrics)

        # Build table rows
        headers = ["Probe"] + all_metrics
        rows = []

        for probe_name, probe_metrics in metrics.items():
            row = [f"{CYAN}{probe_name}{RESET}"]
            for metric_name in all_metrics:
                if metric_name in probe_metrics:
                    value = probe_metrics[metric_name]
                    row.append(format_metric(value))
                else:
                    row.append(f"{GRAY}-{RESET}")
            rows.append(row)

        # Print table with colored headers
        colored_headers = [f"{BOLD}{BRIGHT_WHITE}{h}{RESET}" for h in headers]
        print(tabulate(rows, headers=colored_headers, tablefmt="simple"))
    else:
        for probe_name, probe_metrics in metrics.items():
            # Only print probe name header if not a flat dict (multiple probes)
            if not is_flat:
                print(f"\n{BOLD}{CYAN}{probe_name}{RESET}")
                print(f"{GRAY}{'─' * (len(probe_name) + 2)}{RESET}")

            # Get max metric name length for alignment
            max_name_len = (
                max(len(name) for name in probe_metrics.keys()) if probe_metrics else 0
            )

            for metric_name, value in probe_metrics.items():
                formatted_value = format_metric(value)
                # Keep metric name as-is, just add color
                display_name = f"{BLUE}{metric_name}{RESET}"

                # Print with proper alignment and colors
                spaces = " " * (max_name_len - len(metric_name))
                print(f"  {display_name}{spaces} : {formatted_value}")
