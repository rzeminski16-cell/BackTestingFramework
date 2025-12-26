"""
Interactive handler for user prompts during fundamental data collection.

This module handles:
- Prompting users for decisions on ambiguous data
- Remembering answers to avoid asking the same question repeatedly
- Logging all decisions for audit trail
- Saving/loading remembered answers for session persistence
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field


logger = logging.getLogger(__name__)


@dataclass
class Decision:
    """A recorded decision made during processing."""
    question_key: str
    question_text: str
    options: List[str]
    chosen_option: str
    chosen_index: int
    timestamp: str
    context: Dict[str, Any] = field(default_factory=dict)


class InteractiveHandler:
    """
    Handles interactive prompts with answer memory.

    Features:
    - Asks user for input when encountering ambiguous data
    - Remembers answers based on question keys (not per-symbol)
    - Logs all decisions to a file for audit
    - Can save/load remembered answers for session persistence

    Usage:
        handler = InteractiveHandler(log_dir=Path("logs"))

        # Ask a question (will remember the answer for the same key)
        answer = handler.ask_choice(
            question_key="eps_type",
            question="Which EPS type should be used?",
            options=["EPS Diluted", "EPS Basic", "EPS Reported"],
            context={"symbol": "AAPL", "field_names": ["epsDiluted", "epsBasic"]}
        )

        # Same question key returns remembered answer
        answer2 = handler.ask_choice(
            question_key="eps_type",
            question="Which EPS type?",
            options=["EPS Diluted", "EPS Basic"],
        )  # Returns same answer without prompting
    """

    def __init__(self,
                 log_dir: Optional[Path] = None,
                 memory_file: Optional[Path] = None,
                 auto_save: bool = True):
        """
        Initialize the interactive handler.

        Args:
            log_dir: Directory for decision logs. If None, no logging.
            memory_file: File to persist remembered answers. If None, uses log_dir.
            auto_save: Whether to save memory after each decision.
        """
        self.log_dir = Path(log_dir) if log_dir else None
        self.auto_save = auto_save

        # Decision memory - maps question_key to Decision
        self._memory: Dict[str, Decision] = {}

        # All decisions made (for logging)
        self._decisions: List[Decision] = []

        # Setup logging
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._decision_log_path = self.log_dir / "decisions.log"
            self._memory_path = memory_file or (self.log_dir / "remembered_answers.json")

            # Load existing memory if available
            self._load_memory()
        else:
            self._decision_log_path = None
            self._memory_path = None

    def _load_memory(self):
        """Load remembered answers from file."""
        if self._memory_path and self._memory_path.exists():
            try:
                with open(self._memory_path) as f:
                    data = json.load(f)

                for key, decision_data in data.items():
                    self._memory[key] = Decision(
                        question_key=decision_data['question_key'],
                        question_text=decision_data['question_text'],
                        options=decision_data['options'],
                        chosen_option=decision_data['chosen_option'],
                        chosen_index=decision_data['chosen_index'],
                        timestamp=decision_data['timestamp'],
                        context=decision_data.get('context', {}),
                    )

                logger.info(f"Loaded {len(self._memory)} remembered answers from {self._memory_path}")

            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Could not load memory file: {e}")

    def _save_memory(self):
        """Save remembered answers to file."""
        if self._memory_path:
            data = {}
            for key, decision in self._memory.items():
                data[key] = {
                    'question_key': decision.question_key,
                    'question_text': decision.question_text,
                    'options': decision.options,
                    'chosen_option': decision.chosen_option,
                    'chosen_index': decision.chosen_index,
                    'timestamp': decision.timestamp,
                    'context': decision.context,
                }

            with open(self._memory_path, 'w') as f:
                json.dump(data, f, indent=2)

    def _log_decision(self, decision: Decision):
        """Log a decision to the decision log file."""
        if self._decision_log_path:
            log_entry = {
                'timestamp': decision.timestamp,
                'question_key': decision.question_key,
                'question': decision.question_text,
                'options': decision.options,
                'chosen': decision.chosen_option,
                'chosen_index': decision.chosen_index,
                'context': decision.context,
            }

            with open(self._decision_log_path, 'a') as f:
                f.write(json.dumps(log_entry) + "\n")

    def ask_choice(self,
                   question_key: str,
                   question: str,
                   options: List[str],
                   context: Optional[Dict[str, Any]] = None,
                   default: Optional[int] = None) -> Tuple[str, int]:
        """
        Ask user to choose from options, remembering the answer.

        If this question_key has been asked before, returns the remembered answer.

        Args:
            question_key: Unique identifier for this type of question
            question: The question to display
            options: List of options to choose from
            context: Additional context for logging (e.g., symbol, field names)
            default: Default option index if user just presses Enter

        Returns:
            Tuple of (chosen_option, chosen_index)
        """
        context = context or {}

        # Check if we already have an answer for this question
        if question_key in self._memory:
            remembered = self._memory[question_key]
            logger.info(f"Using remembered answer for '{question_key}': {remembered.chosen_option}")
            print(f"\n[Using remembered answer] {question}: {remembered.chosen_option}")
            return remembered.chosen_option, remembered.chosen_index

        # Display question
        print("\n" + "=" * 60)
        print(f"DECISION REQUIRED: {question}")
        print("=" * 60)

        if context:
            print(f"Context: {json.dumps(context, indent=2)}")

        print("\nOptions:")
        for i, option in enumerate(options):
            default_marker = " (default)" if i == default else ""
            print(f"  [{i + 1}] {option}{default_marker}")

        # Get user input
        while True:
            prompt = f"\nEnter choice [1-{len(options)}]"
            if default is not None:
                prompt += f" (default: {default + 1})"
            prompt += ": "

            try:
                user_input = input(prompt).strip()

                if not user_input and default is not None:
                    choice_idx = default
                else:
                    choice_idx = int(user_input) - 1

                if 0 <= choice_idx < len(options):
                    break
                else:
                    print(f"Please enter a number between 1 and {len(options)}")

            except ValueError:
                print("Please enter a valid number")
            except EOFError:
                # Non-interactive mode - use default or first option
                choice_idx = default if default is not None else 0
                print(f"\n[Non-interactive] Using option: {options[choice_idx]}")
                break

        chosen_option = options[choice_idx]

        # Record decision
        decision = Decision(
            question_key=question_key,
            question_text=question,
            options=options,
            chosen_option=chosen_option,
            chosen_index=choice_idx,
            timestamp=datetime.now().isoformat(),
            context=context,
        )

        self._memory[question_key] = decision
        self._decisions.append(decision)
        self._log_decision(decision)

        if self.auto_save:
            self._save_memory()

        print(f"\n✓ Remembered: '{question_key}' = '{chosen_option}'")
        print("  (This choice will be used for all securities)")

        return chosen_option, choice_idx

    def ask_yes_no(self,
                   question_key: str,
                   question: str,
                   context: Optional[Dict[str, Any]] = None,
                   default: bool = True) -> bool:
        """
        Ask a yes/no question, remembering the answer.

        Args:
            question_key: Unique identifier for this question
            question: The question to display
            context: Additional context for logging
            default: Default answer if user just presses Enter

        Returns:
            True for yes, False for no
        """
        options = ["Yes", "No"]
        default_idx = 0 if default else 1

        _, chosen_idx = self.ask_choice(
            question_key=question_key,
            question=question,
            options=options,
            context=context,
            default=default_idx,
        )

        return chosen_idx == 0

    def confirm(self,
                message: str,
                default: bool = True) -> bool:
        """
        Ask for confirmation (not remembered).

        Args:
            message: Confirmation message
            default: Default answer

        Returns:
            True if confirmed, False otherwise
        """
        default_str = "Y/n" if default else "y/N"

        try:
            response = input(f"\n{message} [{default_str}]: ").strip().lower()

            if not response:
                return default
            return response in ('y', 'yes')

        except EOFError:
            return default

    def log_issue(self,
                  issue_type: str,
                  message: str,
                  context: Optional[Dict[str, Any]] = None,
                  severity: str = "warning"):
        """
        Log an issue encountered during processing.

        Args:
            issue_type: Category of issue (e.g., "missing_data", "api_error")
            message: Description of the issue
            context: Additional context
            severity: "info", "warning", or "error"
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'severity': severity,
            'issue_type': issue_type,
            'message': message,
            'context': context or {},
        }

        # Log to Python logger
        log_func = getattr(logger, severity, logger.warning)
        log_func(f"[{issue_type}] {message}")

        # Log to issues file if log_dir set
        if self.log_dir:
            issues_path = self.log_dir / "issues.log"
            with open(issues_path, 'a') as f:
                f.write(json.dumps(log_entry) + "\n")

    def get_all_decisions(self) -> List[Decision]:
        """Get all decisions made in this session."""
        return self._decisions.copy()

    def get_remembered_answers(self) -> Dict[str, str]:
        """Get dictionary of question_key -> chosen_option."""
        return {k: v.chosen_option for k, v in self._memory.items()}

    def clear_memory(self):
        """Clear all remembered answers."""
        self._memory.clear()
        if self._memory_path and self._memory_path.exists():
            self._memory_path.unlink()
        logger.info("Cleared remembered answers")

    def print_summary(self):
        """Print a summary of decisions made."""
        print("\n" + "=" * 60)
        print("DECISION SUMMARY")
        print("=" * 60)

        if not self._decisions:
            print("No decisions were made in this session.")
            return

        print(f"\nTotal decisions: {len(self._decisions)}")
        print("\nRemembered answers:")

        for key, decision in self._memory.items():
            print(f"  • {key}: {decision.chosen_option}")

        if self._decision_log_path:
            print(f"\nFull decision log: {self._decision_log_path}")

        if self.log_dir:
            issues_path = self.log_dir / "issues.log"
            if issues_path.exists():
                with open(issues_path) as f:
                    issue_count = sum(1 for _ in f)
                print(f"Issues logged: {issue_count} (see {issues_path})")
