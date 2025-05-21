from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class TaskType(Enum):
    """
    Different tasks at the dataset / cell type level.

    ZEROSHOT: the cell type is only used in test.
    FEWSHOT: the cell type is partially used in train/val and mostly in test.
    TRAINING: the cell type is used in train/val and not in test.
    """

    ZEROSHOT = "zeroshot"
    FEWSHOT = "fewshot"
    TRAINING = "training"


@dataclass
class TaskSpec:
    """Specification for a training or testing task"""

    dataset: str  # e.g. "replogle"
    cell_type: Optional[str] = None  # e.g. "jurkat"
    task_type: TaskType = TaskType.ZEROSHOT


def parse_dataset_specs(specs: List[str]) -> List[TaskSpec]:
    """Parse dataset specifications into TaskSpec objects

    Format: ``dataset[:cell_type[:task_type]]``
    Examples:
    - ``replogle``
    - ``replogle:jurkat:zeroshot``
    - ``sciplex:k562:fewshot``
    """
    parsed_specs = []

    for spec in specs:
        parts = spec.split(":")
        dataset = parts[0]
        cell_type = parts[1] if len(parts) > 1 else None
        task_type = TaskType.TRAINING  # Default

        if len(parts) > 2:
            task_type = TaskType[parts[2].upper()]

        parsed_specs.append(
            TaskSpec(dataset=dataset, cell_type=cell_type, task_type=task_type)
        )

    return parsed_specs
