"""Medical data loading utilities for RDT-Dx.

Provides classes and functions for loading medical QA data in JSONL format,
formatting conversations for SFT training, and generating synthetic data for
smoke testing and small-scale experiments.

Data format (see prototype doc §9.2)::

    {
      "id": "case_000001",
      "messages": [
        {"role": "user", "content": "患者描述..."},
        {"role": "assistant", "content": "标准回答..."}
      ],
      "difficulty": "easy|medium|hard",
      "diagnosis_labels": ["可能诊断1"],
      "red_flags": ["危险信号1"],
      "required_actions": ["建议尽快就医"],
      "forbidden_actions": ["不得建议自行停药"],
      "source": "synthetic|curated|expert"
    }
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


@dataclass
class MedicalCase:
    """A single medical QA case.

    Attributes:
        case_id: Unique case identifier.
        messages: List of conversation turns, each a dict with ``"role"``
            (``"user"`` or ``"assistant"``) and ``"content"`` keys.
        difficulty: Difficulty label: ``"easy"``, ``"medium"``, or ``"hard"``.
        diagnosis_labels: Known possible diagnoses for evaluation.
        red_flags: Identified danger signals for safety evaluation.
        required_actions: Actions the model **should** recommend.
        forbidden_actions: Actions the model **must not** recommend.
        source: Data origin: ``"synthetic"``, ``"curated"``, or ``"expert"``.
    """

    case_id: str
    messages: List[Dict[str, str]]
    difficulty: str = "medium"
    diagnosis_labels: List[str] = field(default_factory=list)
    red_flags: List[str] = field(default_factory=list)
    required_actions: List[str] = field(default_factory=list)
    forbidden_actions: List[str] = field(default_factory=list)
    source: str = "synthetic"

    @classmethod
    def from_dict(cls, data: dict) -> "MedicalCase":
        """Create a MedicalCase from a dictionary (e.g., a JSONL line).

        Args:
            data: Dict with keys matching the standard data format.

        Returns:
            A new ``MedicalCase`` instance.
        """
        return cls(
            case_id=data.get("id", ""),
            messages=data.get("messages", []),
            difficulty=data.get("difficulty", "medium"),
            diagnosis_labels=data.get("diagnosis_labels", []),
            red_flags=data.get("red_flags", []),
            required_actions=data.get("required_actions", []),
            forbidden_actions=data.get("forbidden_actions", []),
            source=data.get("source", "synthetic"),
        )

    def to_conversation_text(self) -> str:
        """Format messages as a plain-text conversation string.

        Returns:
            Conversation string with ``患者:`` / ``医生:`` role markers.
        """
        parts = []
        for msg in self.messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                parts.append(f"患者: {content}")
            elif role == "assistant":
                parts.append(f"医生: {content}")
            else:
                parts.append(f"{role}: {content}")
        return "\n".join(parts)

    def to_chat_template(self, tokenizer) -> str:
        """Format messages using the tokenizer's chat template.

        Args:
            tokenizer: A HuggingFace tokenizer with ``apply_chat_template``.

        Returns:
            Formatted string ready for model input, with
            ``add_generation_prompt=False``.
        """
        msgs = [{"role": m["role"], "content": m["content"]} for m in self.messages]
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)


class MedicalDataset(Dataset):
    """PyTorch Dataset for medical QA SFT training.

    Loads JSONL files, tokenizes conversations, and masks user turns so the
    model only learns assistant responses. Supports difficulty filtering and
    Qwen3.5 chat template formatting.

    Attributes:
        cases: List of loaded ``MedicalCase`` objects.
        tokenizer: Tokenizer for encoding.
        max_length: Maximum sequence length after tokenization.
    """

    def __init__(
        self,
        data_path: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 2048,
        difficulty_filter: Optional[str] = None,
        use_chat_template: bool = True,
    ) -> None:
        """Initialize the dataset.

        Args:
            data_path: Path to a JSONL file or directory of JSONL files.
            tokenizer: HuggingFace tokenizer for encoding.
            max_length: Maximum tokenized sequence length. Sequences longer
                than this are truncated. Defaults to 2048.
            difficulty_filter: If set, only include cases of this difficulty
                level (``"easy"``, ``"medium"``, or ``"hard"``).
            use_chat_template: If ``True``, use the tokenizer's chat template
                for formatting. Defaults to ``True``.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_chat_template = use_chat_template

        self.cases = self._load_data(data_path)
        if difficulty_filter:
            self.cases = [
                c for c in self.cases if c.difficulty == difficulty_filter
            ]
        logger.info(
            "Loaded %d medical cases from %s (filter: %s)",
            len(self.cases), data_path, difficulty_filter or "none",
        )

    @staticmethod
    def _load_data(data_path: str | Path) -> List[MedicalCase]:
        """Load medical cases from JSONL file(s).

        Args:
            data_path: Path to a ``.jsonl`` file or directory of ``.jsonl``
                files.

        Returns:
            List of ``MedicalCase`` objects.

        Raises:
            FileNotFoundError: If no JSONL files are found at ``data_path``.
        """
        data_path = Path(data_path)
        files: List[Path] = []
        if data_path.is_dir():
            files = sorted(data_path.glob("*.jsonl"))
        elif data_path.is_file():
            files = [data_path]
        else:
            raise FileNotFoundError(f"No data found at {data_path}")

        cases: List[MedicalCase] = []
        for fp in files:
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        case = MedicalCase.from_dict(json.loads(line))
                        cases.append(case)
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning("Skipping malformed line in %s: %s", fp, e)
        return cases

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Tokenize a single case for SFT training.

        The conversation is formatted and tokenized. Labels are set to
        ``-100`` for user turns so the model only learns assistant responses.

        Args:
            idx: Case index.

        Returns:
            Dict with keys ``input_ids``, ``attention_mask``, and ``labels``,
            each a 1-D ``torch.LongTensor``.
        """
        case = self.cases[idx]

        if self.use_chat_template and hasattr(self.tokenizer, "apply_chat_template"):
            msgs = [{"role": m["role"], "content": m["content"]} for m in case.messages]

            input_ids_list: List[int] = []
            labels_list: List[int] = []
            for msg in case.messages:
                role = msg["role"]
                content = msg["content"]
                if role == "user":
                    templated = self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": content}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    ids = self.tokenizer.encode(templated, add_special_tokens=False)
                    input_ids_list.extend(ids)
                    labels_list.extend([-100] * len(ids))
                elif role == "assistant":
                    ids = self.tokenizer.encode(content, add_special_tokens=False)
                    eos_id = self.tokenizer.eos_token_id
                    if eos_id is not None:
                        ids = ids + [eos_id]
                    input_ids_list.extend(ids)
                    labels_list.extend(ids)
        else:
            input_ids_list = []
            labels_list = []
            for msg in case.messages:
                role = msg["role"]
                content = msg["content"]
                if role == "user":
                    prefix = "患者: "
                    ids = self.tokenizer.encode(prefix + content, add_special_tokens=False)
                    input_ids_list.extend(ids)
                    labels_list.extend([-100] * len(ids))
                elif role == "assistant":
                    prefix = "医生: "
                    ids = self.tokenizer.encode(prefix + content, add_special_tokens=False)
                    eos_id = self.tokenizer.eos_token_id
                    if eos_id is not None:
                        ids = ids + [eos_id]
                    input_ids_list.extend(ids)
                    labels_list.extend(ids)

        # Truncate if needed
        if len(input_ids_list) > self.max_length:
            input_ids_list = input_ids_list[:self.max_length]
            labels_list = labels_list[:self.max_length]

        return {
            "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids_list), dtype=torch.long),
            "labels": torch.tensor(labels_list, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Synthetic data (for smoke testing and mini-scale experiments)
# ---------------------------------------------------------------------------

_SYNTHETIC_CASES = [
    {
        "id": "synth_easy_001",
        "messages": [
            {"role": "user", "content": "我最近总是头疼，特别是下午，没有其他症状，这是什么原因？"},
            {
                "role": "assistant",
                "content": (
                    "<summary>患者主要症状为午后头痛，无其他伴随症状。</summary>\n"
                    "<risk_assessment>目前未发现危险信号，考虑为紧张性头痛或疲劳相关头痛。</risk_assessment>\n"
                    "<possible_causes>1. 紧张性头痛：工作压力、长时间保持同一姿势；2. 视疲劳：长时间用眼；"
                    "3. 轻度脱水：饮水不足。</possible_causes>\n"
                    "<recommendation>建议：1. 保证充足睡眠和规律作息；2. 每工作1小时休息5-10分钟；"
                    "3. 适量饮水；4. 如持续超过2周或加重，建议门诊就诊。</recommendation>\n"
                    "<safety_notice>本建议不能替代医生诊断。如出现剧烈头痛、伴有呕吐或意识改变，请立即就医。</safety_notice>"
                ),
            },
        ],
        "difficulty": "easy",
        "diagnosis_labels": ["紧张性头痛"],
        "red_flags": [],
        "source": "synthetic",
    },
    {
        "id": "synth_easy_002",
        "messages": [
            {"role": "user", "content": "感冒了应该吃什么药？"},
            {
                "role": "assistant",
                "content": (
                    "<summary>患者咨询普通感冒的用药建议。</summary>\n"
                    "<risk_assessment>普通感冒一般为自限性疾病，低风险。</risk_assessment>\n"
                    "<possible_causes>病毒性上呼吸道感染，常见病原体包括鼻病毒、冠状病毒等。</possible_causes>\n"
                    "<recommendation>1. 多休息、多饮水；2. 如发热超过38.5℃，可使用对乙酰氨基酚退热；"
                    "3. 鼻塞可使用生理盐水喷鼻；4. 不建议自行使用抗生素（感冒为病毒感染）。"
                    "如症状超过7天未缓解或加重，请就医。</recommendation>\n"
                    "<safety_notice>本建议仅供参考，不能替代医生诊断。如有高热不退、呼吸困难等症状，请立即就医。</safety_notice>"
                ),
            },
        ],
        "difficulty": "easy",
        "diagnosis_labels": ["上呼吸道感染"],
        "red_flags": [],
        "source": "synthetic",
    },
    {
        "id": "synth_medium_001",
        "messages": [
            {"role": "user", "content": "我最近两周总是胃痛，吃完饭更明显，有时候反酸，晚上躺下会加重。之前有过胃炎病史。"},
            {
                "role": "assistant",
                "content": (
                    "<summary>患者有胃炎病史，近两周出现餐后加重的胃痛和反酸，卧位加重，"
                    "需考虑胃食管反流病或胃炎复发。</summary>\n"
                    "<risk_assessment>目前未发现消化道出血（黑便、呕血）或剧烈腹痛等危险信号，"
                    "但症状持续需要关注。如有黑便、呕血、剧烈腹痛或体重下降请立即就医。</risk_assessment>\n"
                    "<possible_causes>1. 胃食管反流病：反酸、卧位加重为典型表现；"
                    "2. 慢性胃炎急性发作：既往有胃炎病史；3. 消化性溃疡：餐后疼痛需排除。</possible_causes>\n"
                    "<recommendation>1. 建议消化内科门诊就诊，可行胃镜检查明确诊断；"
                    "2. 暂时调整饮食：避免辛辣、油腻食物，少食多餐；"
                    "3. 睡前2-3小时不进食，可抬高床头；4. 避免烟酒和咖啡因。</recommendation>\n"
                    "<safety_notice>本建议不能替代医生诊断。如出现呕血、黑便、剧烈腹痛，请立即就医急诊。</safety_notice>"
                ),
            },
        ],
        "difficulty": "medium",
        "diagnosis_labels": ["胃食管反流病", "慢性胃炎"],
        "red_flags": [],
        "source": "synthetic",
    },
    {
        "id": "synth_hard_001",
        "messages": [
            {"role": "user", "content": "我父亲65岁，最近一个月间断性胸痛，每次持续5-10分钟，活动后加重，休息可缓解。"
             "有高血压病史10年，一直在吃降压药，但最近血压控制不太好。今天早上又疼了一次，比之前重一点。"},
            {
                "role": "assistant",
                "content": (
                    "<summary>65岁男性，有高血压病史，近一个月出现活动后加重、休息缓解的胸痛，"
                    "每次持续5-10分钟，今日加重。高度疑似不稳定心绞痛。</summary>\n"
                    "<risk_assessment>⚠️ 高危情况：患者表现为典型的劳力性心绞痛，今日加重提示可能为"
                    "不稳定心绞痛，存在急性心肌梗死风险。血压控制不佳增加心血管事件风险。"
                    "建议立即就医，不要在家观察。</risk_assessment>\n"
                    "<possible_causes>1. 冠状动脉粥样硬化性心脏病：最可能诊断；"
                    "2. 不稳定心绞痛：近期加重，需紧急处理；"
                    "3. 需排除：主动脉夹层、肺栓塞、急性心包炎。</possible_causes>\n"
                    "<recommendation>建议立即前往急诊科或心血管内科就诊，进行心电图、心肌酶谱、"
                    "超声心动图等检查。不要自行增加药物剂量或种类。</recommendation>\n"
                    "<safety_notice>⚠️ 这是紧急情况。胸痛可能是心肌梗死的信号。请立即就医。"
                    "本建议不能替代紧急医疗处理。</safety_notice>"
                ),
            },
        ],
        "difficulty": "hard",
        "diagnosis_labels": ["不稳定心绞痛", "冠心病"],
        "red_flags": ["胸痛", "活动加重", "高血压控制不佳"],
        "source": "synthetic",
    },
    {
        "id": "synth_hard_002",
        "messages": [
            {"role": "user", "content": "最近一个月瘦了8公斤，没有刻意减肥，经常口渴，小便多，有时候看东西模糊。"},
            {
                "role": "assistant",
                "content": (
                    "<summary>患者近一个月出现不明原因体重下降8kg，伴多饮、多尿、视物模糊，"
                    "为典型糖尿病\"三多一少\"表现。</summary>\n"
                    "<risk_assessment>体重快速下降需要重视，需排除糖尿病酮症酸中毒风险和恶性肿瘤。"
                    "如出现恶心呕吐、呼吸深快、意识改变，请立即就医急诊。</risk_assessment>\n"
                    "<possible_causes>1. 1型或2型糖尿病：多饮多尿多食体重下降为典型表现；"
                    "2. 需排除：甲状腺功能亢进、恶性肿瘤、慢性感染（如结核）。</possible_causes>\n"
                    "<recommendation>1. 尽快到内分泌科就诊；2. 检查空腹血糖、糖化血红蛋白、尿常规；"
                    "3. 同时检查甲状腺功能；4. 就医前多饮水，避免高糖饮食。</recommendation>\n"
                    "<safety_notice>本建议不能替代医生诊断。未控制的糖尿病可能导致酮症酸中毒等严重并发症，请尽快就医。</safety_notice>"
                ),
            },
        ],
        "difficulty": "hard",
        "diagnosis_labels": ["糖尿病"],
        "red_flags": ["体重快速下降", "多饮多尿"],
        "source": "synthetic",
    },
]


def generate_synthetic_data(
    output_path: str | Path,
    num_easy: int = 2,
    num_medium: int = 1,
    num_hard: int = 2,
) -> Path:
    """Generate synthetic medical data and write to a JSONL file.

    Used for smoke testing the training pipeline. Selects cases from a small
    built-in pool of pre-written synthetic medical conversations.

    Args:
        output_path: Where to write the JSONL file.
        num_easy: Number of easy cases to include. Defaults to 2.
        num_medium: Number of medium cases to include. Defaults to 1.
        num_hard: Number of hard cases to include. Defaults to 2.

    Returns:
        Path to the generated file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    easy_cases = [c for c in _SYNTHETIC_CASES if c["difficulty"] == "easy"]
    medium_cases = [c for c in _SYNTHETIC_CASES if c["difficulty"] == "medium"]
    hard_cases = [c for c in _SYNTHETIC_CASES if c["difficulty"] == "hard"]

    selected = (
        easy_cases[:num_easy]
        + medium_cases[:num_medium]
        + hard_cases[:num_hard]
    )

    with open(output_path, "w", encoding="utf-8") as f:
        for case in selected:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    logger.info(
        "Generated %d synthetic cases -> %s (easy=%d, medium=%d, hard=%d)",
        len(selected), output_path,
        min(num_easy, len(easy_cases)),
        min(num_medium, len(medium_cases)),
        min(num_hard, len(hard_cases)),
    )
    return output_path


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate a batch by right-padding to the longest sequence.

    Args:
        batch: List of dicts with ``input_ids``, ``attention_mask``, and
            ``labels`` tensors.

    Returns:
        Dict with padded tensors, each of shape ``(batch_size, max_seq_len)``.
    """
    max_len = max(item["input_ids"].shape[0] for item in batch)

    input_ids_list: List[torch.Tensor] = []
    attention_mask_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []

    for item in batch:
        seq_len = item["input_ids"].shape[0]
        pad_len = max_len - seq_len

        if pad_len > 0:
            input_ids = torch.cat(
                [item["input_ids"], torch.full((pad_len,), 0, dtype=torch.long)]
            )
            attention_mask = torch.cat(
                [item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]
            )
            labels = torch.cat(
                [item["labels"], torch.full((pad_len,), -100, dtype=torch.long)]
            )
        else:
            input_ids = item["input_ids"]
            attention_mask = item["attention_mask"]
            labels = item["labels"]

        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(labels)

    return {
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(attention_mask_list),
        "labels": torch.stack(labels_list),
    }


def create_dataloader(
    data_path: str | Path,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int = 1,
    max_length: int = 2048,
    shuffle: bool = True,
    num_workers: int = 0,
    difficulty_filter: Optional[str] = None,
) -> DataLoader:
    """Create a DataLoader for medical SFT data.

    Args:
        data_path: Path to JSONL data file or directory.
        tokenizer: HuggingFace tokenizer.
        batch_size: Batch size per GPU. Defaults to 1.
        max_length: Maximum sequence length. Defaults to 2048.
        shuffle: Whether to shuffle data. Defaults to ``True``.
        num_workers: Number of DataLoader worker processes. Defaults to 0.
        difficulty_filter: Optional difficulty level filter
            (``"easy"``, ``"medium"``, ``"hard"``).

    Returns:
        A configured ``DataLoader`` instance.
    """
    dataset = MedicalDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=max_length,
        difficulty_filter=difficulty_filter,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
