import re


_ASSISTANT_IDENTITY_PATTERNS = (
    re.compile(
        r"(?:你|您|这个助手|知识库助手).{0,4}(?:是|属于|用|使用|基于|调用|运行).{0,6}(?:什么|哪个|哪种|何种)?(?:大)?模型",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:当前|现在)(?:回答)?(?:所用|使用|选择|运行|用)?的?(?:大)?模型(?:是|叫|为)?(?:什么|哪个|哪种|何种)",
        re.IGNORECASE,
    ),
    re.compile(r"(?:你|您|这个助手|知识库助手)(?:到底)?是谁", re.IGNORECASE),
    re.compile(r"(?:介绍|说明)(?:一下)?(?:你|您|这个助手|知识库助手)(?:自己)?", re.IGNORECASE),
    re.compile(r"\b(?:what|which)\s+(?:ai\s+|language\s+)?model\s+(?:are|do)\s+you\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+(?:ai\s+|language\s+)?model\s+is\s+this\b", re.IGNORECASE),
    re.compile(r"\bwho\s+are\s+you\b", re.IGNORECASE),
)


def is_assistant_identity_question(question: str) -> bool:
    """Return whether the user is asking about this assistant or its active model."""
    normalized = " ".join(question.strip().split())
    return any(pattern.search(normalized) for pattern in _ASSISTANT_IDENTITY_PATTERNS)
