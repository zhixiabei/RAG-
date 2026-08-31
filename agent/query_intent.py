from dataclasses import dataclass
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

_CONVERSATION_ONLY_PATTERNS = (
    re.compile(
        r"^(?:你好|您好|嗨|哈喽|hello|hi|hey|谢谢|谢谢你|感谢|多谢|好的|好|明白了|知道了|收到|再见|拜拜)[！!。.\s]*$",
        re.IGNORECASE,
    ),
)

_KNOWLEDGE_CATALOG_FILE_LOOKUP_PATTERNS = (
    re.compile(r"(?:能否|能不能|可以|可否|能|是否).{0,8}(?:找到|查到|检索到|搜到|存在).{0,8}(?:文件|文档)"),
    re.compile(r"(?:文件|文档).{0,12}(?:存在吗|在吗|有没有|找得到吗|能找到吗)"),
)


@dataclass(frozen=True)
class QueryIntent:
    '''Deterministic routing facts computed for one user question.'''

    assistant_identity: bool
    conversation_only: bool
    catalog_file_lookup: bool

    @property
    def skips_retrieval(self) -> bool:
        return self.assistant_identity or self.conversation_only or self.catalog_file_lookup


def analyze_query_intent(question: str) -> QueryIntent:
    '''Classify all keyword intents in one normalized pass.'''
    normalized = ' '.join(question.strip().split())
    return QueryIntent(
        assistant_identity=any(
            pattern.search(normalized) for pattern in _ASSISTANT_IDENTITY_PATTERNS
        ),
        conversation_only=any(
            pattern.search(normalized) for pattern in _CONVERSATION_ONLY_PATTERNS
        ),
        catalog_file_lookup=any(
            pattern.search(normalized)
            for pattern in _KNOWLEDGE_CATALOG_FILE_LOOKUP_PATTERNS
        ),
    )


def is_assistant_identity_question(question: str) -> bool:
    '''Return whether the user asks about this assistant or its model.'''
    return analyze_query_intent(question).assistant_identity


def is_knowledge_catalog_file_lookup_question(question: str) -> bool:
    '''Return whether the question asks if a file exists in the catalog.'''
    return analyze_query_intent(question).catalog_file_lookup
