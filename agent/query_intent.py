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

_KNOWLEDGE_CATALOG_PATTERNS = (
    re.compile(r"(?:文件夹|目录|路径|子目录)", re.IGNORECASE),
    re.compile(r"(?:有哪些|列出|查看|显示|所有|多少).{0,10}(?:文件|文档)", re.IGNORECASE),
    re.compile(r"(?:文件|文档).{0,10}(?:有哪些|清单|列表|目录)", re.IGNORECASE),
    re.compile(r"\b(?:folder|directory|path|file\s+list|document\s+list)\b", re.IGNORECASE),
)

_KNOWLEDGE_CATALOG_INVENTORY_PATTERNS = (
    re.compile(r"(?:文件夹|目录).{0,12}(?:有哪些|有什么|有啥|包含哪些|列出|多少(?:个)?).{0,8}(?:文件|文档|资料)?", re.IGNORECASE),
    re.compile(r"(?:有哪些|有什么|有啥|列出|显示|多少(?:个)?).{0,10}(?:文件|文档)", re.IGNORECASE),
    re.compile(r"(?:文件|文档).{0,10}(?:清单|列表|有哪些|有多少)", re.IGNORECASE),
    re.compile(r"(?:能否|能不能|可以|可否|能|是否).{0,8}(?:找到|查到|检索到|搜到|存在).{0,8}(?:文件|文档)", re.IGNORECASE),
    re.compile(r"(?:文件|文档).{0,12}(?:存在吗|在吗|有没有|找得到吗|能找到吗)", re.IGNORECASE),
    re.compile(r"\b(?:list|show|count)\b.{0,20}\b(?:files?|documents?)\b", re.IGNORECASE),
)

_KNOWLEDGE_CATALOG_FILE_LOOKUP_PATTERNS = (
    re.compile(r"(?:能否|能不能|可以|可否|能|是否).{0,8}(?:找到|查到|检索到|搜到|存在).{0,8}(?:文件|文档)"),
    re.compile(r"(?:文件|文档).{0,12}(?:存在吗|在吗|有没有|找得到吗|能找到吗)"),
)

_KNOWLEDGE_CATALOG_CONTENT_ACTIONS = re.compile(
    r"(?:总结|概括|归纳|分析|对比|提取|解读|正文|主要内容|讲了什么|说明什么)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class QueryIntent:
    '''Deterministic routing facts computed for one user question.'''

    assistant_identity: bool
    needs_catalog: bool
    catalog_inventory: bool
    catalog_file_lookup: bool

    @property
    def skips_retrieval(self) -> bool:
        return self.assistant_identity or self.catalog_inventory


def analyze_query_intent(question: str) -> QueryIntent:
    '''Classify all keyword intents in one normalized pass.'''
    normalized = ' '.join(question.strip().split())
    return QueryIntent(
        assistant_identity=any(
            pattern.search(normalized) for pattern in _ASSISTANT_IDENTITY_PATTERNS
        ),
        needs_catalog=any(
            pattern.search(normalized) for pattern in _KNOWLEDGE_CATALOG_PATTERNS
        ),
        catalog_inventory=(
            not _KNOWLEDGE_CATALOG_CONTENT_ACTIONS.search(normalized)
            and any(
                pattern.search(normalized)
                for pattern in _KNOWLEDGE_CATALOG_INVENTORY_PATTERNS
            )
        ),
        catalog_file_lookup=any(
            pattern.search(normalized)
            for pattern in _KNOWLEDGE_CATALOG_FILE_LOOKUP_PATTERNS
        ),
    )


def is_assistant_identity_question(question: str) -> bool:
    '''Return whether the user asks about this assistant or its model.'''
    return analyze_query_intent(question).assistant_identity


def needs_knowledge_catalog(question: str) -> bool:
    '''Return whether answering requires knowledge-base metadata.'''
    return analyze_query_intent(question).needs_catalog


def is_knowledge_catalog_inventory_question(question: str) -> bool:
    '''Return whether the question asks for a deterministic file listing.'''
    return analyze_query_intent(question).catalog_inventory


def is_knowledge_catalog_file_lookup_question(question: str) -> bool:
    '''Return whether the question asks if a file exists in the catalog.'''
    return analyze_query_intent(question).catalog_file_lookup
