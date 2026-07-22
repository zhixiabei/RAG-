from uuid import NAMESPACE_URL, uuid5


def vector_point_id(chunk_id: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"rag-chunk:{chunk_id}"))
