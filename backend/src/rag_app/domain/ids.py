from uuid import NAMESPACE_URL, uuid5


def vector_point_id(chunk_id: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"rag-chunk:{chunk_id}"))


def document_vector_point_id(document_id: str, route_kind: str | None = None) -> str:
    suffix = "" if not route_kind else f":{route_kind}"
    return str(uuid5(NAMESPACE_URL, f"rag-document:{document_id}{suffix}"))