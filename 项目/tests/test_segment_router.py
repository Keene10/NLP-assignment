from rag.retriever.rag import HierarchicalRAGService, RetrievedPage
from rag.retriever.segment_router import SegmentRouter


class UnavailableLLM:
    available = False


class FailingLLM:
    available = True

    def generate(self, prompt):
        raise RuntimeError("route failed")


LONG_GLD_FILENAME = "广联达-再谈广联达当前时点下如何看待其三条增长曲线-220217131页.pdf"


def test_multi_segment_route_returns_none_when_llm_unavailable():
    router = SegmentRouter(llm=UnavailableLLM())

    assert router.route("AECORE平台服务模块有哪些作用？", LONG_GLD_FILENAME) is None


def test_multi_segment_route_returns_none_when_llm_fails():
    router = SegmentRouter(llm=FailingLLM())

    assert router.route("AECORE平台服务模块有哪些作用？", LONG_GLD_FILENAME) is None


class NoneRouter:
    def route(self, query, filename):
        return None


class BaseRAGStub:
    def retrieve_pages(self, query, **kwargs):
        return [
            RetrievedPage(
                filename=LONG_GLD_FILENAME,
                page_number=97,
                score=1.0,
                hit_count=1,
                chunk_ids=[1],
                content="AECORE平台服务模块",
            )
        ]


def test_hierarchical_service_falls_back_to_flat_when_route_is_none():
    service = object.__new__(HierarchicalRAGService)
    service.base = BaseRAGStub()
    service.router = NoneRouter()

    pages, route_info = service.retrieve_pages_hierarchical(
        "AECORE平台服务模块有哪些作用？",
        filename=LONG_GLD_FILENAME,
    )

    assert pages[0].page_number == 97
    assert route_info["hierarchical"] is False
    assert route_info["segment"] is None
