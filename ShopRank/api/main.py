"""FastAPI serving the fast lexical candidate generator."""

try:
    from fastapi import FastAPI
    from pydantic import BaseModel, Field
except ImportError as error:
    raise RuntimeError("Install API extras: pip install -e '.[api]'") from error

from shoprank.pipeline import search_demo


app = FastAPI(title="ShopRank API", version="0.1.0")


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=200)
    limit: int = Field(default=10, ge=1, le=20)


class SearchResult(BaseModel):
    product_id: str
    title: str
    relevance_score: float


class SearchResponse(BaseModel):
    query: str
    products: list[SearchResult]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    products = [
        SearchResult(
            product_id=item["product_id"],
            title=item["title"],
            relevance_score=round(item["relevance_score"], 6),
        )
        for item in search_demo(request.query, request.limit)
    ]
    return SearchResponse(query=request.query, products=products)
