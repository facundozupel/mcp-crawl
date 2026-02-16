"""
MCP Server - Crawl4AI Tools (HTTP Client)
Expone herramientas de web crawling/scraping delegando al contenedor Docker
de crawl4ai via su API REST. No requiere playwright ni browser local.
"""

from fastmcp import FastMCP
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
import httpx
import json
import re
import os


# ===========================
# Config
# ===========================

CRAWL4AI_BASE = os.environ.get("CRAWL4AI_BASE_URL", "http://157.180.72.189:11235")
HTTP_TIMEOUT = int(os.environ.get("CRAWL4AI_TIMEOUT", "120"))


# ===========================
# Modelos Pydantic
# ===========================

class MarkdownResult(BaseModel):
    url: str
    raw_markdown: str = ""
    fit_markdown: str = ""
    success: bool = True
    error: Optional[str] = None


class LinkData(BaseModel):
    href: str = ""
    text: str = ""
    title: str = ""
    base_domain: str = ""
    is_internal: bool = True


class LinksResult(BaseModel):
    url: str
    internal_links: List[LinkData] = []
    external_links: List[LinkData] = []
    total_links: int = 0


class MediaItem(BaseModel):
    src: str = ""
    alt: str = ""
    desc: str = ""
    score: Optional[float] = None
    media_type: str = "image"


class MediaResult(BaseModel):
    url: str
    images: List[MediaItem] = []
    videos: List[MediaItem] = []
    audios: List[MediaItem] = []
    total_media: int = 0


class TableData(BaseModel):
    headers: List[str] = []
    rows: List[List[str]] = []
    score: Optional[float] = None


class TablesResult(BaseModel):
    url: str
    tables: List[TableData] = []
    total_tables: int = 0


class ScreenshotResult(BaseModel):
    url: str
    screenshot_base64: str = ""


class StructuredResult(BaseModel):
    url: str
    extracted_data: List[Dict[str, Any]] = []


class FullCrawlResult(BaseModel):
    url: str
    markdown: str = ""
    links: Optional[LinksResult] = None
    media: Optional[MediaResult] = None
    tables: Optional[TablesResult] = None
    status_code: Optional[int] = None


# ===========================
# HTTP client global
# ===========================

_http_client: httpx.AsyncClient | None = None


def get_http() -> httpx.AsyncClient:
    """Obtiene o crea el httpx client global."""
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(
            base_url=CRAWL4AI_BASE,
            timeout=httpx.Timeout(HTTP_TIMEOUT),
        )
        print(f"[crawl4ai] HTTP client creado -> {CRAWL4AI_BASE}")
    return _http_client


mcp = FastMCP(
    name="Crawl4AI Tools",
    instructions=(
        "Servidor MCP con herramientas de web crawling y scraping. "
        "Delega al contenedor Docker de crawl4ai via API REST. "
        "Permite extraer markdown, links, media, tablas, datos estructurados (CSS/XPath), "
        "screenshots y extraccion con LLM de una o multiples URLs."
    ),
)


# ===========================
# Helpers
# ===========================

def limpiar_markdown(texto: str) -> str:
    """Limpia mensajes de debug de crawl4ai del markdown."""
    texto_limpio = re.sub(r'\[(?:INIT|FETCH|SCRAPE|COMPLETE)\][^\n]*\n?', '', texto)
    texto_limpio = re.sub(r'\n\s*\n', '\n\n', texto_limpio)
    return texto_limpio.strip()


def parse_links(links_dict: dict) -> tuple[List[LinkData], List[LinkData]]:
    """Convierte result.links a listas tipadas de internos y externos."""
    internal = []
    external = []
    for link in links_dict.get("internal", []):
        internal.append(LinkData(
            href=link.get("href", ""),
            text=link.get("text", ""),
            title=link.get("title", ""),
            base_domain=link.get("base_domain", ""),
            is_internal=True,
        ))
    for link in links_dict.get("external", []):
        external.append(LinkData(
            href=link.get("href", ""),
            text=link.get("text", ""),
            title=link.get("title", ""),
            base_domain=link.get("base_domain", ""),
            is_internal=False,
        ))
    return internal, external


def parse_media(media_dict: dict) -> tuple[List[MediaItem], List[MediaItem], List[MediaItem]]:
    """Convierte result.media a listas tipadas por tipo."""
    images, videos, audios = [], [], []
    for img in media_dict.get("images", []):
        images.append(MediaItem(
            src=img.get("src", ""),
            alt=img.get("alt", ""),
            desc=img.get("desc", ""),
            score=img.get("score"),
            media_type="image",
        ))
    for vid in media_dict.get("videos", []):
        videos.append(MediaItem(
            src=vid.get("src", ""),
            alt=vid.get("alt", ""),
            desc=vid.get("desc", ""),
            score=vid.get("score"),
            media_type="video",
        ))
    for aud in media_dict.get("audios", []):
        audios.append(MediaItem(
            src=aud.get("src", ""),
            alt=aud.get("alt", ""),
            desc=aud.get("desc", ""),
            score=aud.get("score"),
            media_type="audio",
        ))
    return images, videos, audios


def parse_tables(tables_list: list) -> List[TableData]:
    """Convierte tablas crudas a list[TableData]."""
    parsed = []
    for table in tables_list:
        if isinstance(table, dict):
            parsed.append(TableData(
                headers=table.get("headers", []),
                rows=table.get("rows", []),
                score=table.get("score"),
            ))
        elif isinstance(table, str):
            parsed.append(TableData(headers=[], rows=[[table]], score=None))
    return parsed


def extract_crawl_result(data: dict, url: str) -> dict:
    """Extrae el CrawlResult de una respuesta /crawl para una URL dada."""
    results = data.get("results", [])
    for r in results:
        if r.get("url", "").rstrip("/") == url.rstrip("/") or True:
            return r
    return results[0] if results else {}


# ===========================
# Tools
# ===========================

@mcp.tool()
async def crawl_markdown(
    urls: List[str],
    fit_markdown: bool = False,
) -> List[dict]:
    """
    Extrae el contenido markdown de una o mas URLs.

    Args:
        urls: Lista de URLs a crawlear.
        fit_markdown: Si True, devuelve tambien el markdown filtrado/limpio (fit_markdown).

    Returns:
        Lista de resultados con raw_markdown y opcionalmente fit_markdown por cada URL.
    """
    http = get_http()
    results = []
    for url in urls:
        try:
            filter_type = "fit" if fit_markdown else "raw"
            resp = await http.post("/md", json={"url": url, "f": filter_type})
            resp.raise_for_status()
            data = resp.json()
            if data.get("success"):
                md_text = limpiar_markdown(data.get("markdown", ""))
                raw = md_text if not fit_markdown else ""
                fit = md_text if fit_markdown else ""
                # Si pidieron fit, tambien traer raw
                if fit_markdown:
                    resp_raw = await http.post("/md", json={"url": url, "f": "raw"})
                    if resp_raw.status_code == 200:
                        raw_data = resp_raw.json()
                        raw = limpiar_markdown(raw_data.get("markdown", ""))
                results.append(MarkdownResult(
                    url=url, raw_markdown=raw, fit_markdown=fit,
                ).model_dump())
            else:
                results.append(MarkdownResult(
                    url=url, success=False, error="Crawl failed",
                ).model_dump())
        except Exception as e:
            results.append(MarkdownResult(
                url=url, success=False, error=str(e),
            ).model_dump())
    return results


@mcp.tool()
async def crawl_links(
    urls: List[str],
    include_external: bool = True,
) -> List[dict]:
    """
    Extrae los links internos y externos de una o mas URLs.

    Args:
        urls: Lista de URLs a crawlear.
        include_external: Si True, incluye links externos. Default True.

    Returns:
        Lista con links internos/externos encontrados por URL.
    """
    http = get_http()
    results = []
    resp = await http.post("/crawl", json={"urls": urls})
    resp.raise_for_status()
    data = resp.json()
    for r in data.get("results", []):
        url = r.get("url", "")
        links_raw = r.get("links", {})
        if links_raw:
            internal, external = parse_links(links_raw)
            if not include_external:
                external = []
            total = len(internal) + len(external)
            results.append(LinksResult(
                url=url, internal_links=internal,
                external_links=external, total_links=total,
            ).model_dump())
        else:
            results.append(LinksResult(url=url).model_dump())
    return results


@mcp.tool()
async def crawl_media(
    urls: List[str],
    media_types: List[str] = None,
) -> List[dict]:
    """
    Extrae imagenes, videos y audios de una o mas URLs.

    Args:
        urls: Lista de URLs a crawlear.
        media_types: Filtro de tipos: ["images", "videos", "audios"]. None = todos.

    Returns:
        Lista con media encontrada por URL.
    """
    if media_types is None:
        media_types = ["images", "videos", "audios"]
    http = get_http()
    resp = await http.post("/crawl", json={"urls": urls})
    resp.raise_for_status()
    data = resp.json()
    results = []
    for r in data.get("results", []):
        url = r.get("url", "")
        media_raw = r.get("media", {})
        if media_raw:
            images, videos, audios = parse_media(media_raw)
            if "images" not in media_types:
                images = []
            if "videos" not in media_types:
                videos = []
            if "audios" not in media_types:
                audios = []
            total = len(images) + len(videos) + len(audios)
            results.append(MediaResult(
                url=url, images=images, videos=videos,
                audios=audios, total_media=total,
            ).model_dump())
        else:
            results.append(MediaResult(url=url).model_dump())
    return results


@mcp.tool()
async def crawl_tables(
    urls: List[str],
) -> List[dict]:
    """
    Extrae tablas HTML estructuradas de una o mas URLs.

    Args:
        urls: Lista de URLs a crawlear.

    Returns:
        Lista con tablas encontradas por URL, cada tabla con headers y rows.
    """
    http = get_http()
    resp = await http.post("/crawl", json={"urls": urls})
    resp.raise_for_status()
    data = resp.json()
    results = []
    for r in data.get("results", []):
        url = r.get("url", "")
        raw_tables = r.get("tables", [])
        if isinstance(raw_tables, str):
            try:
                raw_tables = json.loads(raw_tables)
            except (json.JSONDecodeError, TypeError):
                raw_tables = []
        tables = parse_tables(raw_tables) if isinstance(raw_tables, list) else []
        results.append(TablesResult(
            url=url, tables=tables, total_tables=len(tables),
        ).model_dump())
    return results


@mcp.tool()
async def crawl_screenshot(
    url: str,
    screenshot_wait_for: float = 2,
) -> dict:
    """
    Toma un screenshot de una URL y lo devuelve en base64.

    Args:
        url: URL a capturar.
        screenshot_wait_for: Segundos de espera antes de capturar. Default 2.

    Returns:
        Objeto con url y screenshot_base64.
    """
    http = get_http()
    try:
        resp = await http.post("/screenshot", json={
            "url": url,
            "screenshot_wait_for": screenshot_wait_for,
        })
        resp.raise_for_status()
        data = resp.json()
        b64 = data.get("screenshot", "") if data.get("success") else ""
        return ScreenshotResult(url=url, screenshot_base64=b64).model_dump()
    except Exception as e:
        return ScreenshotResult(url=url).model_dump()


@mcp.tool()
async def crawl_structured_css(
    urls: List[str],
    schema: dict,
) -> List[dict]:
    """
    Extrae datos estructurados de URLs usando selectores CSS.

    Args:
        urls: Lista de URLs a crawlear.
        schema: Schema de extraccion CSS para JsonCssExtractionStrategy.
            IMPORTANTE: No usar ":self" como selector, no es soportado.
            Para extraer texto/atributos del propio baseSelector, omitir el campo "selector" en fields.

            Ejemplo links: {"baseSelector": "a[href]", "fields": [{"name": "text", "type": "text"}, {"name": "href", "type": "attribute", "attribute": "href"}]}
            Ejemplo productos: {"baseSelector": "div.product", "fields": [{"name": "title", "selector": "h2", "type": "text"}, {"name": "price", "selector": ".price", "type": "text"}]}

    Returns:
        Lista de datos extraidos por URL segun el schema CSS.
    """
    http = get_http()
    crawler_config = {
        "extraction_strategy": {
            "type": "JsonCssExtractionStrategy",
            "params": {"schema": schema},
        }
    }
    resp = await http.post("/crawl", json={"urls": urls, "crawler_config": crawler_config})
    resp.raise_for_status()
    data = resp.json()
    results = []
    for r in data.get("results", []):
        url = r.get("url", "")
        extracted = r.get("extracted_content")
        parsed = []
        if extracted:
            if isinstance(extracted, str):
                try:
                    extracted = json.loads(extracted)
                except (json.JSONDecodeError, TypeError):
                    extracted = [{"raw": extracted}]
            if isinstance(extracted, list):
                parsed = extracted
            elif isinstance(extracted, dict):
                parsed = [extracted]
        results.append(StructuredResult(url=url, extracted_data=parsed).model_dump())
    return results


@mcp.tool()
async def crawl_structured_xpath(
    urls: List[str],
    schema: dict,
) -> List[dict]:
    """
    Extrae datos estructurados de URLs usando selectores XPath.

    Args:
        urls: Lista de URLs a crawlear.
        schema: Schema de extraccion XPath para JsonXPathExtractionStrategy.
            Para extraer texto/atributos del propio baseSelector, omitir "selector" en fields.

            Ejemplo links: {"baseSelector": "//a[@href]", "fields": [{"name": "text", "type": "text"}, {"name": "href", "type": "attribute", "attribute": "href"}]}
            Ejemplo productos: {"baseSelector": "//div[@class='product']", "fields": [{"name": "title", "selector": ".//h2/text()", "type": "text"}]}

    Returns:
        Lista de datos extraidos por URL segun el schema XPath.
    """
    http = get_http()
    crawler_config = {
        "extraction_strategy": {
            "type": "JsonXPathExtractionStrategy",
            "params": {"schema": schema},
        }
    }
    resp = await http.post("/crawl", json={"urls": urls, "crawler_config": crawler_config})
    resp.raise_for_status()
    data = resp.json()
    results = []
    for r in data.get("results", []):
        url = r.get("url", "")
        extracted = r.get("extracted_content")
        parsed = []
        if extracted:
            if isinstance(extracted, str):
                try:
                    extracted = json.loads(extracted)
                except (json.JSONDecodeError, TypeError):
                    extracted = [{"raw": extracted}]
            if isinstance(extracted, list):
                parsed = extracted
            elif isinstance(extracted, dict):
                parsed = [extracted]
        results.append(StructuredResult(url=url, extracted_data=parsed).model_dump())
    return results


@mcp.tool()
async def crawl_with_js(
    url: str,
    js_code: str,
    wait_for: str = None,
) -> dict:
    """
    Crawlea una URL ejecutando JavaScript personalizado antes de extraer contenido.

    Args:
        url: URL a crawlear.
        js_code: Codigo JavaScript a ejecutar en la pagina. Ej: "document.querySelector('.load-more').click();"
        wait_for: Selector CSS opcional para esperar antes de extraer. Ej: "div.results"

    Returns:
        Markdown resultado tras ejecutar el JS, junto con links y media disponibles.
    """
    http = get_http()
    try:
        # /execute_js devuelve un CrawlResult completo con markdown
        resp = await http.post("/execute_js", json={
            "url": url,
            "scripts": [js_code],
        })
        resp.raise_for_status()
        data = resp.json()
        if data.get("success"):
            md = data.get("markdown", {})
            raw = limpiar_markdown(md.get("raw_markdown", "") if isinstance(md, dict) else str(md))
            return MarkdownResult(url=url, raw_markdown=raw).model_dump()
        else:
            return MarkdownResult(
                url=url, success=False,
                error=data.get("error_message", "JS execution failed"),
            ).model_dump()
    except Exception as e:
        return MarkdownResult(url=url, success=False, error=str(e)).model_dump()


@mcp.tool()
async def crawl_full(
    urls: List[str],
) -> List[dict]:
    """
    Crawl completo: extrae markdown, links, media y tablas de una o mas URLs.

    Args:
        urls: Lista de URLs a crawlear.

    Returns:
        Lista con resultado completo (markdown, links, media, tablas, status_code) por URL.
    """
    http = get_http()
    resp = await http.post("/crawl", json={"urls": urls})
    resp.raise_for_status()
    data = resp.json()
    results = []
    for r in data.get("results", []):
        url = r.get("url", "")

        # Markdown
        md = r.get("markdown", {})
        raw = ""
        if isinstance(md, dict):
            raw = limpiar_markdown(md.get("raw_markdown", ""))
        elif isinstance(md, str):
            raw = limpiar_markdown(md)

        # Links
        links_raw = r.get("links", {})
        internal, external = parse_links(links_raw) if links_raw else ([], [])
        links_result = LinksResult(
            url=url, internal_links=internal,
            external_links=external,
            total_links=len(internal) + len(external),
        )

        # Media
        media_raw = r.get("media", {})
        images, videos, audios = parse_media(media_raw) if media_raw else ([], [], [])
        media_result = MediaResult(
            url=url, images=images, videos=videos,
            audios=audios,
            total_media=len(images) + len(videos) + len(audios),
        )

        # Tables
        raw_tables = r.get("tables", [])
        if isinstance(raw_tables, str):
            try:
                raw_tables = json.loads(raw_tables)
            except (json.JSONDecodeError, TypeError):
                raw_tables = []
        tables = parse_tables(raw_tables) if isinstance(raw_tables, list) else []
        tables_result = TablesResult(
            url=url, tables=tables, total_tables=len(tables),
        )

        results.append(FullCrawlResult(
            url=url, markdown=raw,
            links=links_result, media=media_result,
            tables=tables_result,
            status_code=r.get("status_code"),
        ).model_dump())
    return results


@mcp.tool()
async def crawl_schema_org(
    urls: List[str],
) -> List[dict]:
    """
    Extrae los datos estructurados JSON-LD (schema.org) de una o mas URLs.
    Busca todos los <script type="application/ld+json"> en el HTML.

    Args:
        urls: Lista de URLs de las que extraer schema.org.

    Returns:
        Lista con los objetos JSON-LD encontrados por URL.
    """
    http = get_http()
    results = []
    for url in urls:
        try:
            resp = await http.post("/execute_js", json={
                "url": url,
                "scripts": [],
            })
            resp.raise_for_status()
            data = resp.json()
            html = data.get("html", "")
            # Extraer todos los <script type="application/ld+json">
            scripts = re.findall(
                r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
                html,
                re.DOTALL,
            )
            schemas = []
            for script in scripts:
                try:
                    parsed = json.loads(script.strip())
                    schemas.append(parsed)
                except (json.JSONDecodeError, TypeError):
                    schemas.append({"raw": script.strip()})
            print(f"[crawl4ai] {url} -> {len(schemas)} JSON-LD encontrados")
            results.append({
                "url": url,
                "schemas": schemas,
                "total": len(schemas),
            })
        except Exception as e:
            print(f"[crawl4ai] Error schema_org {url}: {e}")
            results.append({"url": url, "schemas": [], "total": 0, "error": str(e)})
    return results


# @mcp.tool()
# async def extract_with_llm(
#     urls: List[str],
#     instruction: str,
#     schema: str = None,
#     provider: str = None,
# ) -> List[dict]:
#     """
#     Extrae datos estructurados de URLs usando un LLM via el contenedor crawl4ai.
#     Requiere API key del LLM configurada en el contenedor.
#     Descomentar cuando se quiera habilitar.
#     """
#     pass


# ===========================
# Entry point
# ===========================

HOST = os.environ.get("MCP_HOST", "0.0.0.0")
PORT = int(os.environ.get("MCP_PORT", "8080"))

cors = Middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    allow_credentials=False,
    expose_headers=["*"],
    max_age=86400,
)


async def main():
    await mcp.run_http_async(
        transport="streamable-http",
        host=HOST,
        port=PORT,
        path="/mcp",
        middleware=[cors],
        show_banner=True,
    )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
