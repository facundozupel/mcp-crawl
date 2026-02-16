# mcp-crawl

MCP server for web crawling and scraping via [Crawl4AI](https://github.com/unclecode/crawl4ai) Docker API.

Built with [FastMCP](https://github.com/jlowin/fastmcp). No browser or Playwright needed locally — all crawling is delegated to a remote Crawl4AI container.

## Tools

| Tool | Description |
|---|---|
| `crawl_markdown` | Extract markdown from one or more URLs |
| `crawl_links` | Extract internal/external links |
| `crawl_media` | Extract images, videos, audios |
| `crawl_tables` | Extract structured HTML tables |
| `crawl_screenshot` | Take a screenshot (base64) |
| `crawl_structured_css` | Extract data using CSS selectors |
| `crawl_structured_xpath` | Extract data using XPath selectors |
| `crawl_with_js` | Crawl after executing custom JavaScript |
| `crawl_full` | Full crawl: markdown + links + media + tables |
| `extract_with_llm` | Extract structured data using an LLM |

## Prerequisites

A running Crawl4AI Docker container:

```bash
docker run -d -p 11235:11235 unclecode/crawl4ai
```

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python mcp_crawl4ai.py
```

The MCP server starts on `http://0.0.0.0:8080/mcp` (streamable HTTP).

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `CRAWL4AI_BASE_URL` | `http://157.180.72.189:11235` | Crawl4AI container URL |
| `CRAWL4AI_TIMEOUT` | `120` | HTTP timeout in seconds |
| `MCP_HOST` | `0.0.0.0` | MCP server host |
| `MCP_PORT` | `8080` | MCP server port |

## Register in Claude Code

```bash
claude mcp add crawl4ai --transport http http://localhost:8080/mcp
```

## License

MIT
