"""
pipeline/pipeline.py — 知识库采集分析流水线

三阶段流水线：采集 → 分析 → 整理
从 V2/V3 继承，V4 新增每日简报自动发布。
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# 目录常量
RAW_DIR = Path("knowledge/raw")
ARTICLES_DIR = Path("knowledge/articles")


# ============================================================
# Stage 1: 采集（Collector）
# ============================================================

def collect_github_trending(limit: int = 20) -> list[dict]:
    """采集 GitHub Trending 数据。

    通过 GitHub API 获取当日 trending 仓库信息。
    实际部署时需要 GITHUB_TOKEN 环境变量。

    Args:
        limit: 最大采集数量

    Returns:
        原始数据条目列表
    """
    import urllib.request
    import urllib.error

    token = os.environ.get("GITHUB_TOKEN", "")
    today = datetime.now().strftime("%Y-%m-%d")
    date_query = (datetime.now()).strftime("%Y-%m-%d")

    # GitHub Search API — 按 stars 排序近期创建/更新的项目
    url = (
        f"https://api.github.com/search/repositories"
        f"?q=topic:llm+topic:agent+pushed:>{date_query}"
        f"&sort=stars&order=desc&per_page={limit}"
    )

    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "ai-knowledge-base/4.0",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError) as e:
        logger.error(f"[Collector] GitHub API 请求失败: {e}")
        return []

    items = []
    for repo in data.get("items", [])[:limit]:
        items.append({
            "id": f"github-{repo['full_name'].replace('/', '-')}",
            "title": f"{repo['full_name']} — {repo.get('description', '')}",
            "source": "GitHub Trending",
            "url": repo["html_url"],
            "collected_at": datetime.now().isoformat(),
            "stars": repo.get("stargazers_count", 0),
            "language": repo.get("language", ""),
            "topics": repo.get("topics", []),
            "raw_data": {
                "full_name": repo["full_name"],
                "created_at": repo.get("created_at", ""),
                "updated_at": repo.get("updated_at", ""),
            }
        })

    # 保存原始数据
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    output_file = RAW_DIR / f"github-trending-{today}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    logger.info(f"[Collector] GitHub Trending: 采集 {len(items)} 条 → {output_file}")
    return items


def collect_hackernews(limit: int = 20) -> list[dict]:
    """采集 Hacker News Top Stories。

    使用 HN Firebase API（无需认证）。

    Args:
        limit: 最大采集数量

    Returns:
        原始数据条目列表
    """
    import urllib.request
    import urllib.error

    today = datetime.now().strftime("%Y-%m-%d")

    try:
        # 获取 Top Story IDs
        url = "https://hacker-news.firebaseio.com/v0/topstories.json"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=15) as resp:
            story_ids = json.loads(resp.read().decode("utf-8"))[:limit]
    except (urllib.error.URLError, TimeoutError) as e:
        logger.error(f"[Collector] HN API 请求失败: {e}")
        return []

    items = []
    for sid in story_ids:
        try:
            item_url = f"https://hacker-news.firebaseio.com/v0/item/{sid}.json"
            req = urllib.request.Request(item_url)
            with urllib.request.urlopen(req, timeout=10) as resp:
                story = json.loads(resp.read().decode("utf-8"))

            if not story or story.get("type") != "story":
                continue

            items.append({
                "id": f"hn-{sid}",
                "title": story.get("title", ""),
                "source": "Hacker News",
                "url": story.get("url", f"https://news.ycombinator.com/item?id={sid}"),
                "collected_at": datetime.now().isoformat(),
                "score": story.get("score", 0),
                "raw_data": {
                    "hn_id": sid,
                    "by": story.get("by", ""),
                    "time": story.get("time", 0),
                    "descendants": story.get("descendants", 0),
                }
            })
        except Exception as e:
            logger.warning(f"[Collector] HN item {sid} 失败: {e}")
            continue

    # 保存原始数据
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    output_file = RAW_DIR / f"hackernews-top-{today}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    logger.info(f"[Collector] Hacker News: 采集 {len(items)} 条 → {output_file}")
    return items


# ============================================================
# Stage 2: 分析（Analyzer）
# ============================================================

# AI/LLM 相关关键词（用于相关性评分）
AI_KEYWORDS = {
    "llm", "large language model", "agent", "rag", "mcp",
    "transformer", "gpt", "claude", "gemini", "deepseek",
    "fine-tuning", "prompt", "embedding", "vector", "retrieval",
    "multi-agent", "tool-use", "function-calling", "reasoning",
    "openai", "anthropic", "google", "meta", "mistral",
}


def analyze_item(item: dict) -> dict:
    """分析单个原始条目，生成摘要和相关性评分。

    基于关键词匹配的简单分析器。生产环境可替换为 LLM 调用。

    Args:
        item: 原始数据条目

    Returns:
        增强后的条目（新增 summary, tags, relevance_score）
    """
    title = item.get("title", "").lower()
    text = title + " " + json.dumps(item.get("raw_data", {})).lower()

    # 关键词匹配计算相关性
    matched = [kw for kw in AI_KEYWORDS if kw in text]
    score = min(len(matched) / 5.0, 1.0)  # 5 个关键词满分

    # GitHub 项目：stars 加权
    stars = item.get("stars", 0)
    if stars > 1000:
        score = min(score + 0.2, 1.0)
    elif stars > 100:
        score = min(score + 0.1, 1.0)

    # HN 项目：score 加权
    hn_score = item.get("score", 0)
    if hn_score > 200:
        score = min(score + 0.2, 1.0)
    elif hn_score > 50:
        score = min(score + 0.1, 1.0)

    # 生成标签
    tags = []
    tag_map = {
        "llm": ["llm", "large language model", "gpt", "claude", "gemini", "deepseek"],
        "agent": ["agent", "multi-agent", "tool-use", "function-calling"],
        "rag": ["rag", "retrieval", "vector", "embedding"],
        "mcp": ["mcp", "model context protocol"],
        "reasoning": ["reasoning", "chain-of-thought", "o1", "o3"],
        "fine-tuning": ["fine-tuning", "fine-tune", "lora", "qlora"],
    }
    for tag, keywords in tag_map.items():
        if any(kw in text for kw in keywords):
            tags.append(tag)

    # 自动摘要（简单截取，生产环境用 LLM 生成）
    summary = item.get("title", "")
    if item.get("raw_data", {}).get("full_name"):
        summary = f"GitHub 项目 {item['raw_data']['full_name']}：{item['title'].split('—')[-1].strip()}"

    item["summary"] = summary
    item["tags"] = tags if tags else ["general"]
    item["relevance_score"] = round(score, 2)

    return item


def analyze_batch(raw_file: str) -> list[dict]:
    """批量分析原始数据文件中的所有条目。

    Args:
        raw_file: 原始数据 JSON 文件路径

    Returns:
        分析后的条目列表
    """
    with open(raw_file, "r", encoding="utf-8") as f:
        items = json.load(f)

    analyzed = []
    for item in items:
        enriched = analyze_item(item)
        analyzed.append(enriched)

    logger.info(f"[Analyzer] 分析完成：{len(analyzed)} 条，来源 {raw_file}")
    return analyzed


# ============================================================
# Stage 3: 整理（Organizer）
# ============================================================

def organize_articles(analyzed_items: list[dict], min_score: float = 0.6) -> list[dict]:
    """整理分析后的条目，过滤低质量内容，输出为标准知识条目。

    Args:
        analyzed_items: 已分析的条目列表
        min_score: 最低相关性分数阈值

    Returns:
        整理后的知识条目列表
    """
    ARTICLES_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")

    organized = []
    seen_urls = set()

    # 加载已有文章的 URL 做去重
    for existing in ARTICLES_DIR.glob("*.json"):
        if existing.name == "index.json":
            continue
        try:
            with open(existing, "r", encoding="utf-8") as f:
                data = json.load(f)
                seen_urls.add(data.get("url", ""))
        except (json.JSONDecodeError, OSError):
            continue

    for item in analyzed_items:
        # 质量门控
        if item.get("relevance_score", 0) < min_score:
            continue

        # URL 去重
        url = item.get("url", "")
        if url in seen_urls:
            continue
        seen_urls.add(url)

        # 生成 slug
        slug = item.get("id", "unknown").replace("/", "-").replace(" ", "-")[:50]
        filename = f"{today}-{slug}.json"

        # 标准化知识条目格式
        article = {
            "id": item["id"],
            "title": item["title"],
            "source": item["source"],
            "url": url,
            "collected_at": item["collected_at"],
            "summary": item.get("summary", ""),
            "tags": item.get("tags", []),
            "relevance_score": item.get("relevance_score", 0),
        }

        # 写入文件
        output_path = ARTICLES_DIR / filename
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(article, f, ensure_ascii=False, indent=2)

        organized.append(article)

    # 更新索引
    _update_index(organized)

    logger.info(f"[Organizer] 整理完成：{len(organized)} 条通过质量门控")
    return organized


def _update_index(new_articles: list[dict]):
    """更新知识库索引文件。"""
    index_file = ARTICLES_DIR / "index.json"

    # 加载现有索引
    if index_file.exists():
        with open(index_file, "r", encoding="utf-8") as f:
            index = json.load(f)
    else:
        index = {"articles": [], "updated_at": ""}

    # 追加新条目
    for article in new_articles:
        index["articles"].append({
            "id": article["id"],
            "title": article["title"],
            "date": article["collected_at"][:10],
            "tags": article["tags"],
            "score": article["relevance_score"],
        })

    index["updated_at"] = datetime.now().isoformat()

    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)


# ============================================================
# 完整流水线
# ============================================================

async def run_pipeline(publish: bool = True):
    """运行完整的采集→分析→整理流水线。

    V4 新增：流水线结束后自动发布每日简报。

    Args:
        publish: 是否在流水线完成后发布每日简报
    """
    logger.info("=" * 60)
    logger.info(f"[Pipeline] 开始执行 — {datetime.now().isoformat()}")
    logger.info("=" * 60)

    # Stage 1: 采集
    logger.info("[Pipeline] Stage 1/3: 采集")
    github_items = collect_github_trending(limit=20)
    hn_items = collect_hackernews(limit=20)
    all_raw = github_items + hn_items
    logger.info(f"[Pipeline] 采集完成，共 {len(all_raw)} 条原始数据")

    # Stage 2: 分析
    logger.info("[Pipeline] Stage 2/3: 分析")
    analyzed = [analyze_item(item) for item in all_raw]
    logger.info(f"[Pipeline] 分析完成，共 {len(analyzed)} 条")

    # Stage 3: 整理
    logger.info("[Pipeline] Stage 3/3: 整理")
    articles = organize_articles(analyzed, min_score=0.6)
    logger.info(f"[Pipeline] 整理完成，{len(articles)} 条通过质量门控")

    # V4 新增：自动发布每日简报
    if publish and articles:
        logger.info("[Pipeline] 发布每日简报...")
        try:
            from distribution.publisher import publish_daily_digest
            results = await publish_daily_digest()
            for r in results:
                status = "成功" if r.success else f"失败({r.error})"
                logger.info(f"[Pipeline] {r.channel}: {status}")
        except ImportError:
            logger.warning("[Pipeline] distribution 模块未安装，跳过发布")
        except Exception as e:
            logger.error(f"[Pipeline] 发布失败: {e}")

    logger.info("=" * 60)
    logger.info(f"[Pipeline] 完成 — 新增 {len(articles)} 条知识条目")
    logger.info("=" * 60)

    return articles


# ============================================================
# CLI 入口
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    publish = "--no-publish" not in sys.argv
    asyncio.run(run_pipeline(publish=publish))
