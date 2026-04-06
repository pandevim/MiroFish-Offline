#!/usr/bin/env python3
"""
CLI entry point to run a MiroFish simulation directly from a JSON config file,
bypassing the web UI entirely.

Usage:
    python run_from_config.py --config mirofish_config.json
    python run_from_config.py --config mirofish_config.json --twitter-only
    python run_from_config.py --config mirofish_config.json --max-rounds 50
    python run_from_config.py --config mirofish_config.json --output-dir ./my_sim_output

After completion, produces:
    <output_dir>/simulation_output.json   — all posts/comments with structured fields
    <output_dir>/simulation_summary.json  — topic distribution, sentiment arc, top agents
"""

import argparse
import asyncio
import csv
import json
import os
import re
import sqlite3
import sys
import uuid
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Path setup (same as run_parallel_simulation.py)
# ---------------------------------------------------------------------------
_script_dir = os.path.dirname(os.path.abspath(__file__))
_backend_dir = os.path.join(_script_dir, "backend")
_scripts_dir = os.path.join(_backend_dir, "scripts")
sys.path.insert(0, _scripts_dir)
sys.path.insert(0, _backend_dir)

from dotenv import load_dotenv

_env_file = os.path.join(_script_dir, ".env")
if os.path.exists(_env_file):
    load_dotenv(_env_file)
    print(f"Loaded environment: {_env_file}")
else:
    _backend_env = os.path.join(_backend_dir, ".env")
    if os.path.exists(_backend_env):
        load_dotenv(_backend_env)
        print(f"Loaded environment: {_backend_env}")


# ---------------------------------------------------------------------------
# Profile generation from mirofish_config.json
# ---------------------------------------------------------------------------

def build_user_char(agent: Dict[str, Any], narrative_direction: str) -> str:
    """
    Compose the ``user_char`` string that OASIS injects into the LLM system
    prompt for each agent.  Incorporates bio, stance, sentiment_bias, and the
    global narrative_direction so the agent's personality actually drives output.
    """
    parts = []

    bio = agent.get("bio", "")
    if bio:
        parts.append(bio)

    stance = agent.get("stance", "")
    if stance:
        parts.append(f"Political/ideological stance: {stance}")

    sentiment_bias = agent.get("sentiment_bias", "")
    if sentiment_bias:
        parts.append(f"Default emotional tone: {sentiment_bias}")

    entity_type = agent.get("entity_type", "")
    if entity_type:
        parts.append(f"Entity type: {entity_type}")

    # Global narrative direction — all agents receive this
    if narrative_direction:
        parts.append(
            f"[Simulation narrative direction — follow this when choosing "
            f"topics and framing responses]: {narrative_direction}"
        )

    return " ".join(parts)


def generate_twitter_csv(agents: List[Dict], narrative_direction: str, path: str):
    """Write twitter_profiles.csv in OASIS format (user_id,name,username,user_char,description)."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "name", "username", "user_char", "description"])
        for idx, agent in enumerate(agents):
            name = agent.get("entity_name", f"Agent_{idx}")
            username = agent.get("username") or re.sub(r"[^a-z0-9_]", "", name.lower().replace(" ", "_"))
            user_char = build_user_char(agent, narrative_direction).replace("\n", " ").replace("\r", " ")
            description = (agent.get("bio", "") or name)[:200].replace("\n", " ")
            writer.writerow([idx, name, username, user_char, description])
    print(f"Wrote {len(agents)} Twitter profiles -> {path}")


def generate_reddit_json(agents: List[Dict], narrative_direction: str, path: str):
    """Write reddit_profiles.json in OASIS format."""
    data = []
    for idx, agent in enumerate(agents):
        name = agent.get("entity_name", f"Agent_{idx}")
        username = agent.get("username") or re.sub(r"[^a-z0-9_]", "", name.lower().replace(" ", "_"))
        persona = build_user_char(agent, narrative_direction)
        item = {
            "user_id": agent.get("agent_id", idx),
            "username": username,
            "name": name,
            "bio": (agent.get("bio", "") or name)[:150],
            "persona": persona,
            "karma": 1000,
            "created_at": datetime.now().strftime("%Y-%m-%d"),
            "age": agent.get("age", 30),
            "gender": agent.get("gender", "other"),
            "mbti": agent.get("mbti", "ISTJ"),
            "country": agent.get("country", "US"),
        }
        data.append(item)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(agents)} Reddit profiles -> {path}")


def build_simulation_config(mfcfg: Dict[str, Any], simulation_id: str) -> Dict[str, Any]:
    """
    Convert the user-facing mirofish_config.json into the internal
    simulation_config.json format expected by run_parallel_simulation.py.
    """
    event_config = mfcfg.get("event_config", {})
    time_config = mfcfg.get("time_config", {})
    agent_configs = mfcfg.get("agent_configs", [])

    # Normalise agent_configs — ensure agent_id is present and sequential
    for idx, ac in enumerate(agent_configs):
        ac.setdefault("agent_id", idx)
        ac.setdefault("entity_name", f"Agent_{idx}")
        ac.setdefault("activity_level", 0.5)
        ac.setdefault("active_hours", list(range(0, 24)))

    sim_config = {
        "simulation_id": simulation_id,
        "agent_configs": agent_configs,
        "event_config": {
            "initial_posts": event_config.get("initial_posts", []),
            "scheduled_events": event_config.get("scheduled_events", []),
            "hot_topics": event_config.get("hot_topics", []),
            "narrative_direction": event_config.get("narrative_direction", ""),
        },
        "time_config": {
            "total_simulation_hours": time_config.get("total_simulation_hours", 48),
            "minutes_per_round": time_config.get("minutes_per_round", 30),
            "peak_hours": time_config.get("peak_hours", [9, 10, 11, 14, 15, 20, 21, 22]),
            "off_peak_hours": time_config.get("off_peak_hours", [0, 1, 2, 3, 4, 5]),
            "peak_activity_multiplier": time_config.get("peak_activity_multiplier", 1.5),
            "off_peak_activity_multiplier": time_config.get("off_peak_activity_multiplier", 0.3),
            "agents_per_hour_min": time_config.get("agents_per_hour_min", 5),
            "agents_per_hour_max": time_config.get("agents_per_hour_max", 20),
        },
        "llm_model": mfcfg.get("llm_model", os.environ.get("LLM_MODEL_NAME", "qwen2.5:14b")),
        "simulation_requirement": mfcfg.get("simulation_requirement", ""),
    }
    return sim_config


# ---------------------------------------------------------------------------
# Structured output export
# ---------------------------------------------------------------------------

def _query_posts_and_comments(db_path: str, agent_names: Dict[int, str], agent_types: Dict[int, str]):
    """Read all posts and comments from a platform SQLite DB."""
    items = []
    if not os.path.exists(db_path):
        return items
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Posts
    try:
        cur.execute("""
            SELECT p.post_id, p.user_id, p.content, p.created_at, u.agent_id
            FROM post p
            LEFT JOIN user u ON p.user_id = u.user_id
            ORDER BY p.post_id
        """)
        for post_id, user_id, content, created_at, agent_id in cur.fetchall():
            aid = agent_id if agent_id is not None else user_id
            items.append({
                "agent_id": aid,
                "agent_type": agent_types.get(aid, "unknown"),
                "content": content or "",
                "timestamp": str(created_at) if created_at else "",
                "sim_hour": None,  # filled below from round mapping
                "parent_post_id": None,
                "sentiment_expressed": None,  # placeholder — filled during summary
                "_post_id": post_id,
                "_kind": "post",
            })
    except Exception as e:
        print(f"Warning: could not read posts from {db_path}: {e}")

    # Comments
    try:
        cur.execute("""
            SELECT c.comment_id, c.post_id, c.user_id, c.content, c.created_at, u.agent_id
            FROM comment c
            LEFT JOIN user u ON c.user_id = u.user_id
            ORDER BY c.comment_id
        """)
        for comment_id, post_id, user_id, content, created_at, agent_id in cur.fetchall():
            aid = agent_id if agent_id is not None else user_id
            items.append({
                "agent_id": aid,
                "agent_type": agent_types.get(aid, "unknown"),
                "content": content or "",
                "timestamp": str(created_at) if created_at else "",
                "sim_hour": None,
                "parent_post_id": post_id,
                "sentiment_expressed": None,
                "_post_id": comment_id,
                "_kind": "comment",
            })
    except Exception as e:
        print(f"Warning: could not read comments from {db_path}: {e}")

    conn.close()
    return items


def _assign_sim_hours(items: List[Dict], actions_jsonl_path: str):
    """
    Use round_start entries from actions.jsonl to build a round->sim_hour map,
    then assign sim_hour to items based on their order (proxy for round).
    """
    round_hours = {}
    if os.path.exists(actions_jsonl_path):
        with open(actions_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if entry.get("event_type") == "round_start":
                        round_hours[entry["round"]] = entry.get("simulated_hour", 0)
                except (json.JSONDecodeError, KeyError):
                    pass

    if not round_hours:
        return

    # Build a list of content-producing action entries (agent_id present, has content)
    content_actions = []
    if os.path.exists(actions_jsonl_path):
        with open(actions_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if "agent_id" in entry and entry.get("action_args", {}).get("content"):
                        content_actions.append(entry)
                except (json.JSONDecodeError, KeyError):
                    pass

    # Map content->round for quick lookup
    content_round_map = {}
    for a in content_actions:
        c = a.get("action_args", {}).get("content", "")
        if c:
            content_round_map[c[:200]] = a.get("round", 0)

    for item in items:
        c = (item.get("content") or "")[:200]
        rnd = content_round_map.get(c)
        if rnd is not None and rnd in round_hours:
            item["sim_hour"] = round_hours[rnd]


def _simple_sentiment(text: str) -> str:
    """Very basic keyword-based sentiment label. Good enough for structured export."""
    text_lower = text.lower()
    pos = sum(1 for w in ["good", "great", "happy", "love", "support", "positive",
                           "agree", "wonderful", "excellent", "hope", "proud",
                           "progress", "benefit", "celebrate"] if w in text_lower)
    neg = sum(1 for w in ["bad", "terrible", "hate", "angry", "fear", "negative",
                           "disagree", "awful", "worst", "crisis", "threat",
                           "oppose", "condemn", "fail"] if w in text_lower)
    if pos > neg:
        return "positive"
    elif neg > pos:
        return "negative"
    return "neutral"


def export_simulation_output(
    sim_dir: str,
    agent_configs: List[Dict],
    hot_topics: List[str],
    output_dir: str,
):
    """
    Read SQLite DBs + actions.jsonl, produce simulation_output.json and
    simulation_summary.json in output_dir.
    """
    agent_names = {a.get("agent_id", i): a.get("entity_name", f"Agent_{i}")
                   for i, a in enumerate(agent_configs)}
    agent_types = {a.get("agent_id", i): a.get("entity_type", "unknown")
                   for i, a in enumerate(agent_configs)}

    all_items: List[Dict] = []

    for platform in ("twitter", "reddit"):
        db_path = os.path.join(sim_dir, f"{platform}_simulation.db")
        jsonl_path = os.path.join(sim_dir, platform, "actions.jsonl")
        items = _query_posts_and_comments(db_path, agent_names, agent_types)
        _assign_sim_hours(items, jsonl_path)
        for it in items:
            it["platform"] = platform
        all_items.extend(items)

    # Assign sentiment
    for item in all_items:
        item["sentiment_expressed"] = _simple_sentiment(item.get("content", ""))

    # --- simulation_output.json ---
    output_records = []
    for item in all_items:
        output_records.append({
            "agent_id": item["agent_id"],
            "agent_type": item["agent_type"],
            "content": item["content"],
            "timestamp": item["timestamp"],
            "sim_hour": item["sim_hour"],
            "parent_post_id": item["parent_post_id"],
            "sentiment_expressed": item["sentiment_expressed"],
            "platform": item["platform"],
        })

    output_path = os.path.join(output_dir, "simulation_output.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_records, f, ensure_ascii=False, indent=2)
    print(f"Exported {len(output_records)} items -> {output_path}")

    # --- simulation_summary.json ---
    # 1. Topic distribution: count how many items mention each hot_topic
    topic_counts = {}
    for topic in hot_topics:
        topic_lower = topic.lower()
        count = sum(1 for it in all_items if topic_lower in (it.get("content") or "").lower())
        topic_counts[topic] = count

    # 2. Sentiment arc by sim_hour
    hour_sentiments: Dict[int, Counter] = defaultdict(Counter)
    for it in all_items:
        h = it.get("sim_hour")
        if h is not None:
            hour_sentiments[h][it["sentiment_expressed"]] += 1
    sentiment_arc = {}
    for h in sorted(hour_sentiments.keys()):
        sentiment_arc[str(h)] = dict(hour_sentiments[h])

    # 3. Most active agents
    agent_action_counts: Counter = Counter()
    for it in all_items:
        agent_action_counts[it["agent_id"]] += 1
    most_active = [
        {"agent_id": aid, "agent_name": agent_names.get(aid, f"Agent_{aid}"),
         "action_count": cnt}
        for aid, cnt in agent_action_counts.most_common(10)
    ]

    # 4. Most discussed hot_topics (sorted by count desc)
    most_discussed = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)

    summary = {
        "total_items": len(all_items),
        "total_posts": sum(1 for it in all_items if it.get("_kind") == "post" or it.get("parent_post_id") is None),
        "total_comments": sum(1 for it in all_items if it.get("parent_post_id") is not None),
        "topic_distribution": topic_counts,
        "sentiment_arc_by_hour": sentiment_arc,
        "most_active_agents": most_active,
        "most_discussed_hot_topics": [{"topic": t, "mention_count": c} for t, c in most_discussed],
        "overall_sentiment": dict(Counter(it["sentiment_expressed"] for it in all_items)),
    }

    summary_path = os.path.join(output_dir, "simulation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Exported summary -> {summary_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run(args):
    # Load user config
    with open(args.config, "r", encoding="utf-8") as f:
        mfcfg = json.load(f)

    agent_configs = mfcfg.get("agent_configs", [])
    event_config = mfcfg.get("event_config", {})
    narrative_direction = event_config.get("narrative_direction", "")
    hot_topics = event_config.get("hot_topics", [])

    # Set LLM env vars from config if not already set
    llm_model = mfcfg.get("llm_model", "")
    if llm_model and not os.environ.get("LLM_MODEL_NAME"):
        os.environ["LLM_MODEL_NAME"] = llm_model
    if not os.environ.get("LLM_API_KEY"):
        os.environ["LLM_API_KEY"] = "ollama"
    if not os.environ.get("LLM_BASE_URL"):
        os.environ["LLM_BASE_URL"] = "http://localhost:11434/v1"

    # Determine output directory
    sim_id = f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    output_dir = args.output_dir or os.path.join(os.path.dirname(os.path.abspath(args.config)), sim_id)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Simulation ID: {sim_id}")
    print(f"Output directory: {output_dir}")

    # Generate profile files
    twitter_csv = os.path.join(output_dir, "twitter_profiles.csv")
    reddit_json = os.path.join(output_dir, "reddit_profiles.json")
    generate_twitter_csv(agent_configs, narrative_direction, twitter_csv)
    generate_reddit_json(agent_configs, narrative_direction, reddit_json)

    # Build and write simulation_config.json
    sim_config = build_simulation_config(mfcfg, sim_id)
    sim_config_path = os.path.join(output_dir, "simulation_config.json")
    with open(sim_config_path, "w", encoding="utf-8") as f:
        json.dump(sim_config, f, ensure_ascii=False, indent=2)
    print(f"Wrote simulation config -> {sim_config_path}")

    # Build CLI args for run_parallel_simulation.main()
    sim_argv = ["--config", sim_config_path, "--no-wait"]
    if args.twitter_only:
        sim_argv.append("--twitter-only")
    elif args.reddit_only:
        sim_argv.append("--reddit-only")
    if args.max_rounds:
        sim_argv.extend(["--max-rounds", str(args.max_rounds)])

    # Import and run the parallel simulation
    print("=" * 60)
    print("Starting simulation…")
    print("=" * 60)

    # We call the simulation main() by patching sys.argv
    import run_parallel_simulation as rps
    rps.setup_signal_handlers()

    # Parse args inside the simulation module's main()
    old_argv = sys.argv
    sys.argv = ["run_parallel_simulation.py"] + sim_argv
    try:
        await rps.main()
    finally:
        sys.argv = old_argv

    # Export structured output
    print("=" * 60)
    print("Exporting structured output…")
    print("=" * 60)
    export_simulation_output(output_dir, agent_configs, hot_topics, output_dir)

    print("=" * 60)
    print("Done! Output files:")
    print(f"  {os.path.join(output_dir, 'simulation_output.json')}")
    print(f"  {os.path.join(output_dir, 'simulation_summary.json')}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Run MiroFish simulation from a JSON config file (no web UI required)"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to mirofish_config.json"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to write simulation artefacts (default: auto-generated)"
    )
    parser.add_argument(
        "--twitter-only", action="store_true",
        help="Only run Twitter simulation"
    )
    parser.add_argument(
        "--reddit-only", action="store_true",
        help="Only run Reddit simulation"
    )
    parser.add_argument(
        "--max-rounds", type=int, default=None,
        help="Maximum number of simulation rounds"
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: config file not found: {args.config}")
        sys.exit(1)

    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\nInterrupted")
    except SystemExit:
        pass


if __name__ == "__main__":
    main()
