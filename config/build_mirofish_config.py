"""
build_mirofish_config.py

Reads historical_ai_headlines_filtered.csv (calibration partition only: pre-Dec 2020),
selects ~200 high-signal seed headlines, creates diverse agent personas,
and outputs a complete MiroFish simulation config JSON.

Usage:
    python build_mirofish_config.py \
        --input historical_ai_headlines_filtered.csv \
        --output mirofish_config.json \
        --num-seeds 200 \
        --split-date 2020-12-01
"""

import argparse
import csv
import json
import random
from datetime import datetime, timedelta
from collections import defaultdict

# ─── Agent persona definitions ───────────────────────────────────────────────
# Diverse perspectives to simulate real public discourse about AI
AGENT_PERSONAS = [
    # Tech optimists
    {"entity_type": "Tech Entrepreneur", "stance": "strongly_positive", "sentiment_bias": 0.4,
     "bio": "Silicon Valley founder who believes AI will solve humanity's biggest problems. Evangelical about automation and efficiency gains."},
    {"entity_type": "AI Researcher", "stance": "positive", "sentiment_bias": 0.2,
     "bio": "Academic ML researcher excited about breakthroughs but aware of limitations. Focuses on capability advances and benchmarks."},
    {"entity_type": "Tech Journalist", "stance": "neutral", "sentiment_bias": 0.05,
     "bio": "Covers AI industry for a major tech publication. Tries to be balanced but drawn to dramatic narratives and hype cycles."},

    # Workers & economic perspective
    {"entity_type": "Factory Worker", "stance": "negative", "sentiment_bias": -0.3,
     "bio": "Blue-collar worker worried about automation replacing jobs. Skeptical of promises that AI will create more jobs than it destroys."},
    {"entity_type": "HR Manager", "stance": "cautious", "sentiment_bias": -0.1,
     "bio": "Corporate HR professional navigating AI hiring tools, workforce reskilling, and employee anxiety about automation."},
    {"entity_type": "Economist", "stance": "neutral", "sentiment_bias": 0.0,
     "bio": "Labor economist studying AI's impact on wages, productivity, and inequality. Data-driven, sees both displacement and augmentation."},

    # Ethics & policy
    {"entity_type": "Ethics Advocate", "stance": "critical", "sentiment_bias": -0.2,
     "bio": "AI ethics researcher focused on bias, fairness, and accountability. Pushes for regulation and transparency in AI systems."},
    {"entity_type": "Policy Maker", "stance": "cautious", "sentiment_bias": -0.05,
     "bio": "Government technology policy advisor trying to balance innovation with public safety. Interested in regulation frameworks."},
    {"entity_type": "Privacy Activist", "stance": "negative", "sentiment_bias": -0.3,
     "bio": "Digital rights advocate alarmed by facial recognition, surveillance AI, and data harvesting. Calls for bans and strict regulation."},

    # Healthcare
    {"entity_type": "Doctor", "stance": "positive", "sentiment_bias": 0.15,
     "bio": "Physician who uses AI diagnostic tools daily. Sees potential in medical AI but worries about over-reliance and liability."},
    {"entity_type": "Patient Advocate", "stance": "cautious", "sentiment_bias": -0.1,
     "bio": "Health advocacy group leader concerned about AI in healthcare decisions, data privacy, and equitable access to AI-powered care."},

    # Education
    {"entity_type": "Teacher", "stance": "cautious", "sentiment_bias": 0.0,
     "bio": "High school educator exploring AI tutoring tools while worried about cheating, critical thinking erosion, and digital divide."},

    # Creative / cultural
    {"entity_type": "Artist", "stance": "negative", "sentiment_bias": -0.25,
     "bio": "Digital artist threatened by generative AI. Vocal about copyright, creative authenticity, and the devaluation of human artistry."},
    {"entity_type": "Content Creator", "stance": "positive", "sentiment_bias": 0.2,
     "bio": "YouTuber and social media creator who embraces AI tools for editing, writing, and content generation. Sees AI as empowering."},

    # General public
    {"entity_type": "Parent", "stance": "cautious", "sentiment_bias": -0.15,
     "bio": "Parent of teenagers worried about deepfakes, AI-generated misinformation, children's screen time, and social media manipulation."},
    {"entity_type": "College Student", "stance": "positive", "sentiment_bias": 0.1,
     "bio": "Computer science student excited about AI career opportunities but uncertain about the job market and ethical implications."},
    {"entity_type": "Retiree", "stance": "skeptical", "sentiment_bias": -0.1,
     "bio": "Retired professional suspicious of tech hype. Relies on traditional media, worried about scams and losing human connection."},

    # Security / military
    {"entity_type": "Cybersecurity Analyst", "stance": "critical", "sentiment_bias": -0.15,
     "bio": "Security professional focused on AI-powered threats: deepfakes, automated hacking, social engineering at scale."},

    # Business
    {"entity_type": "CEO", "stance": "strongly_positive", "sentiment_bias": 0.35,
     "bio": "Fortune 500 executive pushing AI adoption for competitive advantage. Focused on ROI, productivity, and market disruption."},
    {"entity_type": "Small Business Owner", "stance": "neutral", "sentiment_bias": 0.05,
     "bio": "Local business owner curious about AI tools but overwhelmed by options and skeptical of overpromises. Budget-conscious."},
]


# ─── Topic keywords for hot_topics ────────────────────────────────────────────
# Derived from BERTopic results in the analysis
HOT_TOPICS = [
    "deepfakes and AI-generated misinformation",
    "robotics and physical automation",
    "AI in healthcare and medical diagnosis",
    "facial recognition and biometric surveillance",
    "autonomous vehicles and self-driving cars",
    "voice assistants and conversational AI",
    "AI job displacement and workforce automation",
    "AI ethics, bias, and regulation",
    "AI in education and learning",
    "neural interfaces and brain-computer technology",
    "AI and pandemic response",
    "creative AI and generative content",
    "smart cities and IoT",
    "AI and national security (China, military)",
]

# ─── Narrative direction grounded in theoretical frameworks ───────────────────
NARRATIVE_DIRECTION = """This simulation models how public discourse about AI evolves and shapes human behavior.
Agents should engage with headlines and each other following these empirically-grounded patterns:

1. AMARA'S LAW: Early discussion overestimates short-term AI impact (hype), followed by
   disillusionment when reality falls short, then gradual recognition of long-term transformation.
   Agents should exhibit this cycle in their sentiment and expectations.

2. TECHNOLOGY S-CURVE: Different AI themes (deepfakes, healthcare, jobs) emerge at different
   times and follow adoption curves — slow start, rapid growth, plateau. Agents should reflect
   growing awareness of emerging topics.

3. JEVONS PARADOX: As AI makes tasks easier, people do MORE of them rather than resting.
   Productivity-focused agents should gradually express both efficiency gains AND burnout/overwork.

4. SAPIR-WHORF HYPOTHESIS: The language used to describe AI (threat vs tool vs partner)
   shapes how agents think about it. Framing should evolve over time.

5. PYGMALION EFFECT: Expectations about AI become self-fulfilling. Optimistic agents push
   adoption; fearful agents push regulation. Both responses shape actual outcomes.

6. POWER LAW: A few dominant narratives (jobs, deepfakes) attract most attention while
   important niche topics (healthcare, education) get less coverage despite high impact."""

# ─── Simulation requirement ───────────────────────────────────────────────────
SIMULATION_REQUIREMENT = """Simulate how AI-related public discourse evolves over time and
how these narratives shape human behavior. Agents represent diverse societal perspectives
(workers, researchers, policymakers, parents, artists, etc.) and react to real AI news
headlines from 2018-2020. Track how sentiment, dominant topics, and behavioral attitudes
shift through agent interactions. The goal is to produce emergent predictions about future
AI discourse trajectories and their behavioral consequences."""


def load_calibration_headlines(csv_path, split_date):
    """Load headlines from calibration period (before split_date)."""
    headlines = []
    skipped = 0
    total_rows = 0
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            try:
                date = datetime.strptime(row["date"].strip(), "%Y-%m-%d")
            except (ValueError, KeyError):
                # Try alternative date formats
                try:
                    date = datetime.strptime(row["date"].strip()[:10], "%Y-%m-%d")
                except Exception:
                    skipped += 1
                    continue

            if date < split_date:
                headlines.append({
                    "title": row["title"].strip(),
                    "date": date,
                    "query_overlap": int(row.get("query_overlap_count", 1)),
                    "relevance": float(row.get("ai_relevance_score", 0.5)),
                    "source": row.get("source", ""),
                    "matched_queries": row.get("matched_queries", ""),
                })

            if total_rows % 500 == 0:
                print(f"  ... read {total_rows} rows, {len(headlines)} kept so far", flush=True)

    print(f"  Read {total_rows} total rows, kept {len(headlines)} before {split_date.strftime('%Y-%m-%d')}, skipped {skipped} unparseable", flush=True)
    return headlines


def select_seed_headlines(headlines, num_seeds=200):
    """
    Select high-signal headlines with diversity across time and topics.

    Strategy:
    - Score each headline by: relevance * (1 + log(query_overlap))
    - Bin by quarter to ensure temporal spread
    - Take top headlines from each quarter proportionally
    """
    import math

    print(f"  Scoring {len(headlines)} headlines...", flush=True)
    # Score headlines
    for h in headlines:
        h["score"] = h["relevance"] * (1 + math.log(max(h["query_overlap"], 1)))

    # Bin by quarter
    quarters = defaultdict(list)
    for h in headlines:
        q = f"{h['date'].year}Q{(h['date'].month - 1) // 3 + 1}"
        quarters[q] = quarters.get(q, [])
        quarters[q].append(h)

    print(f"  Quarters found: {sorted(quarters.keys())}", flush=True)
    for q in sorted(quarters.keys()):
        print(f"    {q}: {len(quarters[q])} headlines", flush=True)

    # Sort each quarter by score descending
    for q in quarters:
        quarters[q].sort(key=lambda x: x["score"], reverse=True)

    # Allocate seeds proportionally across quarters, minimum 5 per quarter
    total = sum(len(v) for v in quarters.values())
    selected = []
    sorted_quarters = sorted(quarters.keys())

    for q in sorted_quarters:
        proportion = len(quarters[q]) / total
        n = max(5, int(num_seeds * proportion))
        selected.extend(quarters[q][:n])

    # If we have too many, trim by global score; if too few, add more top-scorers
    if len(selected) > num_seeds:
        print(f"  Trimming {len(selected)} → {num_seeds} by score", flush=True)
        selected.sort(key=lambda x: x["score"], reverse=True)
        selected = selected[:num_seeds]
    elif len(selected) < num_seeds:
        gap = num_seeds - len(selected)
        print(f"  Filling {gap} extra seeds from remaining pool", flush=True)
        used_titles = {s["title"] for s in selected}
        remaining = [h for h in headlines if h["title"] not in used_titles]
        remaining.sort(key=lambda x: x["score"], reverse=True)
        selected.extend(remaining[: num_seeds - len(selected)])

    # Sort chronologically for the simulation
    selected.sort(key=lambda x: x["date"])
    print(f"  Final selection: {len(selected)} seeds, {selected[0]['date'].strftime('%Y-%m-%d')} → {selected[-1]['date'].strftime('%Y-%m-%d')}", flush=True)
    return selected


def build_agent_configs(personas):
    """Convert persona definitions into MiroFish agent_configs."""
    agents = []
    for i, persona in enumerate(personas):
        agents.append({
            "agent_id": i,
            "entity_uuid": f"entity-{i}",
            "entity_name": f"{persona['entity_type']} #{i}",
            "entity_type": persona["entity_type"],
            "bio": persona["bio"],
            "activity_level": round(random.uniform(0.3, 0.8), 2),
            "posts_per_hour": round(random.uniform(0.5, 2.0), 1),
            "comments_per_hour": round(random.uniform(1.0, 4.0), 1),
            "active_hours": list(range(7, 24)),
            "response_delay_min": 5,
            "response_delay_max": 120,
            "sentiment_bias": persona["sentiment_bias"],
            "stance": persona["stance"],
            "influence_weight": round(random.uniform(0.5, 1.5), 2),
        })
    return agents


def build_initial_posts(selected_headlines, num_agents):
    """
    Convert first batch of headlines into initial_posts.
    Use ~30% of seeds as initial posts, rest as scheduled events.
    """
    n_initial = max(10, len(selected_headlines) // 3)
    initial = selected_headlines[:n_initial]

    posts = []
    for i, h in enumerate(initial):
        # Assign to a relevant agent based on headline content (simple keyword matching)
        agent_id = assign_agent(h, num_agents)
        posts.append({
            "content": f"[{h['date'].strftime('%Y-%m-%d')}] {h['title']}",
            "poster_agent_id": agent_id,
            "poster_type": AGENT_PERSONAS[agent_id]["entity_type"],
        })
    return posts, n_initial


def build_scheduled_events(selected_headlines, n_initial, num_agents, total_hours):
    """
    Remaining headlines become scheduled_events injected over simulation time.
    Maps headline chronological order to simulation hours.
    """
    remaining = selected_headlines[n_initial:]
    if not remaining:
        return []

    events = []
    for i, h in enumerate(remaining):
        # Spread across simulation hours
        sim_hour = int((i / len(remaining)) * total_hours)
        agent_id = assign_agent(h, num_agents)
        events.append({
            "trigger_hour": sim_hour,
            "event_type": "inject_post",
            "content": f"[{h['date'].strftime('%Y-%m-%d')}] {h['title']}",
            "poster_agent_id": agent_id,
        })
    return events


def assign_agent(headline, num_agents):
    """
    Simple keyword-based agent assignment.
    Maps headline topics to relevant agent personas.
    """
    title_lower = headline["title"].lower()

    # Keyword → agent_id mapping (index into AGENT_PERSONAS)
    keyword_map = {
        0: ["startup", "innovation", "billion", "funding", "launch"],          # Tech Entrepreneur
        1: ["research", "model", "neural", "algorithm", "benchmark"],          # AI Researcher
        2: ["report", "study", "survey", "according", "analysis"],             # Tech Journalist
        3: ["factory", "manufacturing", "automat", "worker", "labor"],         # Factory Worker
        4: ["hiring", "recruit", "workforce", "talent", "employee"],           # HR Manager
        5: ["economy", "wage", "gdp", "productiv", "inequality"],             # Economist
        6: ["bias", "ethic", "fairness", "accountab", "discriminat"],          # Ethics Advocate
        7: ["regulat", "policy", "government", "law", "legislat", "ban"],      # Policy Maker
        8: ["surveillance", "privacy", "facial recognition", "track", "spy"],  # Privacy Activist
        9: ["health", "medical", "diagnos", "patient", "cancer", "drug"],      # Doctor
        10: ["patient rights", "health equity", "access"],                      # Patient Advocate
        11: ["education", "school", "student", "teach", "learn", "tutor"],     # Teacher
        12: ["art", "creative", "music", "paint", "copyright", "artist"],      # Artist
        13: ["content", "youtube", "social media", "influencer", "creator"],   # Content Creator
        14: ["child", "parent", "kid", "teen", "family", "deepfake"],          # Parent
        15: ["college", "university", "campus", "graduate"],                   # College Student
        16: ["scam", "elder", "retir"],                                         # Retiree
        17: ["cyber", "hack", "security", "malware", "phishing"],             # Cybersecurity Analyst
        18: ["ceo", "enterprise", "corporate", "profit", "revenue"],           # CEO
        19: ["small business", "local", "shop"],                               # Small Business Owner
    }

    for agent_id, keywords in keyword_map.items():
        if agent_id < num_agents and any(kw in title_lower for kw in keywords):
            return agent_id

    # Default: distribute among journalist, researcher, and general agents
    return random.choice([2, 5, 7, 14])


def build_config(selected_headlines, total_sim_hours=48):
    """Build the complete MiroFish simulation config."""
    agents = build_agent_configs(AGENT_PERSONAS)
    num_agents = len(agents)

    initial_posts, n_initial = build_initial_posts(selected_headlines, num_agents)
    scheduled_events = build_scheduled_events(
        selected_headlines, n_initial, num_agents, total_sim_hours
    )

    config = {
        "simulation_id": "raise26_ai_behavior_sim",
        "project_id": "raise26",
        "graph_id": "raise26_graph",
        "simulation_requirement": SIMULATION_REQUIREMENT.strip(),
        "time_config": {
            "total_simulation_hours": total_sim_hours,
            "minutes_per_round": 60,
            "agents_per_hour_min": 3,
            "agents_per_hour_max": 8,
            "peak_hours": [18, 19, 20, 21, 22],
            "peak_activity_multiplier": 1.8,
            "off_peak_hours": [0, 1, 2, 3, 4, 5],
            "off_peak_activity_multiplier": 0.05,
            "morning_hours": [6, 7, 8],
            "morning_activity_multiplier": 0.4,
            "work_hours": [9, 10, 11, 12, 13, 14, 15, 16, 17],
            "work_activity_multiplier": 0.7,
        },
        "agent_configs": agents,
        "event_config": {
            "initial_posts": initial_posts,
            "scheduled_events": scheduled_events,
            "hot_topics": HOT_TOPICS,
            "narrative_direction": NARRATIVE_DIRECTION.strip(),
        },
        "twitter_config": {
            "platform": "twitter",
            "recency_weight": 0.4,
            "popularity_weight": 0.3,
            "relevance_weight": 0.3,
            "viral_threshold": 10,
            "echo_chamber_strength": 0.5,
        },
        "reddit_config": {
            "platform": "reddit",
            "recency_weight": 0.4,
            "popularity_weight": 0.3,
            "relevance_weight": 0.3,
            "viral_threshold": 10,
            "echo_chamber_strength": 0.5,
        },
        "llm_model": "qwen2.5:14b",
        "llm_base_url": "http://localhost:11434",
        "generated_at": datetime.now().isoformat(),
        "generation_reasoning": (
            "Config auto-generated from RAISE-26 historical AI headlines (2018-2020 calibration set). "
            f"{len(selected_headlines)} seed headlines selected by relevance × query overlap score, "
            f"spread across {len(set(f'{h['date'].year}Q{(h['date'].month-1)//3+1}' for h in selected_headlines))} quarters. "
            f"{num_agents} agent personas representing diverse societal perspectives. "
            "Theoretical frameworks (Amara's Law, S-Curve, Jevons Paradox, Sapir-Whorf, Pygmalion, Power Law) "
            "encoded in narrative_direction."
        ),
    }

    # Summary stats
    print(f"\n{'='*60}")
    print("MIROFISH CONFIG SUMMARY")
    print(f"{'='*60}")
    print(f"  Seed headlines:     {len(selected_headlines)}")
    print(f"  Initial posts:      {len(initial_posts)}")
    print(f"  Scheduled events:   {len(scheduled_events)}")
    print(f"  Agents:             {num_agents}")
    print(f"  Hot topics:         {len(HOT_TOPICS)}")
    print(f"  Simulation hours:   {total_sim_hours}")
    print(f"  Date range:         {selected_headlines[0]['date'].strftime('%Y-%m-%d')} → {selected_headlines[-1]['date'].strftime('%Y-%m-%d')}")
    print(f"{'='*60}\n")

    return config


def main():
    parser = argparse.ArgumentParser(description="Build MiroFish simulation config from filtered headlines")
    parser.add_argument("--input", default="historical_ai_headlines_filtered.csv", help="Path to filtered headlines CSV")
    parser.add_argument("--output", default="mirofish_config.json", help="Output JSON config path")
    parser.add_argument("--num-seeds", type=int, default=200, help="Number of seed headlines to select")
    parser.add_argument("--split-date", default="2020-12-01", help="Calibration/validation split date (YYYY-MM-DD)")
    parser.add_argument("--sim-hours", type=int, default=48, help="Total simulation hours")
    args = parser.parse_args()

    split_date = datetime.strptime(args.split_date, "%Y-%m-%d")

    import os

    print(f"\n[1/4] Loading headlines from {args.input} (before {args.split_date})...", flush=True)
    if not os.path.exists(args.input):
        print(f"  ERROR: Input file not found: {os.path.abspath(args.input)}")
        print(f"  CWD: {os.getcwd()}")
        return
    print(f"  File found: {os.path.abspath(args.input)} ({os.path.getsize(args.input):,} bytes)", flush=True)
    headlines = load_calibration_headlines(args.input, split_date)
    if not headlines:
        print("  ERROR: No headlines loaded! Check CSV format (needs 'date' and 'title' columns).")
        return
    print(f"  Loaded {len(headlines)} calibration headlines", flush=True)

    print(f"\n[2/4] Selecting {args.num_seeds} high-signal seeds...", flush=True)
    selected = select_seed_headlines(headlines, args.num_seeds)
    print(f"  Selected {len(selected)} seeds", flush=True)

    print(f"\n[3/4] Building simulation config...", flush=True)
    config = build_config(selected, total_sim_hours=args.sim_hours)

    print(f"\n[4/4] Writing config to {args.output}...", flush=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, default=str)

    output_size = os.path.getsize(args.output)
    print(f"  Config saved to {os.path.abspath(args.output)} ({output_size:,} bytes)", flush=True)
    print("  Done!", flush=True)


if __name__ == "__main__":
    main()