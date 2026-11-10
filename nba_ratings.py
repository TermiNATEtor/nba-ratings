"""
NBA player rating app.

Fetches per-game stats from NBA.com (via the nba_api package), rates every
player 1-10 in 10 categories using dense ranking, and writes index.html: a
single self-contained, phone-friendly, sortable table.

Setup (once):   pip install nba_api pandas
Run:            python nba_ratings.py
Then open:      index.html
"""
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- settings
SEASON = "2026-27"

# Where the stats come from: "espn" (works from GitHub's servers) or
# "nba" (NBA.com, which blocks cloud servers but works from your own computer).
DATA_SOURCE = "espn"
ESPN_SEASON = 2027   # ESPN names a season by the year it ENDS (2026-27 -> 2027)

# "Qualified players only" view (the on/off switch on the page).
# A player must meet BOTH of these to appear in that view:
MIN_GP_PCT = 0.50   # at least 50% of their team's games played
MIN_MPG = 15.0      # at least 15 minutes per game
# In that view, a shooting % is only rated if the player takes at least this
# many attempts per game; otherwise that category gets a 1.
MIN_ATTEMPTS_PER_GAME = {"FG_PCT": 3.0, "FT_PCT": 1.0, "FG3_PCT": 1.0}

# (key, label, source column, lower_is_better, display format)
CATEGORIES = [
    ("ppg", "PPG", "PTS", False, "1d"),
    ("rpg", "RPG", "REB", False, "1d"),
    ("apg", "APG", "AST", False, "1d"),
    ("spg", "SPG", "STL", False, "1d"),
    ("bpg", "BPG", "BLK", False, "1d"),
    ("fgp", "FG%", "FG_PCT", False, "pct"),
    ("ftp", "FT%", "FT_PCT", False, "pct"),
    ("tpp", "3P%", "FG3_PCT", False, "pct"),
    ("dur", "GP%", "DUR", False, "pct"),   # games played / team games played
    ("tov", "TO", "TOV", True, "1d"),      # inverted: fewer turnovers = better
]


# Emoji + full title shown on the front page for each category.
CATEGORY_INFO = {
    "ppg": ("🔥", "Points per game"),
    "rpg": ("💪", "Rebounds per game"),
    "apg": ("🤝", "Assists per game"),
    "spg": ("🖐️", "Steals per game"),
    "bpg": ("🛡️", "Blocks per game"),
    "fgp": ("🎯", "Field goal %"),
    "ftp": ("💸", "Free throw %"),
    "tpp": ("👌", "3-point %"),
    "dur": ("🔋", "Durability (games played %)"),
    "tov": ("🔒", "Fewest turnovers per game"),
}


# Team names so you can search "lakers", "los angeles lakers", "LAL", etc.
# (code, full name, extra search words, other codes ESPN / NBA.com may use)
_TEAMS = [
    ("ATL", "Atlanta Hawks", "", ()), ("BOS", "Boston Celtics", "", ()),
    ("BKN", "Brooklyn Nets", "", ("BRK",)), ("CHA", "Charlotte Hornets", "", ("CHO",)),
    ("CHI", "Chicago Bulls", "", ()), ("CLE", "Cleveland Cavaliers", "cavs", ()),
    ("DAL", "Dallas Mavericks", "mavs", ()), ("DEN", "Denver Nuggets", "", ()),
    ("DET", "Detroit Pistons", "", ()), ("GSW", "Golden State Warriors", "", ("GS",)),
    ("HOU", "Houston Rockets", "", ()), ("IND", "Indiana Pacers", "", ()),
    ("LAC", "Los Angeles Clippers", "la clippers", ()),
    ("LAL", "Los Angeles Lakers", "la lakers", ()),
    ("MEM", "Memphis Grizzlies", "", ()), ("MIA", "Miami Heat", "", ()),
    ("MIL", "Milwaukee Bucks", "", ()), ("MIN", "Minnesota Timberwolves", "wolves", ()),
    ("NOP", "New Orleans Pelicans", "pels", ("NO",)), ("NYK", "New York Knicks", "", ("NY",)),
    ("OKC", "Oklahoma City Thunder", "", ()), ("ORL", "Orlando Magic", "", ()),
    ("PHI", "Philadelphia 76ers", "sixers", ()), ("PHX", "Phoenix Suns", "", ("PHO",)),
    ("POR", "Portland Trail Blazers", "blazers", ()), ("SAC", "Sacramento Kings", "", ()),
    ("SAS", "San Antonio Spurs", "", ("SA",)), ("TOR", "Toronto Raptors", "", ()),
    ("UTA", "Utah Jazz", "", ("UTAH",)), ("WAS", "Washington Wizards", "", ("WSH",)),
]
TEAM_LOOKUP = {code: {"n": name, "a": extra}
               for canon, name, extra, others in _TEAMS
               for code in (canon, *others)}


# ---------------------------------------------------------------- positions
def to_groups(pos):
    """'PG' -> 'G', 'PF' -> 'F', 'C' -> 'C', 'G-F' -> 'GF'. Unknown -> ''."""
    if not pos:
        return ""
    t = str(pos).upper()
    found = set()
    if "GUARD" in t:
        found.add("G")
    if "FORWARD" in t:
        found.add("F")
    if "CENTER" in t:
        found.add("C")
    for tok in re.split(r"[^A-Z]+", t):
        if tok in ("PG", "SG", "G"):
            found.add("G")
        elif tok in ("SF", "PF", "F"):
            found.add("F")
        elif tok == "C":
            found.add("C")
        elif tok in ("GF", "FG"):
            found.update("GF")
        elif tok in ("FC", "CF"):
            found.update("FC")
    return "".join(c for c in "GFC" if c in found)


# ---------------------------------------------------------------- scoring
def tier_score(rank):
    """Dense rank -> 1-10 score. Unranked (NaN) -> 1."""
    if pd.isna(rank):
        return 1
    rank = int(rank)
    if rank == 1:
        return 10
    if rank <= 5:
        return 9
    if rank >= 41:
        return 1
    return 8 - (rank - 6) // 5  # 6-10 -> 8, 11-15 -> 7, ... 36-40 -> 2


def rate_column(series, lower_is_better=False):
    """Dense ranking: tied values share a rank, next value is rank + 1."""
    ranks = series.rank(method="dense", ascending=lower_is_better)
    return ranks.apply(tier_score)


def fmt(value, kind):
    if pd.isna(value):
        return "—"
    return f"{value * 100:.1f}%" if kind == "pct" else f"{value:.1f}"


def compute_ratings(players, teams, qualified_only=False):
    """players / teams are the raw NBA.com per-game DataFrames.
    qualified_only=True drops low-games / low-minutes players BEFORE ranking."""
    df = players.copy()
    if "PLAYER_ID" not in df:
        df["PLAYER_ID"] = df.index.astype(str)
    if "POS" not in df:
        df["POS"] = ""
    if "PHOTO" not in df:  # NBA.com headshots (used when DATA_SOURCE = "nba")
        df["PHOTO"] = df["PLAYER_ID"].map(
            lambda i: f"https://cdn.nba.com/headshots/nba/latest/1040x760/{i}.png")

    # Durability = games played / team games played (capped at 100%)
    team_gp = dict(zip(teams["TEAM_ID"], teams["GP"]))
    df["TEAM_GP"] = df["TEAM_ID"].map(team_gp)
    df["DUR"] = (df["GP"] / df["TEAM_GP"]).clip(upper=1.0)

    # Qualified view: remove low-games / low-minutes players before ranking,
    # so everyone left is ranked only against other qualified players.
    if qualified_only:
        df = df[(df["DUR"] >= MIN_GP_PCT) & (df["MIN"] >= MIN_MPG)].copy()

    # NBA.com reports 0.0% for zero attempts; treat that as "no rating".
    for pct, att in (("FG_PCT", "FGA"), ("FT_PCT", "FTA"), ("FG3_PCT", "FG3A")):
        no_data = df[att] <= 0
        if qualified_only:
            no_data = no_data | (df[att] < MIN_ATTEMPTS_PER_GAME[pct])
        df.loc[no_data, pct] = np.nan

    # Round to the precision shown on NBA.com so ties match what you see.
    for _, _, col, _, kind in CATEGORIES:
        df[col] = df[col].round(3 if kind == "pct" else 1)
    df["MIN"] = df["MIN"].round(1)

    # Score each category, then total (equal weights, max 100).
    for key, _, col, lower, _ in CATEGORIES:
        df["s_" + key] = rate_column(df[col], lower)
    df["OVERALL"] = df[["s_" + c[0] for c in CATEGORIES]].sum(axis=1)

    rows = []
    for _, r in df.sort_values("OVERALL", ascending=False).iterrows():
        photo = r.get("PHOTO")
        rows.append({
            "id": str(r["PLAYER_ID"]),
            "pos": r["POS"] if isinstance(r["POS"], str) else "",
            "g": to_groups(r["POS"]) if isinstance(r["POS"], str) else "",
            "photo": photo if isinstance(photo, str) else "",
            "name": r["PLAYER_NAME"],
            "team": r["TEAM_ABBREVIATION"],
            "mpg": f"{r['MIN']:.1f}",
            "gp": int(r["GP"]),
            "overall": int(r["OVERALL"]),
            "s": {c[0]: int(r["s_" + c[0]]) for c in CATEGORIES},
            "v": {c[0]: fmt(r[c[2]], c[4]) for c in CATEGORIES},
            "n": {c[0]: (None if pd.isna(r[c[2]]) else float(r[c[2]])) for c in CATEGORIES},
        })
    return rows


# ---------------------------------------------------------------- fetching
# ESPN names for each stat we need (first match wins; case-insensitive).
ESPN_STATS = {
    "GP": ["gamesPlayed"],
    "MIN": ["avgMinutes", "minutesPerGame"],
    "PTS": ["avgPoints", "pointsPerGame"],
    "REB": ["avgRebounds", "reboundsPerGame"],
    "AST": ["avgAssists", "assistsPerGame"],
    "STL": ["avgSteals", "stealsPerGame"],
    "BLK": ["avgBlocks", "blocksPerGame"],
    "TOV": ["avgTurnovers", "turnoversPerGame"],
    "FGA": ["avgFieldGoalsAttempted"],
    "FG_PCT": ["fieldGoalPct"],
    "FTA": ["avgFreeThrowsAttempted"],
    "FT_PCT": ["freeThrowPct"],
    "FG3A": ["avgThreePointFieldGoalsAttempted"],
    "FG3_PCT": ["threePointFieldGoalPct", "threePointPct"],
}
ESPN_URL = ("https://site.web.api.espn.com/apis/common/v3/sports/"
            "basketball/nba/statistics/byathlete")


def _raw_position(a):
    p = a.get("position")
    if isinstance(p, dict):
        p = p.get("abbreviation") or p.get("displayName") or p.get("name")
    if not p:
        p = a.get("positionAbbreviation")
    return str(p) if p else ""


def fetch_roster_positions():
    """Backup plan: read each team's roster page, which lists every position."""
    import time
    import requests

    base = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; nba-ratings)"}
    teams = requests.get(base + "/teams", headers=headers, timeout=30).json()
    ids = [t["team"]["id"] for t in teams["sports"][0]["leagues"][0]["teams"]]
    found = {}

    def record(a):
        pos = a.get("position")
        if isinstance(pos, dict):
            pos = pos.get("abbreviation") or pos.get("displayName") or pos.get("name")
        if pos and a.get("id"):
            found[str(a["id"])] = str(pos)

    for tid in ids:
        data = requests.get(f"{base}/teams/{tid}/roster", headers=headers, timeout=30).json()
        for a in data.get("athletes", []):
            if isinstance(a, dict) and "items" in a:   # grouped layout
                for b in a["items"]:
                    record(b)
            elif isinstance(a, dict):
                record(a)
        time.sleep(0.3)
    return found


def fill_positions(df):
    """Use positions from the stats feed; if most are missing, use rosters."""
    if (df["POS"].str.len() > 0).mean() >= 0.8:
        return df
    try:
        pos = fetch_roster_positions()
    except Exception as e:
        print("Warning: could not load player positions:", e)
        return df
    df["POS"] = [p if p else pos.get(str(i), "") for p, i in zip(df["POS"], df["PLAYER_ID"])]
    return df


def team_key(code):
    """One name per team, whether ESPN says 'GS' or NBA.com says 'GSW'."""
    code = str(code)
    return TEAM_LOOKUP.get(code.upper(), {}).get("n") or code


def fetch_team_games():
    """Real games played by each team, from ESPN's standings (wins + losses)."""
    import requests

    url = "https://site.api.espn.com/apis/v2/sports/basketball/nba/standings"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; nba-ratings)"}
    data = requests.get(url, params={"season": ESPN_SEASON}, headers=headers,
                        timeout=30).json()
    games = {}

    def walk(node):
        for e in (node.get("standings") or {}).get("entries", []):
            team = e.get("team") or {}
            st = {x.get("name"): x.get("value") for x in e.get("stats", [])}
            gp = st.get("gamesPlayed")
            if gp is None and st.get("wins") is not None and st.get("losses") is not None:
                gp = st["wins"] + st["losses"]
            if gp is None:
                continue
            for name in (team.get("abbreviation"), team.get("displayName")):
                if name:
                    games[team_key(name)] = int(gp)
        for child in node.get("children", []):
            walk(child)

    walk(data)
    return games


def apply_team_games(players, teams):
    """Replace the 'most games any teammate played' guess with the real count."""
    try:
        real = fetch_team_games()
    except Exception as e:
        print("Warning: could not load team games from standings:", e)
        return teams
    guess = dict(zip(teams["TEAM_ID"], teams["GP"]))
    out, hits = [], 0
    for code in teams["TEAM_ID"]:
        gp = real.get(team_key(code))
        hits += gp is not None
        out.append(gp if gp is not None else guess[code])
    print(f"Team games from ESPN standings: {hits} of {len(out)} teams")
    return pd.DataFrame({"TEAM_ID": teams["TEAM_ID"], "GP": out})


def _page_names(page):
    """ESPN lists each stat group's names once at the top of the page."""
    top = {}
    for i, c in enumerate(page.get("categories", [])):
        top[c.get("name")] = c.get("names", [])
        top[i] = c.get("names", [])
    return top


def _cat_names(cat, top, i):
    """Names for one player's stat group (from the player, else from the page top)."""
    return cat.get("names") or top.get(cat.get("name")) or top.get(i) or []


def parse_espn(pages):
    """Turn ESPN's pages of JSON into the same tables the scoring code uses."""
    # Collect every stat name ESPN gave us, so a mismatch is easy to diagnose.
    available = set()
    for page in pages:
        top = _page_names(page)
        for item in page.get("athletes", []):
            for i, cat in enumerate(item.get("categories", [])):
                available.update(n.lower() for n in _cat_names(cat, top, i))

    if not available:  # structure isn't what we expected: show what we got
        first = (pages[0].get("athletes") or [{}])[0]
        cats = first.get("categories") or [{}]
        raise ValueError("Could not find stat names in ESPN's data. "
                         f"Page keys: {sorted(pages[0].keys())}. "
                         f"Player keys: {sorted(first.keys())}. "
                         f"First stat group keys: {sorted(cats[0].keys())}. "
                         f"Page 'categories' sample: {str(pages[0].get('categories'))[:300]}")

    keys = {}
    for col, candidates in ESPN_STATS.items():
        found = next((c.lower() for c in candidates if c.lower() in available), None)
        if found is None:
            raise ValueError(f"ESPN data has no stat for '{col}'. "
                             f"Stat names it does have: {sorted(available)}")
        keys[col] = found

    rows = []
    for page in pages:
        top = _page_names(page)
        for item in page.get("athletes", []):
            a = item.get("athlete", {})
            flat = {}
            for i, cat in enumerate(item.get("categories", [])):
                for name, val in zip(_cat_names(cat, top, i), cat.get("values", [])):
                    flat[name.lower()] = val
            head = a.get("headshot")
            photo = head.get("href") if isinstance(head, dict) else None
            if not photo and a.get("id"):
                photo = ("https://a.espncdn.com/i/headshots/nba/players/full/"
                         f"{a['id']}.png")
            row = {
                "POS": _raw_position(a),
                "PHOTO": photo,
                "PLAYER_ID": a.get("id"),
                "PLAYER_NAME": a.get("displayName"),
                "TEAM_ABBREVIATION": a.get("teamShortName") or a.get("teamName") or "—",
            }
            for col, key in keys.items():
                row[col] = pd.to_numeric(flat.get(key), errors="coerce")
            rows.append(row)

    df = pd.DataFrame(rows).dropna(subset=["PLAYER_NAME", "GP"])
    df = df[df["GP"] > 0]
    df = df.sort_values("GP", ascending=False).drop_duplicates("PLAYER_ID")

    # ESPN gives percentages like 48.7; the scoring code wants 0.487.
    for pct in ("FG_PCT", "FT_PCT", "FG3_PCT"):
        if df[pct].max() > 1.5:
            df[pct] = df[pct] / 100.0

    # Team games played = the most games any player on that team has played.
    df["TEAM_ID"] = df["TEAM_ABBREVIATION"]
    teams = df.groupby("TEAM_ID", as_index=False)["GP"].max()
    return df.reset_index(drop=True), teams


def fetch_espn():
    import time
    import requests

    headers = {"User-Agent": "Mozilla/5.0 (compatible; nba-ratings)"}
    pages, page = [], 1
    while page <= 40:
        params = {"region": "us", "lang": "en", "contentorigin": "espn",
                  "isqualified": "false", "page": page, "limit": 50,
                  "season": ESPN_SEASON, "seasontype": 2,
                  "sort": "offensive.avgPoints:desc"}
        for attempt in range(3):
            try:
                r = requests.get(ESPN_URL, params=params, headers=headers, timeout=30)
                r.raise_for_status()
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(3)
        data = r.json()
        if not data.get("athletes"):
            break
        pages.append(data)
        total_pages = (data.get("pagination") or {}).get("pages")
        if total_pages and page >= total_pages:
            break
        page += 1
        time.sleep(0.5)
    if not pages:
        raise ValueError("ESPN returned no players")
    players, teams = parse_espn(pages)
    players = fill_positions(players)
    return players, apply_team_games(players, teams)


def fetch():
    if DATA_SOURCE == "espn":
        return fetch_espn()
    return fetch_nba()


def fetch_nba():
    from nba_api.stats.endpoints import leaguedashplayerstats, leaguedashteamstats

    kw = dict(season=SEASON, season_type_all_star="Regular Season",
              per_mode_detailed="PerGame", timeout=60)
    players = leaguedashplayerstats.LeagueDashPlayerStats(**kw).get_data_frames()[0]
    teams = leaguedashteamstats.LeagueDashTeamStats(**kw).get_data_frames()[0]
    return players, teams


# ---------------------------------------------------------------- output
HTML = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NBA Player Ratings</title>
<style>
:root{--bg:#f3f4f8;--card:#fff;--fg:#14161c;--muted:#6b7280;--line:#e5e7eb;--head:#eceef3;--accent:#f97316;--accent2:#c2410c;--shadow:0 2px 10px rgba(20,22,28,.08)}
@media (prefers-color-scheme:dark){:root{--bg:#0d0f14;--card:#171a22;--fg:#eef0f5;--muted:#9aa3b2;--line:#262a35;--head:#232733;--accent2:#fb923c;--shadow:0 2px 12px rgba(0,0,0,.45)}}
*{box-sizing:border-box}
body{margin:0;font:15px/1.35 system-ui,-apple-system,sans-serif;background:var(--bg);color:var(--fg)}
.top{background:linear-gradient(135deg,#1a2550,#0d1330 55%,#6b2a0f);color:#fff;padding:18px 14px 14px}
.top-in{max-width:1100px;margin:0 auto}
.top h1{margin:0;font-size:23px}
.top .sub{color:#c7cbe0;font-size:12px;margin:3px 0 12px}
.ctl{display:flex;flex-wrap:wrap;gap:10px;align-items:center}
.ctl input[type=search]{flex:1;min-width:170px;max-width:340px;padding:9px 12px;border-radius:10px;border:0;background:rgba(255,255,255,.15);color:#fff;font-size:16px;outline:none}
.ctl input::placeholder{color:#c7cbe0}
.seg{display:flex;gap:4px;background:rgba(255,255,255,.12);padding:3px;border-radius:12px}
.seg button{border:0;background:none;color:#dfe2f3;padding:7px 11px;border-radius:9px;font-size:14px;font-weight:600;cursor:pointer}
.seg button.on{background:var(--accent);color:#fff}
.tg{display:flex;align-items:center;gap:6px;font-size:13px;color:#e6e8f5;cursor:pointer}
.tg input{accent-color:var(--accent);width:17px;height:17px}
.pos{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin-top:10px}
.pos .pl{color:#c7cbe0;font-size:13px;margin-right:2px}
.pp{position:relative;cursor:pointer}
.pp input{position:absolute;opacity:0;pointer-events:none}
.pp span{display:inline-block;padding:6px 12px;border-radius:999px;background:rgba(255,255,255,.14);color:#e6e8f5;font-size:13px;font-weight:600;border:1px solid transparent}
.pp input:checked+span{background:var(--accent);color:#fff}
.pp input:checked+span::before{content:"✓ "}
.pp input:focus-visible+span{border-color:#fff}
.main{max-width:1100px;margin:0 auto;padding:12px}
.hide{display:none}
.grid{display:grid;gap:12px;grid-template-columns:repeat(auto-fill,minmax(310px,1fr))}
.stack{display:grid;gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:16px;box-shadow:var(--shadow);overflow:hidden}
.card.first{border-color:var(--accent)}
.ch{display:flex;align-items:center;gap:8px;padding:12px 14px 4px;font-weight:700;font-size:16px}
.ch .em{font-size:21px}
.ct{margin-left:auto;font-weight:500;color:var(--muted);font-size:13px}
.lead{display:flex;align-items:center;gap:12px;margin:6px 10px;padding:10px;border-radius:12px;background:linear-gradient(135deg,rgba(249,115,22,.20),rgba(249,115,22,.05));cursor:pointer}
.lead.static{cursor:default}
.lead:not(.static):hover{background:var(--head)}
.lead .nm{flex:1;min-width:0}
.lead .nm b{font-size:17px;display:block;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.lead .nm small{color:var(--muted)}
.big{font-size:26px;font-weight:800;color:var(--accent2)}
.big small{font-size:12px;color:var(--muted);font-weight:600;margin-left:2px}
.row{display:flex;align-items:center;gap:10px;padding:6px 14px;cursor:pointer}
.row:hover{background:var(--head)}
.rk{width:18px;color:var(--muted);font-size:13px;text-align:right}
.row .nm{flex:1;min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.row .nm small{color:var(--muted);margin-left:6px}
.row .val{font-weight:700}
.row .val small{color:var(--muted);font-weight:600}
.cf{display:block;width:100%;margin-top:6px;padding:11px;border:0;border-top:1px solid var(--line);background:none;color:var(--accent2);font-weight:700;font-size:14px;cursor:pointer}
.cf:hover{background:var(--head)}
.empty{padding:16px 14px;color:var(--muted)}
.st{border:0;background:none;color:var(--muted);font-size:21px;cursor:pointer;padding:0 4px;line-height:1;flex:none}
.st.on{color:#f59e0b}
.av{width:var(--s);height:var(--s);position:relative;overflow:hidden;display:inline-flex;align-items:center;justify-content:center;flex:none;font-weight:700;color:var(--muted);font-size:calc(var(--s)*.34)}
.av.ini{background:var(--head);border-radius:50%}
.av img{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;object-position:center bottom}
.wrap{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:14px}
.wrap.mt{margin-top:12px}
table{border-collapse:separate;border-spacing:0;width:max-content;min-width:100%}
th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:center;white-space:nowrap}
th{background:var(--head);position:sticky;top:0;cursor:pointer;user-select:none;z-index:2}
.static-th th{cursor:default}
th:first-child,td:first-child{position:sticky;left:0;text-align:left;background:var(--card);z-index:1;min-width:210px}
th:first-child{background:var(--head);z-index:3}
tbody tr{cursor:pointer}
tbody tr:hover td{background:var(--head)}
.pc{display:flex;align-items:center;gap:6px}
td small{display:block;color:var(--muted);font-size:10px}
td.sc b{display:block;border-radius:6px;padding:2px 0;min-width:30px}
td.o{font-weight:700}
.tot{display:grid;grid-template-columns:repeat(6,1fr);gap:6px;margin:8px 14px 2px}
.tot div{background:var(--head);border-radius:10px;padding:8px 2px;text-align:center}
.tot b{display:block;font-size:17px}
.tot small{color:var(--muted);font-size:11px}
@media (max-width:520px){.tot{grid-template-columns:repeat(3,1fr)}}
.cap{color:var(--muted);font-size:12px;margin:6px 14px 4px}
.prof{padding:4px 14px 12px}
.foot{max-width:1100px;margin:0 auto;padding:6px 14px 28px;color:var(--muted);font-size:12px}
.ov{position:fixed;inset:0;background:rgba(0,0,0,.55);display:none;align-items:flex-end;justify-content:center;z-index:50}
.ov.show{display:flex}
.sheet{position:relative;background:var(--card);width:100%;max-width:520px;max-height:90vh;overflow:auto;border-radius:20px 20px 0 0;padding:18px 16px 22px}
@media (min-width:600px){.ov{align-items:center}.sheet{border-radius:20px}}
.x{position:absolute;top:8px;right:10px;border:0;background:var(--head);color:var(--fg);width:32px;height:32px;border-radius:50%;font-size:20px;cursor:pointer;line-height:1}
.shh{display:flex;align-items:center;gap:14px;margin-bottom:12px;padding-right:34px}
.shn{font-size:20px;font-weight:800}
.shs{color:var(--muted);font-size:13px}
.shov{margin-left:auto;text-align:right;font-size:30px;font-weight:800;color:var(--accent2);line-height:1}
.shov small{display:block;font-size:11px;color:var(--muted);font-weight:600}
.addbtn{display:block;width:100%;margin:0 0 12px;padding:10px;border-radius:12px;border:1px solid var(--accent);background:none;color:var(--accent2);font-weight:700;font-size:14px;cursor:pointer}
.addbtn:hover{background:var(--head)}
.br{display:grid;grid-template-columns:128px 1fr 24px 54px;gap:8px;align-items:center;padding:5px 0;font-size:14px}
.br.avg{grid-template-columns:128px 1fr 40px}
.bt{height:10px;background:var(--head);border-radius:6px;overflow:hidden}
.bt i{display:block;height:100%;border-radius:6px}
.bs{font-weight:800;text-align:right}
.bv{color:var(--muted);text-align:right;font-size:13px}
.note{color:var(--muted);font-size:12px;margin:12px 0 0}
</style></head><body>
<div class="top"><div class="top-in">
<h1>🏀 NBA Player Ratings</h1>
<div class="sub">Season __SEASON__ · updated __UPDATED__</div>
<div class="ctl">
<input id="q" type="search" placeholder="Search a player or team">
<div class="seg"><button id="tabL" class="on">Leaders</button><button id="tabM">My team</button><button id="tabT">Full list</button></div>
<label class="tg"><input id="qt" type="checkbox" checked> Qualified players only <span id="cnt"></span></label>
</div>
<div class="pos"><span class="pl">Position</span><label class="pp"><input type="checkbox" value="G"><span>Guard (G)</span></label><label class="pp"><input type="checkbox" value="F"><span>Forward (F)</span></label><label class="pp"><input type="checkbox" value="C"><span>Center (C)</span></label></div>
</div></div>
<div class="main">
<div id="leaders"></div>
<div id="myteam" class="hide"></div>
<div id="tablewrap" class="wrap hide"><table><thead id="h"></thead><tbody id="b"></tbody></table></div>
</div>
<div class="foot"><div id="rules"></div><div>Scores are 1 to 10 in each of 10 categories (10 = league leader). Overall is the sum, out of 100. MPG is shown for context only. Tap any player for their full breakdown, and tap ☆ to add them to My team.</div></div>
<div class="ov" id="ov"><div class="sheet" id="sheet"></div></div>
<script>
const CATS=__CATS__, ALL=__ROWS_ALL__, QUAL=__ROWS_Q__, TEAMS=__TEAMS__;
// CATS entry: [key, label, lowerIsBetter, emoji, title]
const $=id=>document.getElementById(id);
let view="L", key="overall", dir=-1;
const cur=()=>$("qt").checked?QUAL:ALL;
const esc=s=>String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const norm=s=>String(s).normalize("NFD").replace(/[\u0300-\u036f]/g,"").toLowerCase();
const hue=sc=>"hsl("+((sc-1)*13)+",70%,45%)";
const tInfo=r=>TEAMS[String(r.team).toUpperCase()]||{};
const tName=r=>tInfo(r).n||r.team;
const tText=r=>norm(r.name+" "+r.team+" "+tName(r)+" "+(tInfo(r).a||""));
const sub=r=>esc(r.team)+(r.pos?" · "+esc(r.pos):"");

/* ---------- position filter (checkboxes: G / F / C) ---------- */
const PK="nba_pos_v1", POSN={G:"Guards",F:"Forwards",C:"Centers"};
const posSel=()=>[...document.querySelectorAll(".pos input:checked")].map(i=>i.value);
function pool(){
  const s=posSel(), rows=cur();
  return s.length?rows.filter(r=>r.g&&[...r.g].some(c=>s.includes(c))):rows;
}
function posChip(){const s=posSel();return s.length?'<span class="ct">'+s.map(x=>POSN[x]).join(" + ")+'</span>':"";}
function restorePos(){try{const s=JSON.parse(localStorage.getItem(PK)||"[]");document.querySelectorAll(".pos input").forEach(i=>{i.checked=s.includes(i.value);});}catch(e){}}
function savePos(){try{localStorage.setItem(PK,JSON.stringify(posSel()));}catch(e){}}

/* ---------- my team (saved in this browser) ---------- */
const SK="nba_myteam_v1";
function loadTeam(){try{const t=JSON.parse(localStorage.getItem(SK)||"[]");return Array.isArray(t)?t.map(String):[];}catch(e){return [];}}
let team=loadTeam();
function saveTeam(){try{localStorage.setItem(SK,JSON.stringify(team));}catch(e){}}
const inTeam=id=>team.includes(id);
function toggleTeam(id){team=inTeam(id)?team.filter(x=>x!==id):[...team,id];saveTeam();}
const star=id=>'<button class="st'+(inTeam(id)?' on':'')+'" data-st="'+id+'" aria-label="Add or remove from my team">'+(inTeam(id)?'★':'☆')+'</button>';

function avFail(img){const s=img.parentNode;s.classList.add("ini");s.textContent=s.dataset.ini;}
function av(r,size){
  const ini=esc(r.name.split(" ").map(w=>w[0]).slice(0,2).join(""));
  if(!r.photo)return '<span class="av ini" style="--s:'+size+'px">'+ini+'</span>';
  return '<span class="av" data-ini="'+ini+'" style="--s:'+size+'px"><img loading="lazy" alt="" src="'+esc(r.photo)+'" onerror="avFail(this)"></span>';
}

/* ---------- search (players AND teams) ---------- */
function prow(r){
  return '<div class="row" data-id="'+r.id+'">'+av(r,36)+'<span class="nm">'+esc(r.name)+'<small>'+sub(r)+'</small></span><span class="val">'+r.overall+'<small>/100</small></span>'+star(r.id)+'</div>';
}
function searchHTML(q){
  const rows=pool(), nq=norm(q.trim());
  const teams={};
  rows.forEach(r=>{(teams[tName(r)]=teams[tName(r)]||[]).push(r);});
  const mt=Object.keys(teams).filter(n=>{const r=teams[n][0];return norm(n+" "+r.team+" "+(tInfo(r).a||"")).includes(nq);});
  const byTeam=mt.length>=1&&mt.length<=3;
  const shown=new Set();
  let h="";
  if(byTeam){
    mt.forEach(n=>{
      const list=teams[n].slice().sort((a,b)=>b.overall-a.overall);
      list.forEach(r=>shown.add(r.id));
      h+='<section class="card"><div class="ch">🏀 '+esc(n)+'<span class="ct">'+list.length+' players</span></div>'+list.map(prow).join("")+'</section>';
    });
  }
  const cand=byTeam?rows.filter(r=>!shown.has(r.id)&&norm(r.name).includes(nq)):rows.filter(r=>tText(r).includes(nq));
  const pm=cand.slice().sort((a,b)=>b.overall-a.overall).slice(0,30);
  if(pm.length)h+='<section class="card"><div class="ch">Players</div>'+pm.map(prow).join("")+'</section>';
  if(!h)h='<section class="card"><div class="empty">No players or teams found.'+($("qt").checked?' Try turning off “Qualified players only”.':'')+'</div></section>';
  return '<div class="stack">'+h+'</div>';
}

/* ---------- Leaders (front page) ---------- */
function topList(rows,k,lower){
  return rows.filter(r=>r.n[k]!==null).sort((a,b)=>{
    if(a.s[k]!==b.s[k])return b.s[k]-a.s[k];
    if(a.n[k]!==b.n[k])return lower?a.n[k]-b.n[k]:b.n[k]-a.n[k];
    return b.overall-a.overall;
  }).slice(0,10);
}
function card(em,title,list,val,k,cls){
  if(!list.length)return "";
  const f=list[0];
  let h='<section class="card '+(cls||"")+'"><div class="ch"><span class="em">'+em+'</span>'+title+posChip()+'</div>';
  h+='<div class="lead" data-id="'+f.id+'">'+av(f,72)+'<div class="nm"><b>'+esc(f.name)+'</b><small>'+sub(f)+' · '+f.mpg+' MPG</small></div><div class="big">'+val(f)+'</div></div>';
  list.slice(1).forEach((r,i)=>{
    h+='<div class="row" data-id="'+r.id+'"><span class="rk">'+(i+2)+'</span>'+av(r,36)+'<span class="nm">'+esc(r.name)+'<small>'+sub(r)+'</small></span><span class="val">'+val(r)+'</span></div>';
  });
  return h+'<button class="cf" data-k="'+k+'">Full ranking →</button></section>';
}
function renderLeaders(){
  const rows=pool(), q=$("q").value.trim();
  let h="";
  if(!rows.length){h='<section class="card"><div class="empty">No players match the selected positions. Position data comes from ESPN, and some players may be missing it.</div></section>';}
  else if(q){h=searchHTML(q);}
  else{
    const overall=[...rows].sort((a,b)=>b.overall-a.overall).slice(0,10);
    h='<div class="grid">'+card("👑","Overall rating",overall,r=>r.overall+'<small>/100</small>',"overall","first");
    CATS.forEach(c=>{h+=card(c[3],c[4],topList(rows,c[0],c[2]),r=>r.v[c[0]],c[0]);});
    h+='</div>';
  }
  $("leaders").innerHTML=h;
}

/* ---------- table rows (Full list + My team) ---------- */
const cols=[["name","Player"],["team","Team"],["pos","Pos"],["mpg","MPG"],["gp","GP"],["overall","Overall"],...CATS.map(c=>[c[0],c[1]])];
function val(r,k){return k in r.s?r.s[k]:(k==="mpg"?parseFloat(r.mpg):r[k]);}
function cmp(a,b){
  if(key in a.s){
    if(a.s[key]!==b.s[key])return a.s[key]-b.s[key];
    const lower=CATS.find(c=>c[0]===key)[2],x=a.n[key],y=b.n[key];
    if(x===null&&y===null)return 0;if(x===null)return -1;if(y===null)return 1;
    return lower?y-x:x-y;
  }
  const x=val(a,key),y=val(b,key);return x>y?1:x<y?-1:0;
}
function trow(r){
  return '<tr data-id="'+r.id+'"><td><span class="pc">'+star(r.id)+av(r,28)+'<span>'+esc(r.name)+'</span></span></td><td>'+esc(r.team)+'</td><td>'+esc(r.pos||"—")+'</td><td>'+r.mpg+'</td><td>'+r.gp+'</td><td class="o">'+r.overall+'</td>'+
    CATS.map(c=>'<td class="sc"><b style="background:'+hue(r.s[c[0]]).replace("hsl","hsla").replace(")",",.35)")+'">'+r.s[c[0]]+'</b><small>'+r.v[c[0]]+'</small></td>').join("")+'</tr>';
}
function renderTable(){
  $("h").innerHTML="<tr>"+cols.map(c=>'<th data-k="'+c[0]+'">'+c[1]+(c[0]===key?(dir<0?" ▼":" ▲"):"")+"</th>").join("")+"</tr>";
  document.querySelectorAll("#h th").forEach(t=>t.onclick=()=>{const k=t.dataset.k;if(k===key)dir=-dir;else{key=k;dir=(k==="name"||k==="team")?1:-1;}renderTable();});
  const nq=norm($("q").value.trim());
  const rows=pool().filter(r=>tText(r).includes(nq)).sort((a,b)=>cmp(a,b)*dir);
  $("b").innerHTML=rows.map(trow).join("");
}

/* ---------- My team ---------- */
function summary(mine){
  const n=mine.length;
  const avgO=(mine.reduce((s,r)=>s+r.overall,0)/n).toFixed(1);
  const avg=c=>mine.reduce((s,r)=>s+r.s[c[0]],0)/n;
  const ranked=CATS.map(c=>({c:c,a:avg(c)})).sort((x,y)=>y.a-x.a);
  const lab=o=>o.c[3]+" "+o.c[1];
  const best=ranked.slice(0,2).map(lab).join(", "), worst=ranked.slice(-2).reverse().map(lab).join(", ");
  const tot=k=>mine.reduce((s,r)=>s+(r.n[k]||0),0).toFixed(1);
  let h='<section class="card"><div class="ch"><span class="em">⭐</span>My team<span class="ct">'+n+' player'+(n===1?"":"s")+'</span></div>';
  h+='<div class="lead static"><div class="nm"><b>Average overall</b><small>Strongest: '+best+'<br>Weakest: '+worst+'</small></div><div class="big">'+avgO+'<small>/100</small></div></div>';
  h+='<div class="cap">Combined per-game totals</div><div class="tot">'+
    [["ppg","PTS"],["rpg","REB"],["apg","AST"],["spg","STL"],["bpg","BLK"],["tov","TO"]].map(p=>'<div><b>'+tot(p[0])+'</b><small>'+p[1]+'</small></div>').join("")+'</div>';
  h+='<div class="cap">Average category score (1 to 10)</div><div class="prof">'+
    CATS.map(c=>{const a=avg(c);return '<div class="br avg"><span>'+c[3]+' '+c[1]+'</span><span class="bt"><i style="width:'+(a*10)+'%;background:'+hue(a)+'"></i></span><span class="bs">'+a.toFixed(1)+'</span></div>';}).join("")+'</div></section>';
  return h;
}
function renderTeam(){
  const rows=cur(), byId={};
  rows.forEach(r=>{byId[r.id]=r;});
  const q=$("q").value.trim();
  const mine=team.map(id=>byId[id]).filter(Boolean).sort((a,b)=>b.overall-a.overall);
  const missing=team.filter(id=>!byId[id]).length;
  let h="";
  if(q)h+='<div class="cap" style="margin-left:2px">Tap ☆ to add players to your team</div>'+searchHTML(q)+'<div style="height:12px"></div>';
  if(!team.length){
    h+='<section class="card"><div class="empty">Your team is empty. Type a player or a team (try “Lakers”) in the search box above, then tap ☆ next to a player to add them.</div></section>';
  }else{
    if(mine.length)h+=summary(mine);
    if(mine.length){
      h+='<div class="wrap mt"><table class="static-th"><thead><tr>'+cols.map(c=>'<th>'+c[1]+'</th>').join("")+'</tr></thead><tbody>'+mine.map(trow).join("")+'</tbody></table></div>';
    }
    if(missing)h+='<p class="note">'+missing+' saved player'+(missing===1?" isn’t":"s aren’t")+' rated in this view (not qualified, or not in the data). Turn off “Qualified players only” to see them.</p>';
    h+='<p class="note">Your team is saved in this browser on this device. Ratings update every night.</p>';
  }
  $("myteam").innerHTML=h;
}

/* ---------- Player sheet ---------- */
function openPlayer(id){
  const r=cur().find(x=>x.id===id)||ALL.find(x=>x.id===id);
  if(!r)return;
  let h='<button class="x" id="cl" aria-label="Close">×</button>';
  h+='<div class="shh">'+av(r,72)+'<div><div class="shn">'+esc(r.name)+'</div><div class="shs">'+esc(tName(r))+(r.pos?" · "+esc(r.pos):"")+' · '+r.mpg+' MPG · '+r.gp+' GP</div></div><div class="shov">'+r.overall+'<small>of 100</small></div></div>';
  h+='<button class="addbtn" data-st="'+r.id+'">'+(inTeam(r.id)?"★ On my team · tap to remove":"☆ Add to my team")+'</button>';
  CATS.forEach(c=>{const sc=r.s[c[0]];
    h+='<div class="br"><span>'+c[3]+' '+c[1]+'</span><span class="bt"><i style="width:'+(sc*10)+'%;background:'+hue(sc)+'"></i></span><span class="bs">'+sc+'</span><span class="bv">'+r.v[c[0]]+'</span></div>';});
  h+='<p class="note">'+($("qt").checked?"Ranked among qualified players only.":"Ranked against all players.")+'</p>';
  $("sheet").innerHTML=h;$("ov").classList.add("show");
}
const closeSheet=()=>$("ov").classList.remove("show");

/* ---------- wiring ---------- */
const HOLD={L:"Search a player or team",M:"Search to add players (name or team)",T:"Filter by player or team"};
function setView(v){
  view=v;
  $("leaders").classList.toggle("hide",v!=="L");
  $("myteam").classList.toggle("hide",v!=="M");
  $("tablewrap").classList.toggle("hide",v!=="T");
  $("tabL").classList.toggle("on",v==="L");$("tabM").classList.toggle("on",v==="M");$("tabT").classList.toggle("on",v==="T");
  $("q").placeholder=HOLD[v];
  draw();
}
function draw(){
  $("cnt").textContent="("+cur().length+")";
  if(view==="L")renderLeaders();else if(view==="M")renderTeam();else renderTable();
}
document.addEventListener("click",e=>{
  const st=e.target.closest("[data-st]");
  if(st){
    const id=st.dataset.st,fromSheet=st.classList.contains("addbtn");
    toggleTeam(id);draw();
    if(fromSheet)openPlayer(id);
    return;
  }
  const cf=e.target.closest(".cf");
  if(cf){key=cf.dataset.k;dir=-1;setView("T");window.scrollTo(0,0);return;}
  if(e.target.id==="ov"||e.target.id==="cl"){closeSheet();return;}
  if(e.target.closest("th"))return;
  const el=e.target.closest("[data-id]");
  if(el)openPlayer(el.dataset.id);
});
document.addEventListener("keydown",e=>{if(e.key==="Escape")closeSheet();});
$("q").oninput=draw;$("qt").onchange=draw;
restorePos();document.querySelectorAll(".pos input").forEach(i=>{i.onchange=()=>{savePos();draw();};});
$("tabL").onclick=()=>setView("L");$("tabM").onclick=()=>setView("M");$("tabT").onclick=()=>setView("T");
$("rules").textContent="__RULES__";
draw();
</script></body></html>"""


def write_html(rows_all, rows_q, path="index.html"):
    cats = [[c[0], c[1], c[3], *CATEGORY_INFO[c[0]]] for c in CATEGORIES]
    a = MIN_ATTEMPTS_PER_GAME
    rules = (f"Qualified = at least {MIN_GP_PCT:.0%} of team games and {MIN_MPG:g} MPG. "
             f"Shooting % needs {a['FG_PCT']:g} FGA, {a['FT_PCT']:g} FTA, "
             f"{a['FG3_PCT']:g} 3PA per game.")
    html = (HTML.replace("__SEASON__", SEASON)
                .replace("__CATS__", json.dumps(cats))
                .replace("__RULES__", rules)
                .replace("__TEAMS__", json.dumps(TEAM_LOOKUP))
                .replace("__UPDATED__", datetime.now(ZoneInfo("America/Toronto")).strftime("%b %d, %I:%M %p ET"))
                .replace("__ROWS_ALL__", json.dumps(rows_all))
                .replace("__ROWS_Q__", json.dumps(rows_q)))
    Path(path).write_text(html, encoding="utf-8")
    print(f"Wrote {path} ({len(rows_all)} players, {len(rows_q)} qualified)")


if __name__ == "__main__":
    try:
        players_df, teams_df = fetch()
    except Exception as e:  # NBA.com often blocks cloud IPs; ESPN feed can change
        sys.exit(f"Could not get the stats: {e}")
    rows_all = compute_ratings(players_df, teams_df)
    rows_q = compute_ratings(players_df, teams_df, qualified_only=True)
    counts = {}
    for r in rows_all:
        counts[r["g"] or "none"] = counts.get(r["g"] or "none", 0) + 1
    print("Positions found:", counts)
    write_html(rows_all, rows_q)
