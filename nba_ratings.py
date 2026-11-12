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
from datetime import date, datetime, timedelta
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
TEAM_ABBR = {name: canon for canon, name, _extra, _others in _TEAMS}   # "Boston Celtics" -> "BOS"
TEAM_LOOKUP = {code: {"n": name, "a": extra}
               for canon, name, extra, others in _TEAMS
               for code in (canon, *others)}


ESPN_PHOTO = "https://a.espncdn.com/i/headshots/nba/players/full/{}.png"


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
        if photo == ESPN_PHOTO.format(r["PLAYER_ID"]):
            photo = ""   # the page rebuilds this standard link itself
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


def fetch_team_games(season=None):
    """Real games played by each team, from ESPN's standings (wins + losses)."""
    import requests

    url = "https://site.api.espn.com/apis/v2/sports/basketball/nba/standings"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; nba-ratings)"}
    data = requests.get(url, params={"season": season or ESPN_SEASON}, headers=headers,
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


def apply_team_games(players, teams, season=None):
    """Replace the 'most games any teammate played' guess with the real count."""
    try:
        real = fetch_team_games(season)
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


def fetch_espn(season=None):
    import time
    import requests

    headers = {"User-Agent": "Mozilla/5.0 (compatible; nba-ratings)"}
    pages, page = [], 1
    while page <= 40:
        params = {"region": "us", "lang": "en", "contentorigin": "espn",
                  "isqualified": "false", "page": page, "limit": 50,
                  "season": season or ESPN_SEASON, "seasontype": 2,
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
    return players, apply_team_games(players, teams, season)


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


# ---------------------------------------------------------------- extras
# Injuries, this/next week's schedule, last-5-games form, and week-ago history.
# Each piece is optional: if one fails, the rest of the page still works.
ESPN_SITE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
HISTORY_FILE = "history.json"   # nightly ratings snapshots (for rank-change arrows)
GAMES_FILE = "games.json"       # saved box scores (for last-5-games form)
HISTORY_KEEP_DAYS = 21
GAMES_KEEP_DAYS = 35
BACKFILL_DAYS = 30
FORM_GAMES = 5
FORM_KEYS = ["ppg", "rpg", "apg", "spg", "bpg", "tov"]
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; nba-ratings)"}


def _get_json(url, params=None, retries=2):
    import time
    import requests

    for attempt in range(retries + 1):
        try:
            r = requests.get(url, params=params, headers=_HEADERS, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception:
            if attempt == retries:
                raise
            time.sleep(2)


def load_json_file(path, default):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return default


def save_json_file(path, obj):
    Path(path).write_text(json.dumps(obj, separators=(",", ":")), encoding="utf-8")


def _athlete_id(a):
    if not isinstance(a, dict):
        return None
    if a.get("id"):
        return str(a["id"])
    parts = [str(a.get("$ref") or ""), str(a.get("uid") or ""), str(a.get("guid") or "")]
    parts += [str(link.get("href") or "") for link in a.get("links", []) if isinstance(link, dict)]
    blob = " ".join(parts)
    m = (re.search(r"/athletes/(\d+)", blob) or re.search(r"a:(\d+)", blob)
         or re.search(r"/id/(\d+)", blob))
    return m.group(1) if m else None


def _athlete_name(a):
    if not isinstance(a, dict):
        return str(a or "")
    full = " ".join(x for x in (a.get("firstName"), a.get("lastName")) if x)
    return a.get("displayName") or a.get("fullName") or full or a.get("shortName") or ""


def _text(v):
    if isinstance(v, dict):
        return str(v.get("name") or v.get("description") or v.get("detail") or "")
    return str(v) if v else ""


def _collect_injuries(node, out):
    """Find every injury entry in ESPN's JSON, however deeply it is nested."""
    if isinstance(node, dict):
        a = node.get("athlete")
        if isinstance(a, dict):
            detail = _text(node.get("shortComment"))
            d = node.get("details")
            if not detail and isinstance(d, dict):
                detail = " ".join(x for x in (d.get("type"), d.get("detail")) if x)
            status = _text(node.get("status")) or _text(node.get("type"))
            if not status and detail:
                status = "Injured"
            out.append({"id": _athlete_id(a), "name": _athlete_name(a),
                        "s": status, "d": detail[:160]})
        for v in node.values():
            _collect_injuries(v, out)
    elif isinstance(node, list):
        for v in node:
            _collect_injuries(v, out)


def fetch_injuries():
    """Injury entries from ESPN: league-wide report, else team by team."""
    import time

    recs = []
    data = _get_json(ESPN_SITE + "/injuries")
    _collect_injuries(data, recs)
    if not recs:  # backup: ask each team for its own report
        teams = _get_json(ESPN_SITE + "/teams")
        for t in teams["sports"][0]["leagues"][0]["teams"]:
            _collect_injuries(_get_json(f"{ESPN_SITE}/teams/{t['team']['id']}/injuries"), recs)
            time.sleep(0.2)
    if not recs:
        raise ValueError("no injury entries found in ESPN's data. What it sent: "
                         + json.dumps(data)[:300])
    return recs


def _norm_name(s):
    import unicodedata

    t = unicodedata.normalize("NFD", str(s)).encode("ascii", "ignore").decode().lower()
    t = re.sub(r"[^a-z ]", "", t)
    return " ".join(w for w in t.split() if w not in ("jr", "sr", "ii", "iii", "iv"))


def match_injuries(recs, rows):
    """{app player id: {"s": status, "d": detail}}, matched by id, else by name."""
    ids = {r["id"] for r in rows}
    by_name = {}
    for r in rows:
        by_name.setdefault(_norm_name(r["name"]), []).append(r["id"])
    out = {}
    for rec in recs:
        pid = rec["id"] if rec["id"] in ids else None
        if pid is None:
            cands = by_name.get(_norm_name(rec["name"]), [])
            pid = cands[0] if len(cands) == 1 else None
        if pid and rec["s"]:
            out[pid] = {"s": rec["s"], "d": rec["d"]}
    return out


def fetch_events(day, cache):
    """Games on one day (cached for this run): teams, finished?, season type."""
    import time

    key = day.strftime("%Y%m%d")
    if key not in cache:
        data = _get_json(ESPN_SITE + "/scoreboard", {"dates": key, "limit": 100})
        events = []
        for e in data.get("events", []):
            comp = (e.get("competitions") or [{}])[0]
            comps = []
            for c in comp.get("competitors", []):
                t = c.get("team") or {}
                comps.append({"t": team_key(t.get("abbreviation") or t.get("displayName") or ""),
                              "h": c.get("homeAway") == "home", "sc": c.get("score")})
            teams = [c["t"] for c in comps]
            status = (e.get("status") or {}).get("type") or {}
            events.append({"id": str(e.get("id")), "teams": teams, "comps": comps,
                           "done": bool(status.get("completed")),
                           "type": (e.get("season") or {}).get("type")})
        cache[key] = events
        time.sleep(0.2)
    return cache[key]


def build_schedule(today, cache):
    """Games per team for this week and next (Monday to Sunday)."""
    monday = today - timedelta(days=today.weekday())
    weeks = []
    for w in range(2):
        start = monday + timedelta(days=7 * w)
        end = start + timedelta(days=6)
        teams = {}
        for i in range(7):
            for ev in fetch_events(start + timedelta(days=i), cache):
                if ev["type"] not in (None, 2):
                    continue
                comps = ev.get("comps") or []
                for c in comps:
                    if not c["t"]:
                        continue
                    opp = next((o for o in comps if o is not c), None)
                    entry = {"s": 2 if ev["done"] else 1,
                             "o": TEAM_ABBR.get(opp["t"], opp["t"]) if opp else "",
                             "h": 1 if c["h"] else 0}
                    if ev["done"] and opp:
                        try:
                            mine, theirs = int(float(c["sc"])), int(float(opp["sc"]))
                            entry["w"] = 1 if mine > theirs else 0
                            entry["p"] = f"{mine}-{theirs}"
                        except (TypeError, ValueError):
                            pass
                    teams.setdefault(c["t"], [0] * 7)[i] = entry
        label = (f"{'This' if w == 0 else 'Next'} week: {start:%b} {start.day} to "
                 f"{end:%b} {end.day} (Mon to Sun)")
        weeks.append({"label": label, "teams": teams})
    return weeks


def fetch_boxscore(event_id):
    """{player id: [min, pts, reb, ast, stl, blk, tov]} for one finished game."""
    data = _get_json(ESPN_SITE + "/summary", {"event": event_id})
    lines = {}
    for team in (data.get("boxscore") or {}).get("players", []):
        for block in team.get("statistics", []):
            labels = [str(x).upper() for x in (block.get("labels") or block.get("names") or [])]
            if "TOV" in labels and "TO" not in labels:
                labels[labels.index("TOV")] = "TO"
            idx = {k: (labels.index(k) if k in labels else None)
                   for k in ("MIN", "PTS", "REB", "AST", "STL", "BLK", "TO")}
            if idx["MIN"] is None or idx["PTS"] is None:
                raise ValueError(f"unexpected box score columns: {labels}")
            for a in block.get("athletes", []):
                stats = a.get("stats") or []
                pid = (a.get("athlete") or {}).get("id")
                if not pid or not stats or a.get("didNotPlay"):
                    continue

                def num(k):
                    i = idx[k]
                    if i is None or i >= len(stats):
                        return 0.0
                    m = re.match(r"-?\d+(\.\d+)?", str(stats[i]))
                    return float(m.group()) if m else 0.0

                if num("MIN") > 0:
                    lines[str(pid)] = [num(k) for k in ("MIN", "PTS", "REB", "AST", "STL", "BLK", "TO")]
    return lines


def update_games(today, cache):
    """Save box scores of newly finished games (only fetches what's new)."""
    import time

    games = load_json_file(GAMES_FILE, {}).get("games", {})
    start = today - timedelta(days=BACKFILL_DAYS)
    if games:
        start = max(start, date.fromisoformat(max(g["d"] for g in games.values())) - timedelta(days=1))
    new, first_error, day = 0, None, start
    while day <= today:
        for ev in fetch_events(day, cache):
            if ev["type"] not in (None, 2) or not ev["done"] or ev["id"] in games:
                continue
            try:
                lines = fetch_boxscore(ev["id"])
            except Exception as e:
                first_error = first_error or e
                continue
            if lines:
                games[ev["id"]] = {"d": day.isoformat(), "p": lines}
                new += 1
            time.sleep(0.25)
        day += timedelta(days=1)
    cutoff = (today - timedelta(days=GAMES_KEEP_DAYS)).isoformat()
    games = {k: v for k, v in games.items() if v["d"] >= cutoff}
    save_json_file(GAMES_FILE, {"games": games})
    print(f"Box scores saved: {len(games)} games ({new} new)")
    if first_error:
        print("Warning: some box scores could not be read:", first_error)
    return games


def compute_form(rows_q, games):
    """Last-5-games production vs season, scored with the app's 1-10 tiers."""
    by_player = {}
    for g in sorted(games.values(), key=lambda g: g["d"]):
        for pid, line in g["p"].items():
            by_player.setdefault(pid, []).append(line)
    ids, last5, season = [], [], []
    for r in rows_q:
        recent = by_player.get(r["id"], [])[-FORM_GAMES:]
        sv = [r["n"][k] for k in FORM_KEYS]
        if len(recent) < FORM_GAMES or any(v is None for v in sv):
            continue
        ids.append(r["id"])
        last5.append([round(sum(x[i] for x in recent) / len(recent), 1) for i in range(1, 7)])
        season.append(sv)
    if len(ids) < 10:
        return {}
    dl = pd.DataFrame(last5, columns=FORM_KEYS)
    ds = pd.DataFrame(season, columns=FORM_KEYS)
    idx_l = sum(rate_column(dl[k], k == "tov") for k in FORM_KEYS)
    idx_s = sum(rate_column(ds[k], k == "tov") for k in FORM_KEYS)
    return {pid: {"a": last5[i], "l": int(idx_l[i]), "s": int(idx_s[i]),
                  "d": int(idx_l[i] - idx_s[i])} for i, pid in enumerate(ids)}


def make_snapshot(rows_all, rows_q):
    q = {r["id"]: r["overall"] for r in rows_q}
    return {r["id"]: [r["overall"], q.get(r["id"])] + [r["n"][c[0]] for c in CATEGORIES]
            for r in rows_all}


def update_history(snapshot, today):
    hist = load_json_file(HISTORY_FILE, {})
    hist[today.isoformat()] = snapshot
    cutoff = (today - timedelta(days=HISTORY_KEEP_DAYS)).isoformat()
    hist = {k: v for k, v in hist.items() if k >= cutoff}
    save_json_file(HISTORY_FILE, hist)
    return hist


def pick_baseline(hist, today, back=7):
    """The saved snapshot closest to `back` days ago (or the oldest we have)."""
    older = sorted(k for k in hist if k < today.isoformat())
    if not older:
        return None
    target = (today - timedelta(days=back)).isoformat()
    ok = [k for k in older if k <= target]
    key = ok[-1] if ok else older[0]
    d = date.fromisoformat(key)
    return {"days": (today - d).days, "date": f"{d:%b} {d.day}", "p": hist[key]}


PREV_FILE = "prev_season.json"


def build_previous_season():
    """Last season's final stats, rated the same way (saved after the first run)."""
    if DATA_SOURCE != "espn":
        return None
    year = ESPN_SEASON - 1
    settings = [MIN_GP_PCT, MIN_MPG, MIN_ATTEMPTS_PER_GAME, [c[0] for c in CATEGORIES], 2]
    cached = load_json_file(PREV_FILE, None)
    if cached and cached.get("season") == year and cached.get("settings") == settings:
        return cached
    players, teams = fetch_espn(year)
    out = {"season": year, "label": f"{year - 1}-{str(year)[2:]}", "settings": settings,
           "all": compute_ratings(players, teams),
           "q": compute_ratings(players, teams, qualified_only=True)}
    save_json_file(PREV_FILE, out)
    return out


def build_extras(rows_all, rows_q, today):
    ex = {"inj": {}, "sched": [], "form": {}, "base": None}
    cache = {}
    try:
        recs = fetch_injuries()
        ex["inj"] = match_injuries(recs, rows_all)
        print(f"Injuries: {len(recs)} in ESPN's report, {len(ex['inj'])} matched to players in the app")
        if recs and not ex["inj"]:
            print("  Sample from the report (ids/names did not match):", recs[:3])
    except Exception as e:
        print("Warning: could not load injuries:", e)
    try:
        ex["sched"] = build_schedule(today, cache)
        print("Schedule: teams playing this week:", len(ex["sched"][0]["teams"]),
              "| next week:", len(ex["sched"][1]["teams"]))
    except Exception as e:
        print("Warning: could not load the schedule:", e)
    try:
        games = update_games(today, cache)
        ex["form"] = compute_form(rows_q, games)
        print(f"Hot/cold form calculated for {len(ex['form'])} players")
    except Exception as e:
        print("Warning: could not build hot/cold form:", e)
    try:
        if rows_all:
            hist = update_history(make_snapshot(rows_all, rows_q), today)
            d1, d7 = pick_baseline(hist, today, 1), pick_baseline(hist, today, 7)
            ex["base"] = {"d1": d1, "d7": d7} if (d1 or d7) else None
            print("History:", (f"yesterday's comparison = {d1['days']} day(s) ago, weekly comparison = {d7['days']} day(s) ago"
                               if ex["base"] else "first snapshot saved, arrows start tomorrow"))
    except Exception as e:
        print("Warning: could not save history:", e)
    return ex


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
.seg button{border:0;background:none;color:#dfe2f3;padding:7px 9px;border-radius:9px;font-size:13.5px;font-weight:600;cursor:pointer}
.seg button.on{background:var(--accent);color:#fff}
.tg{display:flex;align-items:center;gap:6px;font-size:13px;color:#e6e8f5;cursor:pointer}
.tg input{accent-color:var(--accent);width:17px;height:17px}
.ssel select{background:rgba(255,255,255,.15);color:#fff;border:0;border-radius:10px;padding:7px 10px;font-size:14px;font-weight:600;margin-bottom:10px}
.ssel option{color:#000}
.pos{display:flex;flex-wrap:nowrap;align-items:center;gap:4px;margin-top:10px;overflow-x:auto}
.pos .pl{color:#c7cbe0;font-size:11.5px;flex:none}
.pp{position:relative;cursor:pointer;flex:none}
.pp input{position:absolute;opacity:0;pointer-events:none}
.pp span{display:inline-block;padding:5px 7px;white-space:nowrap;border-radius:999px;background:rgba(255,255,255,.14);color:#e6e8f5;font-size:12.5px;font-weight:600;border:1px solid transparent}
.pp input:checked+span{background:var(--accent);color:#fff}
.pp input:focus-visible+span{border-color:#fff}
.main.wide{max-width:1500px}
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
.hwrap{position:sticky;top:0;z-index:6;overflow:hidden;background:var(--head);border:1px solid var(--line);border-bottom:0;border-radius:14px 14px 0 0}
.bwrap{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-top:0;border-radius:0 0 14px 14px}
.ft{table-layout:fixed;min-width:100%}
.hwrap th{border-bottom:1px solid var(--line)}
table{border-collapse:separate;border-spacing:0;width:max-content;min-width:100%}
th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:center;white-space:nowrap}
th{background:var(--head);position:sticky;top:0;cursor:pointer;user-select:none;z-index:2}
.static-th th,#myteam th{cursor:default}
th:first-child,td:first-child{position:sticky;left:0;background:var(--card);z-index:1;padding:4px 6px;border-right:1px solid var(--line)}
th:first-child{background:var(--head);z-index:3}
td.nmc,th:nth-child(2){text-align:left;min-width:170px}
td.nmc{overflow:hidden}
.nmc .pc{min-width:0}
.nmt{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;min-width:0}
.ib{flex:none}
th{overflow:hidden;text-overflow:ellipsis;font-size:14px}
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
.abr{display:flex;gap:8px;margin:0 0 12px}
.abr .addbtn{flex:1;width:auto;margin:0;padding:7px 6px;font-size:13px}
.br{display:grid;grid-template-columns:128px 1fr 24px 54px;gap:8px;align-items:center;padding:5px 0;font-size:14px}
.br.avg{grid-template-columns:128px 1fr 40px}
.bt{height:10px;background:var(--head);border-radius:6px;overflow:hidden}
.bt i{display:block;height:100%;border-radius:6px}
.bs{font-weight:800;text-align:right}
.bv{color:var(--muted);text-align:right;font-size:13px}
.note{color:var(--muted);font-size:12px;margin:12px 0 0}
.ib{display:inline-block;font-size:10px;font-weight:800;padding:1px 5px;border-radius:6px;margin-right:5px;vertical-align:1px;color:#fff;background:#d97706;letter-spacing:.3px}
.ib.out{background:#dc2626}
.mv{font-size:11px;font-weight:700;margin-left:6px;white-space:nowrap}
.mv.up{color:#16a34a}.mv.dn{color:#dc2626}
.big .mv{font-size:13px}
.row .val{display:inline-flex;align-items:baseline;justify-content:flex-end;flex:none}
.val .mv,.big .mv{margin:0 6px 0 0}
.big{white-space:nowrap}
.ch{white-space:nowrap}
.ttl{overflow:hidden;text-overflow:ellipsis;min-width:0}
.lbar{display:flex;flex-wrap:wrap;gap:6px 10px;align-items:center;margin:0 2px 10px;font-size:13px;color:var(--muted)}
.fbar{flex:1 1 100%}
.lbar .pill{padding:4px 10px;font-size:12px}
.pb{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin:8px 14px 2px}
.pb div{background:var(--head);border-radius:10px;padding:8px 2px;text-align:center}
.pb b{display:block;font-size:17px}
.pb small{color:var(--muted);font-size:11px}
.sug{margin:6px 14px 8px;font-size:13px}
.sh{margin-top:12px}
.tm{display:grid;grid-template-columns:190px auto 22px;justify-content:start;gap:2px 10px;align-items:center;padding:4px 14px}
.tm .tn{font-size:14px}
.wk{display:grid;grid-template-columns:repeat(7,54px);gap:4px}
.wk.hd span{text-align:center;font-size:10px;color:var(--muted)}
.dc{display:flex;flex-direction:column;align-items:center;gap:3px;min-height:27px}
.dc small{font-size:10px;color:var(--muted);line-height:1;white-space:nowrap}
.dc small.sc{font-size:9px;font-weight:700;letter-spacing:-.2px}
.dc small.sc.w{color:#16a34a}
.dc small.sc.l{color:#dc2626}
@media (max-width:560px){.tm{grid-template-columns:1fr 22px;justify-content:stretch}.tm .wk{grid-column:1/-1;grid-template-columns:repeat(7,1fr)}}
.rs{display:grid;gap:12px}
.cin{padding:6px 14px 8px}
.cin input{width:100%;padding:9px 12px;border-radius:10px;border:1px solid var(--line);background:var(--bg);color:var(--fg);font-size:16px;outline:none}
#cres .row{padding:6px 4px}
.cg{display:grid;gap:6px 8px;padding:6px 14px 14px;align-items:center}
.cc{text-align:center;position:relative;display:flex;flex-direction:column;align-items:center;gap:2px;font-size:13px}
.cc b{line-height:1.15}
.cc small{color:var(--muted);font-size:11px}
.rmv{position:absolute;z-index:3;top:-2px;right:0;border:0;background:var(--head);color:var(--fg);width:22px;height:22px;border-radius:50%;cursor:pointer;font-size:14px;line-height:1}
.cl{font-size:13px;color:var(--muted)}
.cv{text-align:center;padding:4px 2px;border-radius:8px;font-size:13px}
.cv.w{background:rgba(34,197,94,.18);font-weight:700}
.cv .bt{display:block;margin:0 auto 3px;max-width:80px}
.cv small{color:var(--muted)}
.pill{border:1px solid var(--line);background:none;color:var(--fg);padding:5px 12px;border-radius:999px;font-size:13px;font-weight:600;cursor:pointer}
.pill.on{background:var(--accent);border-color:var(--accent);color:#fff}
.mini{display:flex;gap:6px;padding:4px 14px}
.gh{padding:8px 14px 2px;font-weight:700;font-size:13px;color:var(--accent2)}
.tr2{display:flex;align-items:center;gap:10px;padding:4px 14px}
.tr2 .tn{flex:1;font-size:14px}
.gc{width:18px;text-align:right}
.dots{display:flex;gap:4px}
.dl span{width:11px;text-align:center;font-size:9px;color:var(--muted)}
.dot{width:11px;height:11px;border-radius:50%;border:2px solid var(--line)}
.dot.u{background:var(--accent);border-color:var(--accent)}
.dot.d{background:var(--muted);border-color:var(--muted)}
.fl{display:block;color:var(--muted);font-size:12px}
.val.up{color:#16a34a}.val.dn{color:#dc2626}
.inj{margin:-4px 0 10px;font-size:13px;color:var(--muted)}
</style></head><body>
<div class="top"><div class="top-in">
<h1>🏀 NBA Player Ratings</h1>
<div class="sub" id="sub">Season __SEASON__ · updated __UPDATED__</div>
<div class="ssel" id="sselwrap" style="display:none"><select id="ssel" aria-label="Season"></select></div>
<div class="ctl">
<input id="q" type="search" placeholder="Search a player or team">
<div class="seg"><button id="tabL" class="on">Leaders</button><button id="tabM">My team</button><button id="tabR">Research</button><button id="tabT">Full list</button></div>
<label class="tg"><input id="qt" type="checkbox" checked> Qualified players only <span id="cnt"></span></label>
</div>
<div class="pos"><span class="pl">Position</span><label class="pp"><input type="checkbox" value="G"><span>Guard (G)</span></label><label class="pp"><input type="checkbox" value="F"><span>Forward (F)</span></label><label class="pp"><input type="checkbox" value="C"><span>Center (C)</span></label></div>
<label class="tg" id="prkwrap" style="display:none;margin-top:8px"><input id="prk" type="checkbox"> Re-rank within selected positions</label>
</div></div>
<div class="main">
<div id="leaders"></div>
<div id="myteam" class="hide"></div>
<div id="research" class="hide"></div>
<div id="tablewrap" class="hide">
<div id="hwrap" class="hwrap"><table class="ft" id="t1"><colgroup id="cg1"></colgroup><thead id="h"></thead></table></div>
<div id="bwrap" class="bwrap"><table class="ft" id="t2"><colgroup id="cg2"></colgroup><tbody id="b"></tbody></table></div>
</div>
</div>
<div class="foot"><div id="rules"></div><div id="mvnote"></div><div>Scores are 1 to 10 in each of 10 categories (10 = league leader). Overall is the sum, out of 100. MPG is shown for context only. Tap any player for their full breakdown, and tap ☆ to add them to My team. With a position selected, “Re-rank within selected positions” scores each player only against others at that position.</div></div>
<div class="ov" id="ov"><div class="sheet" id="sheet"></div></div>
<script>
const CATS=__CATS__, DATA=__DATA__, TEAMS=__TEAMS__;
let season="cur", ALL=DATA.cur.all, QUAL=DATA.cur.q;
const isCur=()=>season==="cur";
const ESPNP="https://a.espncdn.com/i/headshots/nba/players/full/";
const INJ=__INJ__, SCHED=__SCHED__, FORM=__FORM__, BASES=__BASE__;
let trend="d1", BASE=null;
try{trend=localStorage.getItem("nba_trend_v1")||"d1";}catch(e){}
function pickBase(){BASE=BASES?(BASES[trend]||BASES.d7||BASES.d1||null):null;}
pickBase();
const agoText=d=>d===1?"yesterday":d+" days ago";
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
const PK="nba_pos_v1", PRK="nba_prk_v1", POSN={G:"Guards",F:"Forwards",C:"Centers"};
const posSel=()=>[...document.querySelectorAll(".pos input:checked")].map(i=>i.value);
const rel=()=>posSel().length>0&&$("prk").checked;
const posNames=()=>posSel().map(x=>POSN[x]).join(" + ");
/* same scoring as the script: dense ranking, 10 = leader, 9 = ranks 2-5, 8 = 6-10, ... 1 = outside the top 40 */
function tierScore(rank){if(rank===1)return 10;if(rank<=5)return 9;if(rank>=41)return 1;return 8-Math.floor((rank-6)/5);}
function denseScores(vals,lower){
  const uniq=[...new Set(vals.filter(v=>v!==null&&v!==undefined))].sort((a,b)=>lower?a-b:b-a);
  const rk=new Map(uniq.map((v,i)=>[v,i+1]));
  return vals.map(v=>(v===null||v===undefined)?1:tierScore(rk.get(v)));
}
function rescore(rows){
  const sc=CATS.map(c=>denseScores(rows.map(r=>r.n[c[0]]),c[2]));
  return rows.map((r,i)=>{const s={};let t=0;CATS.forEach((c,j)=>{s[c[0]]=sc[j][i];t+=sc[j][i];});return Object.assign({},r,{s:s,overall:t});});
}
function pool(){
  const s=posSel(), rows=cur();
  if(!s.length)return rows;
  const sub=rows.filter(r=>r.g&&[...r.g].some(c=>s.includes(c)));
  return $("prk").checked?rescore(sub):sub;
}
function posChip(){const s=posSel();return s.length?'<span class="ct">'+posNames()+(rel()?' · ranked among them':'')+'</span>':"";}
function restorePos(){try{const s=JSON.parse(localStorage.getItem(PK)||"[]");document.querySelectorAll(".pos input").forEach(i=>{i.checked=s.includes(i.value);});$("prk").checked=localStorage.getItem(PRK)==="1";}catch(e){}}
function savePos(){try{localStorage.setItem(PK,JSON.stringify(posSel()));localStorage.setItem(PRK,$("prk").checked?"1":"0");}catch(e){}}

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
  return '<span class="av" data-ini="'+ini+'" style="--s:'+size+'px"><img loading="lazy" alt="" src="'+esc(r.photo||(ESPNP+r.id+".png"))+'" onerror="avFail(this)"></span>';
}

/* ---------- search (players AND teams) ---------- */
function prow(r){
  return '<div class="row" data-id="'+r.id+'">'+av(r,36)+'<span class="nm">'+esc(r.name)+'<small>'+ib(r.id)+sub(r)+'</small></span><span class="val">'+r.overall+'<small>/100</small></span>'+star(r.id)+'</div>';
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

/* ---------- injury badges + rank movement ---------- */
function ibt(s){const t=String(s||"").toLowerCase();return t.includes("out")?"OUT":t.includes("day")?"DTD":t.includes("question")?"Q":t.includes("doubt")?"D":t.includes("prob")?"P":s?String(s).slice(0,4).toUpperCase():"";}
function ib(id){const i=isCur()?INJ[id]:null;if(!i||!i.s)return "";const t=ibt(i.s);return '<span class="ib'+(t==="OUT"?" out":"")+'" title="'+esc(i.s+(i.d?": "+i.d:""))+'">'+t+'</span>';}
/* when scores are re-ranked within a position, last time's overall is re-ranked the same way so the comparison is like for like */
function relBaseOverall(rows){
  const have=rows.filter(r=>BASE.p[r.id]);
  const sc=CATS.map((c,j)=>denseScores(have.map(r=>BASE.p[r.id][2+j]),c[2])), m={};
  have.forEach((r,i)=>{m[r.id]=sc.reduce((t,a)=>t+a[i],0);});
  return m;
}
function overallDelta(r){
  if(!BASE||!isCur())return null;
  const b=BASE.p[r.id];if(!b)return null;
  const bo=rel()?relBaseOverall(pool())[r.id]:($("qt").checked?b[1]:b[0]);
  return(bo===null||bo===undefined)?null:r.overall-bo;
}
function moves(rows,k,lower){
  if(!BASE||!isCur())return null;
  const idx=k==="overall"?-1:CATS.findIndex(c=>c[0]===k), both=[];
  const rb=(k==="overall"&&rel())?relBaseOverall(rows):null;
  rows.forEach(r=>{
    const b=BASE.p[r.id];if(!b)return;
    const cv=k==="overall"?r.overall:r.n[k], bv=k==="overall"?(rb?rb[r.id]:($("qt").checked?b[1]:b[0])):b[2+idx];
    if(cv===null||cv===undefined||bv===null||bv===undefined)return;
    both.push({id:r.id,c:cv,b:bv});
  });
  const better=(x,y)=>lower?x<y:x>y, out={};
  both.forEach(p=>{let cr=1,br=1;both.forEach(o=>{if(better(o.c,p.c))cr++;if(better(o.b,p.b))br++;});out[p.id]=br-cr;});
  return out;
}
function mvh(mv,id){
  const d=mv&&mv[id];if(!d)return "";
  return '<span class="mv '+(d>0?"up":"dn")+'" title="Rank change vs '+agoText(BASE.days)+'">'+(d>0?"▲":"▼")+Math.abs(d)+'</span>';
}

/* ---------- Leaders (front page) ---------- */
function topList(rows,k,lower){
  return rows.filter(r=>r.n[k]!==null).sort((a,b)=>{
    if(a.s[k]!==b.s[k])return b.s[k]-a.s[k];
    if(a.n[k]!==b.n[k])return lower?a.n[k]-b.n[k]:b.n[k]-a.n[k];
    return b.overall-a.overall;
  }).slice(0,10);
}
function card(em,title,list,val,k,cls,mv){
  if(!list.length)return "";
  const f=list[0];
  let h='<section class="card '+(cls||"")+'"><div class="ch"><span class="em">'+em+'</span><span class="ttl">'+title+'</span></div>';
  h+='<div class="lead" data-id="'+f.id+'">'+av(f,72)+'<div class="nm"><b>'+esc(f.name)+'</b><small>'+ib(f.id)+sub(f)+' · '+f.mpg+' MPG</small></div><div class="big">'+mvh(mv,f.id)+val(f)+'</div></div>';
  list.slice(1).forEach((r,i)=>{
    h+='<div class="row" data-id="'+r.id+'"><span class="rk">'+(i+2)+'</span>'+av(r,36)+'<span class="nm">'+esc(r.name)+'<small>'+ib(r.id)+sub(r)+'</small></span><span class="val">'+mvh(mv,r.id)+val(r)+'</span></div>';
  });
  return h+'<button class="cf" data-k="'+k+'">Full ranking →</button></section>';
}
function lbar(){
  let h="";
  if(posSel().length)h+='<div class="fbar">Showing '+posNames().toLowerCase()+' only · '+(rel()?'scores re-ranked among them':'scores still compare with the whole league')+'</div>';
  if(BASES&&isCur())h+='<span>Trend arrows:</span>'+["d1","d7"].filter(k=>BASES[k]).map(k=>'<button class="pill'+(trend===k?' on':'')+'" data-tr="'+k+'">'+(k==="d1"?"vs yesterday":"vs last week")+'</button>').join("");
  return h?'<div class="lbar">'+h+'</div>':"";
}
function renderLeaders(){
  const rows=pool(), q=$("q").value.trim();
  let h="";
  if(!rows.length){h='<section class="card"><div class="empty">No players match the selected positions. Position data comes from ESPN, and some players may be missing it.</div></section>';}
  else if(q){h=searchHTML(q);}
  else{
    const overall=[...rows].sort((a,b)=>b.overall-a.overall).slice(0,10);
    h=lbar()+'<div class="grid">'+card("👑","Overall rating",overall,r=>r.overall+'<small>/100</small>',"overall","first",moves(rows,"overall",false));
    CATS.forEach(c=>{h+=card(c[3],c[4],topList(rows,c[0],c[2]),r=>r.v[c[0]],c[0],"",moves(rows,c[0],c[2]));});
    h+='</div>';
  }
  $("leaders").innerHTML=h;
}

/* ---------- table rows (Full list + My team) ---------- */
const COLW=[46,270,64,48,52,44,92,...CATS.map(()=>60)];
function layoutTable(){
  const cg=COLW.map(w=>'<col style="width:'+w+'px">').join(""), tw=COLW.reduce((a,b)=>a+b,0)+"px";
  ["cg1","cg2"].forEach(id=>{$(id).innerHTML=cg;});
  ["t1","t2"].forEach(id=>{$(id).style.minWidth=tw;$(id).style.width="100%";});
  $("bwrap").addEventListener("scroll",()=>{$("hwrap").scrollLeft=$("bwrap").scrollLeft;});
}
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
  return '<tr data-id="'+r.id+'"><td class="pk">'+av(r,32)+'</td><td class="nmc"><span class="pc">'+star(r.id)+'<span class="nmt" title="'+esc(r.name)+'">'+esc(r.name)+'</span>'+ib(r.id)+'</span></td><td>'+esc(r.team)+'</td><td>'+esc(r.pos||"—")+'</td><td>'+r.mpg+'</td><td>'+r.gp+'</td><td class="o">'+r.overall+'</td>'+
    CATS.map(c=>'<td class="sc"><b style="background:'+hue(r.s[c[0]]).replace("hsl","hsla").replace(")",",.35)")+'">'+r.s[c[0]]+'</b><small>'+r.v[c[0]]+'</small></td>').join("")+'</tr>';
}
function renderTable(){
  $("h").innerHTML='<tr><th class="pk"></th>'+cols.map(c=>'<th data-k="'+c[0]+'">'+c[1]+(c[0]===key?(dir<0?" ▼":" ▲"):"")+"</th>").join("")+"</tr>";
  document.querySelectorAll("#h th[data-k]").forEach(t=>t.onclick=()=>{const k=t.dataset.k;if(k===key)dir=-dir;else{key=k;dir=(k==="name"||k==="team")?1:-1;}renderTable();});
  const nq=norm($("q").value.trim());
  const rows=pool().filter(r=>tText(r).includes(nq)).sort((a,b)=>cmp(a,b)*dir);
  $("b").innerHTML=rows.map(trow).join("");
}

/* ---------- My team ---------- */
function posBox(mine){
  const cnt={G:0,F:0,C:0}, w={G:0,F:0,C:0}, raw={};
  let hyb=0, unk=0;
  mine.forEach(r=>{
    const g=(r.g||"").split("").filter(Boolean);
    if(!g.length)unk++;
    if(g.length>1)hyb++;
    g.forEach(c=>{cnt[c]++;w[c]+=1/g.length;});
    if(r.pos)raw[r.pos]=(raw[r.pos]||0)+1;
  });
  const ORD=["PG","SG","G","G-F","SF","PF","F","F-C","C"], rk=k=>{const i=ORD.indexOf(k);return i<0?99:i;};
  const rawTxt=Object.keys(raw).sort((a,b)=>rk(a)-rk(b)||a.localeCompare(b)).map(k=>k+" "+raw[k]);
  if(hyb)rawTxt.push(hyb+" player"+(hyb===1?"":"s")+" listed at two positions count in both");
  if(unk)rawTxt.push(unk+" without a listed position");
  const n=mine.length, tot=w.G+w.F+w.C;
  const LONG={G:"guard",F:"forward",C:"center"}, WHY={G:"assists, steals and 3-point shooting",F:"a mix of scoring and rebounding",C:"rebounds, blocks and field goal %"};
  let sug;
  if(n<4||tot===0)sug="Add a few more players to get a position check.";
  else{
    const ord=["G","F","C"].sort((a,b)=>w[a]-w[b]), lo=ord[0], hi=ord[2];
    const lop=w[hi]/tot>=0.6||(w[lo]===0&&n>=5);
    sug=lop?"⚖️ Your team leans "+LONG[hi]+" ("+cnt[hi]+" of "+n+" players). Consider adding a "+LONG[lo]+" for "+WHY[lo]+".":"✅ Your position mix looks balanced.";
  }
  return '<div class="cap">Position breakdown</div><div class="pb">'+["G","F","C"].map(c=>'<div><b>'+cnt[c]+'</b><small>'+POSN[c]+'</small></div>').join("")+'</div>'+
    (rawTxt.length?'<div class="cap">'+rawTxt.join(" · ")+'</div>':'')+'<div class="sug">'+sug+'</div>';
}
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
  const hurt=isCur()?mine.filter(r=>INJ[r.id]&&INJ[r.id].s):[];
  if(hurt.length)h+='<div class="cap">⚠️ Injury report: '+hurt.map(r=>esc(r.name)+' ('+ibt(INJ[r.id].s)+')').join(", ")+'</div>';
  h+=posBox(mine);
  h+='<div class="cap">Combined per-game totals</div><div class="tot">'+
    [["ppg","PTS"],["rpg","REB"],["apg","AST"],["spg","STL"],["bpg","BLK"],["tov","TO"]].map(p=>'<div><b>'+tot(p[0])+'</b><small>'+p[1]+'</small></div>').join("")+'</div>';
  h+='<div class="cap">Average category score (1 to 10)</div><div class="prof">'+
    CATS.map(c=>{const a=avg(c);return '<div class="br avg"><span>'+c[3]+' '+c[1]+'</span><span class="bt"><i style="width:'+(a*10)+'%;background:'+hue(a)+'"></i></span><span class="bs">'+a.toFixed(1)+'</span></div>';}).join("")+'</div></section>';
  return h;
}
const TW=COLW.reduce((a,b)=>a+b,0), CGH=COLW.map(w=>'<col style="width:'+w+'px">').join("");
function shell(id,head,body){
  const st='style="min-width:'+TW+'px;width:100%"';
  return '<div class="sh"><div class="hwrap" id="'+id+'h"><table class="ft" '+st+'><colgroup>'+CGH+'</colgroup><thead>'+head+'</thead></table></div><div class="bwrap" id="'+id+'b"><table class="ft" '+st+'><colgroup>'+CGH+'</colgroup><tbody>'+body+'</tbody></table></div></div>';
}
function syncShell(id){const b=$(id+"b");if(b)b.addEventListener("scroll",()=>{$(id+"h").scrollLeft=b.scrollLeft;});}
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
      h+=shell("mt",'<tr><th class="pk"></th>'+cols.map(c=>'<th>'+c[1]+'</th>').join("")+'</tr>',mine.map(trow).join(""));
    }
    if(missing)h+='<p class="note">'+missing+' saved player'+(missing===1?" isn’t":"s aren’t")+' rated in this view (not qualified, or not in the data). Turn off “Qualified players only” to see them.</p>';
    h+='<p class="note">Your team is saved in this browser on this device. Ratings update every night.</p>';
  }
  $("myteam").innerHTML=h;
  syncShell("mt");
}

/* ---------- Research tab: compare, games this week, hot and cold ---------- */
let cmpIds=[], wk=0;
const findRow=id=>cur().find(r=>r.id===id)||ALL.find(r=>r.id===id);
function addCmp(id){if(cmpIds.includes(id))return;if(cmpIds.length>=3)cmpIds.shift();cmpIds.push(id);}
function cmpSearch(v){
  const box=$("cres"),nq=norm(v.trim());
  if(!box)return;
  if(!nq){box.innerHTML="";return;}
  const m=cur().filter(r=>tText(r).includes(nq)).sort((a,b)=>b.overall-a.overall).slice(0,6);
  box.innerHTML=m.length?m.map(r=>'<div class="row" data-cmp="'+r.id+'">'+av(r,32)+'<span class="nm">'+esc(r.name)+'<small>'+ib(r.id)+sub(r)+'</small></span><span class="val">'+r.overall+'<small>/100</small></span></div>').join(""):'<div class="empty">No matches</div>';
}
function cmpTable(rows){
  const n=rows.length, maxO=Math.max(...rows.map(r=>r.overall));
  const best=c=>{const v=rows.filter(r=>r.n[c[0]]!==null).map(r=>r.n[c[0]]);return v.length?(c[2]?Math.min(...v):Math.max(...v)):null;};
  let h='<div class="cg" style="grid-template-columns:92px repeat('+n+',1fr)"><div></div>';
  rows.forEach(r=>{h+='<div class="cc"><button class="rmv" data-rm="'+r.id+'" aria-label="Remove">×</button>'+av(r,56)+'<b>'+esc(r.name)+'</b><small>'+sub(r)+'</small>'+ib(r.id)+'</div>';});
  h+='<div class="cl">Overall</div>';
  rows.forEach(r=>{h+='<div class="cv'+(n>1&&r.overall===maxO?' w':'')+'"><b>'+r.overall+'</b><small>/100</small></div>';});
  h+='<div class="cl">MPG · GP</div>';
  rows.forEach(r=>{h+='<div class="cv"><small>'+r.mpg+' · '+r.gp+'</small></div>';});
  CATS.forEach(c=>{
    const b=best(c);
    h+='<div class="cl">'+c[3]+' '+c[1]+'</div>';
    rows.forEach(r=>{const sc=r.s[c[0]], win=n>1&&b!==null&&r.n[c[0]]===b;
      h+='<div class="cv'+(win?' w':'')+'"><span class="bt"><i style="width:'+(sc*10)+'%;background:'+hue(sc)+'"></i></span><b>'+sc+'</b> <small>'+r.v[c[0]]+'</small></div>';});
  });
  return h+'</div>';
}
function weekCard(){
  let h='<section class="card"><div class="ch"><span class="em">📅</span>Games this week</div>';
  if(!isCur())return h+'<div class="empty">Games this week is for the current season. Switch the season at the top of the page.</div></section>';
  if(!SCHED.length||!Object.keys(SCHED[wk].teams).length)return h+'<div class="empty">The schedule isn’t available yet. It will appear after the next update.</div></section>';
  const w=SCHED[wk];
  h+='<div class="mini">'+SCHED.map((x,i)=>'<button class="pill'+(i===wk?' on':'')+'" data-wk="'+i+'">'+(i===0?"This week":"Next week")+'</button>').join("")+'</div><div class="cap">'+esc(w.label)+'</div>';
  const es=Object.keys(w.teams).map(t=>({t:t,d:w.teams[t],n:w.teams[t].filter(x=>x).length})).sort((a,b)=>b.n-a.n||a.t.localeCompare(b.t));
  h+='<div class="tm"><span></span><div class="wk hd">'+["Mon","Tue","Wed","Thu","Fri","Sat","Sun"].map(x=>'<span>'+x+'</span>').join("")+'</div><span></span></div>';
  let last=-1;
  es.forEach(e=>{
    if(e.n!==last){h+='<div class="gh">'+e.n+' game'+(e.n===1?"":"s")+'</div>';last=e.n;}
    h+='<div class="tm"><span class="tn">'+esc(e.t)+'</span><div class="wk">'+e.d.map(x=>'<div class="dc"><span class="dot'+(x?(x.s===2?' d':' u'):'')+'"></span><small>'+(x?(x.h?"":"@")+esc(x.o):"")+'</small>'+(x&&x.s===2&&x.p?'<small class="sc '+(x.w?'w':'l')+'">'+(x.w?'W':'L')+' '+esc(x.p)+'</small>':'')+'</div>').join("")+'</div><b class="gc">'+e.n+'</b></div>';
  });
  const max=es[0].n, top=new Set(es.filter(e=>e.n===max).map(e=>e.t));
  const ps=pool().filter(r=>top.has(tName(r))).sort((a,b)=>b.overall-a.overall).slice(0,8);
  h+='<div class="gh" style="margin-top:8px">Best players on '+max+'-game teams</div>'+ps.map(prow).join("");
  return h+'<div class="cap">Orange dots are upcoming games, gray dots are games already played. “@” means an away game. Finished games show W or L and the final score, with the team’s own score first.</div></section>';
}
function frow(x){
  const r=x.r,a=x.f.a;
  return '<div class="row" data-id="'+r.id+'">'+av(r,36)+'<span class="nm">'+esc(r.name)+'<small>'+ib(r.id)+sub(r)+'</small><span class="fl">'+a[0]+' pts · '+a[1]+' reb · '+a[2]+' ast</span></span><span class="val '+(x.f.d>0?"up":"dn")+'">'+(x.f.d>0?"+":"")+x.f.d+'</span>'+star(r.id)+'</div>';
}
function formCard(){
  let h='<section class="card"><div class="ch"><span class="em">🌡️</span>Hot and cold (last 5 games)</div>';
  if(!isCur())return h+'<div class="empty">Hot and cold is for the current season. Switch the season at the top of the page.</div></section>';
  const byId={};pool().forEach(r=>{byId[r.id]=r;});
  const list=Object.keys(FORM).filter(id=>byId[id]).map(id=>({r:byId[id],f:FORM[id]}));
  if(!list.length)return h+'<div class="empty">The hot and cold list will appear once the app has saved each player’s last 5 games. It builds up over the first update or two.</div></section>';
  const hot=list.filter(x=>x.f.d>0).sort((a,b)=>b.f.d-a.f.d).slice(0,6);
  const cold=list.filter(x=>x.f.d<0).sort((a,b)=>a.f.d-b.f.d).slice(0,6);
  h+='<div class="gh">🔥 Hot</div>'+(hot.map(frow).join("")||'<div class="empty">None right now.</div>');
  h+='<div class="gh">🧊 Cold</div>'+(cold.map(frow).join("")||'<div class="empty">None right now.</div>');
  return h+'<div class="cap">Form compares the last 5 games with the season, using the same 1-10 scoring on points, rebounds, assists, steals, blocks and turnovers. +9 means about 9 more rating points (out of 60) than usual. Shooting percentages are left out because 5 games is too few to trust.</div></section>';
}
function renderResearch(){
  const prev=$("cq")?$("cq").value:"";
  const rows=cmpIds.map(findRow).filter(Boolean);
  let c='<section class="card"><div class="ch"><span class="em">⚖️</span>Compare players<span class="ct">up to 3</span></div><div class="cin"><input id="cq" type="search" placeholder="Search a player to add" autocomplete="off"><div id="cres"></div></div>';
  c+=rows.length?cmpTable(rows)+(rows.length<2?'<div class="cap">Add one more player to compare.</div>':''):'<div class="empty">Search above to pick two or three players.</div>';
  c+='</section>';
  $("research").innerHTML='<div class="rs">'+c+weekCard()+formCard()+'</div>';
  if(prev){$("cq").value=prev;cmpSearch(prev);}
}

/* ---------- Player sheet ---------- */
function openPlayer(id){
  const r=pool().find(x=>x.id===id)||cur().find(x=>x.id===id)||ALL.find(x=>x.id===id);
  if(!r)return;
  let h='<button class="x" id="cl" aria-label="Close">×</button>';
  const od=overallDelta(r);
  const dl=(od===null||od===0)?"":' · <span class="mv '+(od>0?"up":"dn")+'" style="margin:0">'+(od>0?"▲ +":"▼ −")+Math.abs(od)+' vs '+agoText(BASE.days)+'</span>';
  const ij=isCur()?INJ[r.id]:null;
  h+='<div class="shh">'+av(r,72)+'<div><div class="shn">'+esc(r.name)+'</div><div class="shs">'+esc(tName(r))+(r.pos?" · "+esc(r.pos):"")+' · '+r.mpg+' MPG · '+r.gp+' GP</div></div><div class="shov">'+r.overall+'<small>of 100'+dl+'</small></div></div>';
  if(ij&&ij.s)h+='<div class="inj">'+ib(r.id)+esc(ij.s+(ij.d?": "+ij.d:""))+'</div>';
  h+='<div class="abr"><button class="addbtn" data-st="'+r.id+'">'+(inTeam(r.id)?"★ On my team":"☆ Add to my team")+'</button><button class="addbtn" data-cmpopen="'+r.id+'">⚖️ Compare players</button></div>';
  CATS.forEach(c=>{const sc=r.s[c[0]];
    h+='<div class="br"><span>'+c[3]+' '+c[1]+'</span><span class="bt"><i style="width:'+(sc*10)+'%;background:'+hue(sc)+'"></i></span><span class="bs">'+sc+'</span><span class="bv">'+r.v[c[0]]+'</span></div>';});
  h+='<p class="note">'+(rel()?"Ranked only among "+posNames().toLowerCase()+($("qt").checked?" (qualified players).":"."):$("qt").checked?"Ranked among qualified players only.":"Ranked against all players.")+'</p>';
  $("sheet").innerHTML=h;$("ov").classList.add("show");
}
const closeSheet=()=>$("ov").classList.remove("show");

/* ---------- wiring ---------- */
const HOLD={L:"Search a player or team",M:"Search to add players (name or team)",R:"",T:"Filter by player or team"};
function setView(v){
  view=v;
  $("leaders").classList.toggle("hide",v!=="L");
  $("myteam").classList.toggle("hide",v!=="M");
  document.querySelector(".main").classList.toggle("wide",v==="T"||v==="M");
  $("research").classList.toggle("hide",v!=="R");
  $("tablewrap").classList.toggle("hide",v!=="T");
  $("tabL").classList.toggle("on",v==="L");$("tabM").classList.toggle("on",v==="M");$("tabR").classList.toggle("on",v==="R");$("tabT").classList.toggle("on",v==="T");
  $("q").placeholder=HOLD[v];$("q").style.display=v==="R"?"none":"";
  draw();
}
function draw(){
  $("cnt").textContent="("+cur().length+")";
  $("prkwrap").style.display=posSel().length?"":"none";
  if(view==="L")renderLeaders();else if(view==="M")renderTeam();else if(view==="R")renderResearch();else renderTable();
}
document.addEventListener("click",e=>{
  const rmv=e.target.closest("[data-rm]");
  if(rmv){cmpIds=cmpIds.filter(x=>x!==rmv.dataset.rm);renderResearch();return;}
  const ca=e.target.closest("[data-cmp]");
  if(ca){addCmp(ca.dataset.cmp);if($("cq"))$("cq").value="";renderResearch();return;}
  const co=e.target.closest("[data-cmpopen]");
  if(co){addCmp(co.dataset.cmpopen);closeSheet();setView("R");window.scrollTo(0,0);return;}
  const tb=e.target.closest("[data-tr]");
  if(tb){trend=tb.dataset.tr;try{localStorage.setItem("nba_trend_v1",trend);}catch(x){}pickBase();updateNote();draw();return;}
  const wb=e.target.closest("[data-wk]");
  if(wb){wk=+wb.dataset.wk;renderResearch();return;}
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
layoutTable();
$("q").oninput=draw;$("qt").onchange=draw;
restorePos();document.querySelectorAll(".pos input,#prk").forEach(i=>{i.onchange=()=>{savePos();draw();};});
$("tabL").onclick=()=>setView("L");$("tabM").onclick=()=>setView("M");$("tabR").onclick=()=>setView("R");$("tabT").onclick=()=>setView("T");
document.addEventListener("input",e=>{if(e.target.id==="cq")cmpSearch(e.target.value);});
$("rules").textContent="__RULES__";
const SUBCUR=$("sub").textContent;
function updateNote(){
  $("mvnote").textContent=!isCur()?"Showing final "+DATA.prev.label+" stats, rated the same way. Rank-change arrows, injuries, the schedule and hot/cold are current-season only.":BASE?"Arrows show rank change vs "+agoText(BASE.days)+" ("+BASE.date+"). Badges show the ESPN injury report.":"Rank-change arrows appear once the app has saved a day or two of history. Badges show the ESPN injury report.";
}
function setSeason(k){
  season=k;ALL=DATA[k].all;QUAL=DATA[k].q;
  $("sub").textContent=k==="cur"?SUBCUR:"Season "+DATA.prev.label+" · final stats";
  updateNote();draw();
}
if(DATA.prev){
  $("ssel").innerHTML='<option value="cur">'+DATA.cur.label+' season (current)</option><option value="prev">'+DATA.prev.label+' season (last year)</option>';
  $("sselwrap").style.display="";
  $("ssel").onchange=()=>setSeason($("ssel").value);
}
updateNote();
draw();
</script></body></html>"""


def write_html(rows_all, rows_q, path="index.html", extras=None, prev=None):
    ex = {"inj": {}, "sched": [], "form": {}, "base": None, **(extras or {})}
    data = {"cur": {"label": SEASON, "all": rows_all, "q": rows_q}, "prev": None}
    if prev:
        data["prev"] = {"label": prev["label"], "all": prev["all"], "q": prev["q"]}
    cats = [[c[0], c[1], c[3], *CATEGORY_INFO[c[0]]] for c in CATEGORIES]
    a = MIN_ATTEMPTS_PER_GAME
    rules = (f"Qualified = at least {MIN_GP_PCT:.0%} of team games and {MIN_MPG:g} MPG. "
             f"Shooting % needs {a['FG_PCT']:g} FGA, {a['FT_PCT']:g} FTA, "
             f"{a['FG3_PCT']:g} 3PA per game.")
    html = (HTML.replace("__SEASON__", SEASON)
                .replace("__CATS__", json.dumps(cats))
                .replace("__RULES__", rules)
                .replace("__TEAMS__", json.dumps(TEAM_LOOKUP))
                .replace("__INJ__", json.dumps(ex["inj"]))
                .replace("__SCHED__", json.dumps(ex["sched"]))
                .replace("__FORM__", json.dumps(ex["form"]))
                .replace("__BASE__", json.dumps(ex["base"]))
                .replace("__UPDATED__", datetime.now(ZoneInfo("America/Toronto")).strftime("%b %d, %I:%M %p ET"))
                .replace("__DATA__", json.dumps(data)))
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
    today = datetime.now(ZoneInfo("America/Toronto")).date()
    prev = None
    try:
        prev = build_previous_season()
        if prev:
            print(f"Last season ({prev['label']}): {len(prev['all'])} players rated")
    except Exception as e:
        print("Warning: could not load last season (the season menu will be hidden):", e)
    write_html(rows_all, rows_q, extras=build_extras(rows_all, rows_q, today), prev=prev)
