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
        rows.append({
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
            row = {
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
    return parse_espn(pages)


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
HTML = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NBA Player Ratings</title>
<style>
:root{--bg:#fff;--fg:#1a1a1a;--muted:#6b7280;--line:#e5e7eb;--head:#f3f4f6}
@media (prefers-color-scheme:dark){:root{--bg:#121212;--fg:#eee;--muted:#9ca3af;--line:#2a2a2a;--head:#1e1e1e}}
*{box-sizing:border-box}
body{margin:0;font:14px system-ui,sans-serif;background:var(--bg);color:var(--fg)}
header{padding:12px;position:sticky;left:0}
h1{font-size:18px;margin:0 0 4px}
.sub{color:var(--muted);font-size:12px;margin-bottom:8px}
input[type=search]{width:100%;max-width:320px;padding:8px;border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--fg);font-size:16px}
.wrap{overflow-x:auto}
table{border-collapse:separate;border-spacing:0;width:max-content;min-width:100%}
th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:center;white-space:nowrap}
th{background:var(--head);position:sticky;top:0;cursor:pointer;user-select:none;z-index:2}
th:first-child,td:first-child{position:sticky;left:0;text-align:left;background:var(--bg);z-index:1;min-width:150px}
th:first-child{background:var(--head);z-index:3}
td small{display:block;color:var(--muted);font-size:10px}
td.sc b{display:block;border-radius:6px;padding:2px 0;min-width:30px}
.ov{font-weight:700}
.tg{display:block;margin:8px 0 4px}
</style></head><body>
<header><h1>NBA Player Ratings</h1>
<div class="sub">Season __SEASON__ · updated __UPDATED__ · tap a column to sort · scores 1-10, overall out of 100 · MPG is context only</div>
<input id="q" type="search" placeholder="Search player or team">
<label class="tg"><input id="qt" type="checkbox" checked> Qualified players only <span id="cnt"></span></label>
<div class="sub" id="rules"></div></header>
<div class="wrap"><table><thead id="h"></thead><tbody id="b"></tbody></table></div>
<script>
const CATS=__CATS__, ALL=__ROWS_ALL__, QUAL=__ROWS_Q__;
let key="overall", dir=-1;
const cols=[["name","Player"],["team","Team"],["mpg","MPG"],["gp","GP"],["overall","Overall"],...CATS.map(c=>[c[0],c[1]])];
function val(r,k){return k in r.s?r.s[k]:(k==="mpg"?parseFloat(r.mpg):r[k]);}
function head(){document.getElementById("h").innerHTML="<tr>"+cols.map(c=>`<th data-k="${c[0]}">${c[1]}${c[0]===key?(dir<0?" ▼":" ▲"):""}</th>`).join("")+"</tr>";
document.querySelectorAll("th").forEach(t=>t.onclick=()=>{const k=t.dataset.k;if(k===key)dir=-dir;else{key=k;dir=(k==="name"||k==="team")?1:-1;}draw();});}
function cmp(a,b){
if(key in a.s){
if(a.s[key]!==b.s[key])return a.s[key]-b.s[key];
const lower=CATS.find(c=>c[0]===key)[2],x=a.n[key],y=b.n[key];
if(x===null&&y===null)return 0;if(x===null)return -1;if(y===null)return 1;
return lower?y-x:x-y;}
const x=val(a,key),y=val(b,key);return x>y?1:x<y?-1:0;}
function draw(){head();const q=document.getElementById("q").value.toLowerCase();
const ROWS=document.getElementById("qt").checked?QUAL:ALL;document.getElementById("cnt").textContent="("+ROWS.length+" players)";
const rows=ROWS.filter(r=>(r.name+" "+r.team).toLowerCase().includes(q)).sort((a,b)=>cmp(a,b)*dir);
document.getElementById("b").innerHTML=rows.map(r=>`<tr><td>${r.name}</td><td>${r.team}</td><td>${r.mpg}</td><td>${r.gp}</td><td class="ov">${r.overall}</td>`+
CATS.map(c=>`<td class="sc"><b style="background:hsla(${(r.s[c[0]]-1)*13},70%,45%,.35)">${r.s[c[0]]}</b><small>${r.v[c[0]]}</small></td>`).join("")+"</tr>").join("");}
document.getElementById("q").oninput=draw;document.getElementById("qt").onchange=draw;
document.getElementById("rules").textContent="__RULES__";draw();
</script></body></html>"""


def write_html(rows_all, rows_q, path="index.html"):
    cats = [[c[0], c[1], c[3]] for c in CATEGORIES]
    a = MIN_ATTEMPTS_PER_GAME
    rules = (f"Qualified = at least {MIN_GP_PCT:.0%} of team games and {MIN_MPG:g} MPG. "
             f"Shooting % needs {a['FG_PCT']:g} FGA, {a['FT_PCT']:g} FTA, "
             f"{a['FG3_PCT']:g} 3PA per game.")
    html = (HTML.replace("__SEASON__", SEASON)
                .replace("__CATS__", json.dumps(cats))
                .replace("__RULES__", rules)
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
    write_html(compute_ratings(players_df, teams_df),
               compute_ratings(players_df, teams_df, qualified_only=True))
