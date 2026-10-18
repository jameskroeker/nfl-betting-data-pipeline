"""Scheduled NFL pull: results + lines, line history, and merge of completed weeks.

Run by .github/workflows/nfl_pull.yml (or locally with API_SPORTS_KEY and ODDS_API_KEY set).

Each run:
  1. Pulls the season's games from API-Sports (1 call): schedule, status, scores, OT, venue.
  2. Pulls current DraftKings lines from The-Odds-API (3 credits). Lines only exist for games
     that haven't kicked off, so a game keeps the last line captured before kickoff.
  3. Updates data/weekly/{season}_W{n}.csv for every week not yet in the master that starts
     within the next 9 days, and appends each captured line to data/lines_history/.
  4. Merges each fully completed week into the master, in order, via merge_weeks.py
     (which refuses incomplete weeks and stops on any validation failure).
"""
import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import merge_weeks  # noqa: E402

SEASON = int(os.environ.get("SEASON", "2026"))
MASTER = "data/nfl_master_2022_2025.parquet"
WEEKLY_DIR = "data/weekly"
HISTORY_DIR = "data/lines_history"
FINAL = {"FT", "AOT"}
LOOKAHEAD_DAYS = 9
WEEKLY_COLS = ["game_id", "wk", "kickoff_et", "home_team_name", "away_team_name", "status", "home_total",
               "away_total", "home_ot", "away_ot", "venue_name", "venue_city", "closing_spread", "game_total",
               "home_moneyline", "away_moneyline", "bookmaker_name", "line_pulled_utc"]


# ---------------------------------------------------------------- sources

def fetch_games(key: str) -> pd.DataFrame:
    r = requests.get("https://v1.american-football.api-sports.io/games", headers={"x-apisports-key": key},
                     params={"league": 1, "season": SEASON}, timeout=60)
    r.raise_for_status()
    body = r.json()
    if body.get("errors"):
        sys.exit(f"STOP: API-Sports error: {body['errors']}")
    rows = []
    for x in body["response"]:
        gm, sc, v = x["game"], x["scores"], x["game"].get("venue") or {}
        rows.append({"game_id": gm["id"], "stage": gm["stage"], "api_week": gm["week"], "ts": gm["date"].get("timestamp"),
                     "status": gm["status"]["short"], "home_team_name": x["teams"]["home"]["name"],
                     "away_team_name": x["teams"]["away"]["name"], "home_total": sc["home"].get("total"),
                     "away_total": sc["away"].get("total"), "home_ot": sc["home"].get("overtime"),
                     "away_ot": sc["away"].get("overtime"), "venue_name": v.get("name"), "venue_city": v.get("city")})
    g = pd.DataFrame(rows)
    g = g[(g["stage"] == "Regular Season") & g["api_week"].astype(str).str.match(r"^Week \d+$") & g["ts"].notna()].copy()
    g["wk"] = g["api_week"].str.extract(r"(\d+)")[0].astype(int)
    g["kickoff_utc"] = pd.to_datetime(g["ts"], unit="s", utc=True)
    g["kickoff_et"] = g["kickoff_utc"].dt.tz_convert("America/New_York")
    return g


def fetch_dk_lines(key: str) -> dict:
    r = requests.get("https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds",
                     params={"apiKey": key, "regions": "us", "markets": "h2h,spreads,totals",
                             "bookmakers": "draftkings", "oddsFormat": "american"}, timeout=60)
    r.raise_for_status()
    print(f"Odds API credits used: {r.headers.get('x-requests-used')} | remaining: {r.headers.get('x-requests-remaining')}")
    lines = {}
    for ev in r.json():
        bk = next((b for b in ev.get("bookmakers", []) if b["key"] == "draftkings"), None)
        if not bk:
            continue
        mk = {m["key"]: m["outcomes"] for m in bk["markets"]}
        h, a = ev["home_team"], ev["away_team"]
        ml = {o["name"]: o["price"] for o in mk.get("h2h", [])}
        lines[frozenset([h, a])] = {
            "home": h,
            "spread": next((o["point"] for o in mk.get("spreads", []) if o["name"] == h), np.nan),
            "total": next((o["point"] for o in mk.get("totals", []) if o["name"] == "Over"), np.nan),
            "home_ml": ml.get(h), "away_ml": ml.get(a),
        }
    return lines


# ---------------------------------------------------------------- helpers

def line_is_consistent(spread, home_ml, away_ml) -> bool:
    """Reject a line whose moneyline disagrees with its spread about who is favored."""
    if pd.isna(spread) or home_ml is None or away_ml is None or spread == 0 or abs(spread) < 1.5:
        return True  # pick'em-ish lines can legitimately disagree
    return (spread < 0) == (home_ml < away_ml)


def weekly_path(wk: int) -> str:
    return f"{WEEKLY_DIR}/{SEASON}_W{wk}.csv"


def update_week(wk: int, games: pd.DataFrame, dk: dict, now: datetime) -> pd.DataFrame:
    """Refresh results for every game in the week; take DraftKings lines only for games not yet started."""
    path = weekly_path(wk)
    old = pd.read_csv(path).set_index("game_id") if os.path.exists(path) else pd.DataFrame()
    stamp = now.strftime("%Y-%m-%dT%H:%MZ")
    out, history = [], []
    for _, x in games[games["wk"] == wk].sort_values("kickoff_utc").iterrows():
        row = {c: x[c] for c in ["game_id", "wk", "home_team_name", "away_team_name", "status", "home_total",
                                 "away_total", "home_ot", "away_ot", "venue_name", "venue_city"]}
        row["kickoff_et"] = x["kickoff_et"].isoformat(sep=" ")
        prev = old.loc[x["game_id"]] if (len(old) and x["game_id"] in old.index) else None
        for c in ["closing_spread", "game_total", "home_moneyline", "away_moneyline", "bookmaker_name", "line_pulled_utc"]:
            row[c] = prev[c] if prev is not None and c in prev.index else np.nan
        started = x["kickoff_utc"] <= now or x["status"] != "NS"
        ln = dk.get(frozenset([x["home_team_name"], x["away_team_name"]]))
        if ln and not started:
            sp, hm, am = ln["spread"], ln["home_ml"], ln["away_ml"]
            if ln["home"] != x["home_team_name"]:
                sp, hm, am = -sp, am, hm
            if pd.isna(sp) or pd.isna(ln["total"]):
                print(f"  W{wk} {x['away_team_name']} @ {x['home_team_name']}: incomplete DraftKings line, kept previous")
            elif not line_is_consistent(sp, hm, am):
                print(f"  W{wk} {x['away_team_name']} @ {x['home_team_name']}: spread {sp} contradicts moneyline "
                      f"{hm}/{am}, kept previous")
            else:
                if prev is not None and pd.notna(prev.get("closing_spread")) and abs(sp - prev["closing_spread"]) >= 3:
                    print(f"  note: W{wk} {x['away_team_name']} @ {x['home_team_name']} moved "
                          f"{prev['closing_spread']} -> {sp}")
                row.update(closing_spread=sp, game_total=ln["total"], home_moneyline=hm, away_moneyline=am,
                           bookmaker_name="DraftKings", line_pulled_utc=stamp)
                history.append({"snapshot_utc": stamp, "game_id": x["game_id"], "wk": wk,
                                "kickoff_et": row["kickoff_et"], "home_team_name": x["home_team_name"],
                                "away_team_name": x["away_team_name"], "spread": sp, "total": ln["total"],
                                "home_moneyline": hm, "away_moneyline": am})
        out.append(row)
    week = pd.DataFrame(out)[WEEKLY_COLS]
    week.to_csv(path, index=False)
    if history:
        hpath = f"{HISTORY_DIR}/{SEASON}_W{wk}.csv"
        h = pd.DataFrame(history)
        h.to_csv(hpath, mode="a", header=not os.path.exists(hpath), index=False)
    final = int(week["status"].isin(FINAL).sum())
    missing = int(week["closing_spread"].isna().sum())
    print(f"W{wk}: {len(week)} games | final {final} | lines captured this run {len(history)} | without a line {missing}")
    return week


# ---------------------------------------------------------------- main

def run(games: pd.DataFrame, dk: dict, now: datetime) -> None:
    os.makedirs(WEEKLY_DIR, exist_ok=True)
    os.makedirs(HISTORY_DIR, exist_ok=True)
    master = pd.read_parquet(MASTER)
    merged_ids = set(master.loc[master["season"] == SEASON, "game_id"])

    pending = sorted(int(w) for w in games["wk"].unique()
                     if not set(games.loc[games["wk"] == w, "game_id"]) <= merged_ids)
    horizon = pd.Timestamp(now + timedelta(days=LOOKAHEAD_DAYS))
    active = [w for w in pending if games.loc[games["wk"] == w, "kickoff_utc"].min() <= horizon]
    print(f"Weeks not yet merged: {pending[:6]}{'...' if len(pending) > 6 else ''} | updating: {active}")

    weeks = {w: update_week(w, games, dk, now) for w in active}

    # Merge completed weeks in order; stop at the first incomplete one (records build week to week).
    for w in active:
        wk_df = weeks[w]
        if not wk_df["status"].isin(FINAL).all():
            print(f"W{w} not complete yet; nothing more to merge.")
            break
        if wk_df[["closing_spread", "game_total"]].isna().any().any():
            sys.exit(f"STOP: W{w} is complete but some games never got a line: "
                     f"{wk_df.loc[wk_df['closing_spread'].isna(), ['away_team_name', 'home_team_name']].values.tolist()}")
        print(f"Merging W{w} into the master...")
        merge_weeks.main(MASTER, MASTER, [weekly_path(w)])


def main():
    now = datetime.now(timezone.utc)
    # .strip(): secrets pasted with a trailing newline otherwise break the request (seen as %0A in the URL)
    games = fetch_games(os.environ["API_SPORTS_KEY"].strip())
    dk = fetch_dk_lines(os.environ["ODDS_API_KEY"].strip())
    print(f"Run at {now:%Y-%m-%d %H:%M} UTC | {SEASON} regular-season games: {len(games)} | DraftKings games listed: {len(dk)}")
    run(games, dk, now)


if __name__ == "__main__":
    main()
