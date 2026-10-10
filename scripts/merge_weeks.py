"""Merge completed weekly game files into the NFL master parquet.

Input: weekly CSVs in the layout produced by the weekly pull (one row per game):
  game_id, wk, kickoff_et, home_team_name, away_team_name, status, home_total, away_total,
  home_ot, away_ot, venue_name, venue_city, closing_spread (home perspective), game_total,
  home_moneyline, away_moneyline, bookmaker_name

Usage:
  python scripts/merge_weeks.py data/nfl_master_2022_2025.parquet OUT.parquet W3.csv W4.csv [...]

Rules (see project notes):
  - Only complete weeks are merged (every game status FT/AOT); the script refuses otherwise.
  - Per-game results come from scores + lines; running and entering records are rebuilt for the
    whole season with the formulas validated against 2022-2025.
  - All derived columns for the season are recomputed and the already-merged weeks must come out
    identical to what is stored, or the script stops.
"""
import sys

import numpy as np
import pandas as pd

SEASON = 2026
FINAL = {"FT", "AOT"}
CITY_FIX = {"Foxboro": "Foxborough", "Rio De Janeiro": "Rio de Janeiro"}


def flag(x):
    if pd.isna(x):
        return None
    return True if x > 0 else (False if x < 0 else None)


def rec(w, l, t=None):
    if t is None:
        return [f"W{a}L{b}" for a, b in zip(w, l)]
    return [f"W{a}L{b}" + (f"T{c}" if c else "") for a, b, c in zip(w, l, t)]


def recp(w, l, p):
    return [f"W{a}L{b}P{c}" for a, b, c in zip(w, l, p)]


def rebuild_records(df):
    s = df.sort_values(["season", "team", "game_date"])
    cur = pd.DataFrame(index=s.index)
    tw, sc, th = s["team_won"], s["spread_covered"], s["total_hit_over"]
    cur["w"] = (tw == True).astype(int)
    cur["l"] = (tw == False).astype(int)
    cur["t"] = (tw.isna() & s["team_score"].notna() & (s["team_score"] == s["opponent_score"])).astype(int)
    cur["aw"] = (sc == True).astype(int)
    cur["al"] = (sc == False).astype(int)
    cur["ap"] = (sc.isna() & (s["spread_margin"].abs() < 0.001)).astype(int)
    cur["ow"] = (th == True).astype(int)
    cur["ou"] = (th == False).astype(int)
    cur["op"] = (th.isna() & (s["total_margin"].abs() < 0.001)).astype(int)
    cum = cur.groupby([s["season"], s["team"]]).cumsum()
    ent = cum - cur
    o = pd.DataFrame(index=s.index)
    o["season_wins"], o["season_losses"], o["season_ties"] = cum["w"], cum["l"], cum["t"]
    o["season_record"] = rec(cum["w"], cum["l"], cum["t"])
    o["ats_wins"], o["ats_losses"], o["ats_pushes"] = cum["aw"], cum["al"], cum["ap"]
    o["ats_record"] = recp(cum["aw"], cum["al"], cum["ap"])
    o["over_wins"], o["under_wins"], o["total_pushes"] = cum["ow"], cum["ou"], cum["op"]
    o["total_record"] = recp(cum["ow"], cum["ou"], cum["op"])
    o["entering_season_wins"], o["entering_season_losses"], o["entering_season_ties"] = ent["w"], ent["l"], ent["t"]
    o["entering_season_record"] = rec(ent["w"], ent["l"], ent["t"])
    o["entering_ats_wins"], o["entering_ats_losses"], o["entering_ats_pushes"] = ent["aw"], ent["al"], ent["ap"]
    o["entering_ats_record"] = recp(ent["aw"], ent["al"], ent["ap"])
    o["entering_over_wins"], o["entering_under_wins"], o["entering_total_pushes"] = ent["ow"], ent["ou"], ent["op"]
    o["entering_total_record"] = recp(ent["ow"], ent["ou"], ent["op"])
    gp = ent["w"] + ent["l"] + ent["t"]
    o["entering_win_pct"] = (ent["w"] / gp.where(gp > 0)).fillna(0.0)
    ad = ent["aw"] + ent["al"]
    o["entering_ats_win_pct"] = (ent["aw"] / ad.where(ad > 0)).fillna(0.0)
    return o.reindex(df.index)


def schedule_columns(df):
    """Rest, previous-game and road-trip columns, computed within each season."""
    s = df.sort_values(["season", "team", "game_date"])
    g = s.groupby(["season", "team"])
    o = pd.DataFrame(index=s.index)
    rd = (s["game_date"] - g["game_date"].shift(1)).dt.days.astype(float)
    o["rest_days"] = rd
    o["rest_status"] = np.select([rd.isna(), rd < 7, rd >= 10], ["season_opener", "short_rest", "extended_rest"], "normal_rest")
    o["is_short_rest"] = (rd < 7).fillna(False).astype(bool)
    o["is_extended_rest"] = (rd >= 10).fillna(False).astype(bool)
    o["is_off_bye"] = (rd >= 13).fillna(False).astype(bool)
    pm = g["point_margin"].shift(1)
    o["prev_point_margin"] = pm
    o["prev_result"] = pd.Series(np.select([pm <= -17, pm < 0, pm == 0, pm < 17], ["blowout_loss", "loss", "tie", "win"],
                                           "blowout_win"), index=s.index).where(pm.notna())
    o["prev_spread_covered"] = g["spread_covered"].shift(1)
    ps, pw = g["team_spread"].shift(1), g["team_won"].shift(1)
    o["prev_was_favorite"] = pd.Series(np.where(ps < 0, True, False), index=s.index).where(ps.notna())
    o["prev_upset"] = pd.Series(np.select([(ps > 0) & (pw == True), (ps < 0) & (pw == False)], ["upset_win", "upset_loss"],
                                          "none"), index=s.index).where(ps.notna())
    o["prev_game_was_overtime"] = g["is_overtime_game"].shift(1)
    run = g["home_away"].transform(lambda x: (x != x.shift()).cumsum())
    pos = s.groupby([s["season"], s["team"], run]).cumcount() + 1
    o["road_trip_game"] = np.where(s["home_away"] == "Away", pos, 0)
    prev_rt = o.groupby([s["season"], s["team"]])["road_trip_game"].shift(1).fillna(0)
    o["home_after_road_trip"] = np.where(s["home_away"] == "Home", prev_rt, 0).astype(int)
    return o.reindex(df.index)


def same(a, b):
    an, bn = a.isna(), b.isna()
    if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b) and not pd.api.types.is_bool_dtype(a):
        eq = np.isclose(a.astype(float).fillna(0), b.astype(float).fillna(0))
    else:
        eq = a.astype(str).values == b.astype(str).values
    return (an & bn) | (~an & ~bn & eq)


def build_rows(wk_df, master):
    hm = master[(master["home_away"] == "Home") & (master["is_neutral_site"] == False)]
    home_city = hm.groupby("team")["venue_city"].agg(lambda s: s.mode().iat[0])
    home_venue = hm.sort_values("game_date").groupby("team")[["venue_name", "venue_city"]].last()
    div = master[master["season"] == master["season"].max()].drop_duplicates("team").set_index("team")["team_division"]

    rows = []
    for _, g in wk_df.iterrows():
        k = pd.Timestamp(g["kickoff_et"])
        day = k.day_name()
        hr = k.hour + k.minute / 60
        night = hr >= 19
        slot = ("TNF" if night and day == "Thursday" else "SNF" if night and day == "Sunday" else
                "MNF" if night and day == "Monday" else f"{day} night" if night else
                "Sun morning" if day == "Sunday" and hr < 12 else "Sun early" if day == "Sunday" and hr < 16 else
                "Sun late" if day == "Sunday" else f"{day} day")
        city = CITY_FIX.get(g["venue_city"], g["venue_city"])
        home = g["home_team_name"]
        neutral = city != home_city.get(home)
        # keep the master's stadium name when the game is at the team's usual home city
        vname = home_venue.loc[home, "venue_name"] if (not neutral and home in home_venue.index) else g["venue_name"]
        h_ot, a_ot = g["home_ot"], g["away_ot"]
        is_ot = pd.notna(h_ot) or pd.notna(a_ot)
        for is_home in (True, False):
            team, opp = (g["home_team_name"], g["away_team_name"]) if is_home else (g["away_team_name"], g["home_team_name"])
            ts, os_ = (g["home_total"], g["away_total"]) if is_home else (g["away_total"], g["home_total"])
            sp = g["closing_spread"] if is_home else -g["closing_spread"]
            tml, oml = (g["home_moneyline"], g["away_moneyline"]) if is_home else (g["away_moneyline"], g["home_moneyline"])
            tot = g["game_total"]
            rows.append({
                "game_id": int(g["game_id"]), "season": SEASON, "week": f"W{int(g['wk'])}",
                "game_date": pd.Timestamp(k.date()), "day_of_week": day,
                "team": team, "opponent": opp, "home_away": "Home" if is_home else "Away",
                "team_score": float(ts), "opponent_score": float(os_),
                "team_spread": float(sp), "opponent_spread": float(-sp), "game_total": float(tot),
                "team_moneyline": float(tml) if pd.notna(tml) else np.nan,
                "opponent_moneyline": float(oml) if pd.notna(oml) else np.nan,
                "bookmaker": g["bookmaker_name"], "game_completed": True,
                "point_margin": ts - os_, "spread_margin": ts - os_ + sp, "total_margin": ts + os_ - tot,
                "team_won": flag(ts - os_), "spread_covered": flag(ts - os_ + sp), "total_hit_over": flag(ts + os_ - tot),
                "is_primetime": bool(night), "is_sunday_game": day == "Sunday",
                "team_division": div[team], "opponent_division": div[opp], "is_divisional_game": bool(div[team] == div[opp]),
                "venue_name": vname, "venue_city": city, "kickoff_slot": slot,
                "game_time_utc": k.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
                "is_overtime_game": bool(is_ot),
                "team_ot_points": float(h_ot if is_home else a_ot) if is_ot else np.nan,
                "opponent_ot_points": float(a_ot if is_home else h_ot) if is_ot else np.nan,
                "is_neutral_site": bool(neutral),
            })
    return pd.DataFrame(rows)


def main(master_path, out_path, week_files):
    master = pd.read_parquet(master_path)
    master["game_date"] = pd.to_datetime(master["game_date"])
    weeks = pd.concat([pd.read_csv(f) for f in week_files], ignore_index=True)

    bad = weeks[~weeks["status"].isin(FINAL)]
    if len(bad):
        sys.exit(f"STOP: {len(bad)} games not final: {bad[['wk', 'away_team_name', 'home_team_name', 'status']].values.tolist()}")
    if weeks[["closing_spread", "game_total"]].isna().any().any():
        sys.exit("STOP: missing spread or total")
    already = set(master.loc[master["season"] == SEASON, "game_id"])
    dup = weeks[weeks["game_id"].isin(already)]
    if len(dup):
        sys.exit(f"STOP: {len(dup)} games already in master")

    new = build_rows(weeks, master).reindex(columns=master.columns)
    df = pd.concat([master, new], ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"])
    s26 = df["season"] == SEASON
    old26 = s26 & df["game_id"].isin(already)

    # Recompute season records and schedule columns; already-merged weeks must not change.
    derived = pd.concat([rebuild_records(df[s26]), schedule_columns(df[s26])], axis=1)
    problems = {c: int((~same(df.loc[old26, c], derived.loc[old26, c])).sum()) for c in derived.columns}
    problems = {c: n for c, n in problems.items() if n}
    if problems:
        sys.exit(f"STOP: recomputing existing {SEASON} weeks changed stored values: {problems}")
    new_rows = s26 & ~df["game_id"].isin(already)
    for c in derived.columns:
        df.loc[new_rows, c] = derived.loc[new_rows, c]

    for c in master.columns:
        if df[c].dtype != master[c].dtype:
            try:
                df[c] = df[c].astype(master[c].dtype)
            except (TypeError, ValueError) as ex:
                print(f"  note: kept {c} as {df[c].dtype} ({ex})")

    # Checks
    hist_same = df[df["season"] != SEASON].reset_index(drop=True).equals(master[master["season"] != SEASON].reset_index(drop=True))
    per_game = df[s26].groupby("game_id").size()
    n = df[new_rows]
    mirror = n.merge(n, left_on=["game_id", "team"], right_on=["game_id", "opponent"], suffixes=("", "_o"))
    mirror_bad = int((~np.isclose(mirror["team_spread"], -mirror["team_spread_o"])).sum())
    swap = n[(n["team_moneyline"].notna()) & (n["team_spread"] != 0) &
             ((n["team_spread"] < 0) != (n["team_moneyline"] < n["opponent_moneyline"]))]
    end = df[s26].sort_values("game_date").groupby("team").tail(1)
    print(f"Rows: {len(master)} -> {len(df)} (+{len(new)})")
    print("History untouched:", hist_same)
    print("2026 rows per week:", df[s26].groupby("week").size().to_dict())
    print("Games with != 2 rows:", int((per_game != 2).sum()), "| mirror spread mismatches:", mirror_bad,
          "| spread/moneyline favorite swaps:", len(swap))
    print(f"2026 ATS W-L-P: {int(end['ats_wins'].sum())}-{int(end['ats_losses'].sum())}-{int(end['ats_pushes'].sum())} | "
          f"O/U: {int(end['over_wins'].sum())}-{int(end['under_wins'].sum())}-{int(end['total_pushes'].sum())}")
    print("New neutral-site games:", n[n["is_neutral_site"] & (n["home_away"] == "Home")][["week", "team", "opponent", "venue_city"]].values.tolist())
    print("New OT games:", int(n["is_overtime_game"].sum() // 2))
    df.to_parquet(out_path, index=False)
    print("Saved; reload identical:", pd.read_parquet(out_path).equals(df))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
