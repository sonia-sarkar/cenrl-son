"""
Feature analysis script.
Computes block rates per category, TLD, rank bin, and proposed feature groups
across all three blocklists (GFW/China, Russia, Kazakhstan).

Run from the repo root:
    python analyze_features.py
"""

import pandas as pd
from ast import literal_eval

TRANCO = "inputs/tranco/tranco_categories_subdomain_tld_entities_top10k.csv"
BLOCKLISTS = {
    "china":      "inputs/gfwatch/gfwatch-blocklist.csv",
    "russia":     "inputs/russia/russia-blocklist.csv",
    "kazakhstan": "inputs/kazakhstan/kazakhstan-blocklist.csv",
}

# --- Feature groupings (your prior knowledge encoded here) ---
CIRCUMVENTION = {"P2P", "Anonymizer", "File Sharing", "Redirect", "Hacking"}
ADULT         = {"Pornography", "Adult Themes", "Dating & Relationships", "Nudity",
                 "Lingerie & Bikini", "Sex Education"}
NEWS_MEDIA    = {"News & Media", "Magazines", "Politics, Advocacy, and Government-Related",
                 "Forums", "Personal Blogs", "News, Portal & Search",
                 "News & Media, Video Streaming"}
SOCIAL        = {"Social Networks", "Instant Messengers", "Chat",
                 "Professional Networking", "Messaging"}
SEARCH        = {"Search Engines", "News, Portal & Search"}
STREAMING     = {"Video Streaming", "Audio Streaming", "Television", "Music",
                 "Radio", "Home Video/DVD"}
INFRASTRUCTURE = {"Content Servers", "Advertisements", "APIs", "Login Screens",
                  "Webmail", "Redirect"}
MAJOR_US_TECH = {"Google LLC", "Meta Platforms, Inc.", "Twitter, Inc.",
                 "Amazon.com, Inc.", "Microsoft Corporation", "Apple Inc.",
                 "Alphabet Inc."}


def load_tranco():
    df = pd.read_csv(TRANCO, delimiter="|")
    df["categories"] = df["categories"].apply(literal_eval)
    return df


def load_blocklist(path):
    """Load a blocklist, stripping wildcard prefixes."""
    bl = pd.read_csv(path, header=0, names=["domain"], on_bad_lines="skip")
    bl["domain"] = bl["domain"].str.lstrip("*.")
    return set(bl["domain"].dropna().unique())


def merge_blocklist(tranco_df, blocked_set, col_name):
    tranco_df[col_name] = tranco_df["domain"].isin(blocked_set).astype(int)
    return tranco_df


def block_rate(df, group_col, blocked_col, min_count=5):
    result = (
        df.groupby(group_col)[blocked_col]
        .agg(blocked="sum", total="count")
        .assign(block_rate=lambda x: x["blocked"] / x["total"])
        .query("total >= @min_count")
        .sort_values("block_rate", ascending=False)
    )
    return result


def assign_feature_flags(df):
    exploded = df.explode("categories")

    def any_in(series, group):
        return series.isin(group).groupby(level=0).any()

    df["is_circumvention"] = any_in(exploded["categories"], CIRCUMVENTION)
    df["is_adult"]         = any_in(exploded["categories"], ADULT)
    df["is_news"]          = any_in(exploded["categories"], NEWS_MEDIA)
    df["is_social"]        = any_in(exploded["categories"], SOCIAL)
    df["is_search"]        = any_in(exploded["categories"], SEARCH)
    df["is_streaming"]     = any_in(exploded["categories"], STREAMING)
    df["is_infra"]         = any_in(exploded["categories"], INFRASTRUCTURE)
    df["is_major_us_tech"] = df["entity"].isin(MAJOR_US_TECH)
    df["is_top_200"]       = df["rank"] <= 200
    df["is_mid_tier"]      = df["rank"].between(5000, 7000)

    return df


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def main():
    print("Loading data...")
    df = load_tranco()

    blocklists = {}
    for name, path in BLOCKLISTS.items():
        try:
            blocklists[name] = load_blocklist(path)
            print(f"  {name}: {len(blocklists[name]):,} blocked domains")
        except Exception as e:
            print(f"  {name}: failed to load ({e})")

    for name, blocked_set in blocklists.items():
        df = merge_blocklist(df, blocked_set, f"blocked_{name}")

    df = assign_feature_flags(df)

    # ---- Per-country analysis ----
    for country, col in [(n, f"blocked_{n}") for n in blocklists]:
        total_blocked = df[col].sum()
        print_section(f"{country.upper()}  —  {total_blocked} blocked in top 10k")

        # Block rate by category
        exploded = df.explode("categories").copy()
        print("\nBy category (top 20, min 5 domains):")
        print(block_rate(exploded, "categories", col, min_count=5).head(20).to_string())

        # Block rate by proposed feature flags
        print("\nBy feature flag:")
        flags = ["is_circumvention", "is_adult", "is_news", "is_social",
                 "is_search", "is_streaming", "is_infra", "is_major_us_tech",
                 "is_top_200", "is_mid_tier"]
        rows = []
        for flag in flags:
            grp = df.groupby(flag)[col].agg(blocked="sum", total="count")
            grp["block_rate"] = grp["blocked"] / grp["total"]
            if True in grp.index and False in grp.index:
                rows.append({
                    "feature": flag,
                    "blocked (flag=1)": int(grp.loc[True, "blocked"]),
                    "total (flag=1)":   int(grp.loc[True, "total"]),
                    "block_rate (flag=1)": round(grp.loc[True, "block_rate"], 3),
                    "block_rate (flag=0)": round(grp.loc[False, "block_rate"], 3),
                    "lift": round(grp.loc[True, "block_rate"] / max(grp.loc[False, "block_rate"], 0.001), 1),
                })
        flag_df = pd.DataFrame(rows).sort_values("lift", ascending=False)
        print(flag_df.to_string(index=False))

        # Block rate by TLD
        print("\nBy TLD (top 15, min 5 domains):")
        print(block_rate(df, "tld", col, min_count=5).head(15).to_string())

        # Block rate by rank bin
        print("\nBy rank bin (top 10):")
        print(block_rate(df, "bin", col, min_count=1).head(10).to_string())


if __name__ == "__main__":
    main()
