import argparse
import sys
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "danceability", "energy", "acousticness",
    "instrumentalness", "liveness", "speechiness",
    "valence", "tempo", "loudness"
]

REQ_COLS = ["track_id", "track_name", "artist_name"] + FEATURES
OPTIONAL_COLS = ["popularity"]  


def load_dataset(csv_path: str, subset: Optional[int] = None) -> pd.DataFrame:
    df = pd.read_csv("data/SpotifyAudioFeaturesApril2019.csv")
    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    keep_cols = [c for c in REQ_COLS + OPTIONAL_COLS if c in df.columns]
    df = df[keep_cols].dropna(subset=FEATURES).drop_duplicates(subset=["track_id"])

    if subset is not None and subset < len(df):
        df = df.sample(n=subset, random_state=42).reset_index(drop=True)

    return df.reset_index(drop=True)


class ContentRecommender:
    def __init__(self, k: int = 100, metric: str = "cosine"):
        self.k = k
        self.metric = metric
        self.scaler: Optional[StandardScaler] = None
        self.nn: Optional[NearestNeighbors] = None
        self.df: Optional[pd.DataFrame] = None
        self.X: Optional[np.ndarray] = None

    def fit(self, df: pd.DataFrame) -> None:
        self.df = df.reset_index(drop=True)
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.df[FEATURES].values)
        n_neighbors = min(self.k, len(self.df))
        self.nn = NearestNeighbors(n_neighbors=n_neighbors, metric=self.metric)
        self.nn.fit(self.X)

    def _idx_by_track_id(self, tid: str) -> int:
        hits = self.df.index[self.df["track_id"] == tid].tolist()
        if not hits:
            raise KeyError(f"track_id not found: {tid}")
        return hits[0]

    def recommend_by_id(self, track_id: str, n: int = 10, include_seed: bool = False) -> pd.DataFrame:
        idx = self._idx_by_track_id(track_id)
        x = self.X[idx].reshape(1, -1)
        distances, indices = self.nn.kneighbors(x, n_neighbors=min(self.k, len(self.df)))
        distances, indices = distances[0], indices[0]

        rows = []
        for i, d in zip(indices, distances):
            if (not include_seed) and (i == idx):
                continue
            row = self.df.iloc[i].to_dict()
            similarity = 1.0 - float(d)
            row.update({"rank": len(rows) + 1, "distance": float(d), "similarity": similarity})
            rows.append(row)
            if len(rows) >= n:
                break
        cols = ["rank", "track_name", "artist_name", "track_id", "similarity", "distance"]
        if "popularity" in self.df.columns:
            cols.insert(4, "popularity")
        return pd.DataFrame(rows)[cols]

    def recommend_from_multiple(self, track_ids: List[str], n: int = 10) -> pd.DataFrame:
        idxs = [self._idx_by_track_id(tid) for tid in track_ids]
        centroid = np.mean(self.X[idxs], axis=0).reshape(1, -1)
        distances, indices = self.nn.kneighbors(centroid, n_neighbors=min(self.k, len(self.df)))
        distances, indices = distances[0], indices[0]
        seed_set = set(idxs)

        rows = []
        for i, d in zip(indices, distances):
            if i in seed_set:
                continue
            row = self.df.iloc[i].to_dict()
            similarity = 1.0 - float(d)
            row.update({"rank": len(rows) + 1, "distance": float(d), "similarity": similarity})
            rows.append(row)
            if len(rows) >= n:
                break
        cols = ["rank", "track_name", "artist_name", "track_id", "similarity", "distance"]
        if "popularity" in self.df.columns:
            cols.insert(4, "popularity")
        return pd.DataFrame(rows)[cols]


def search_tracks(df: pd.DataFrame, query: str, limit: int = 10) -> pd.DataFrame:
    q = query.strip().lower()
    hit_mask = df["track_name"].str.lower().str.contains(q) | df["artist_name"].str.lower().str.contains(q)
    hits = df[hit_mask].head(limit).copy()
    return hits[["track_name", "artist_name", "track_id"]]


def main():
    p = argparse.ArgumentParser(description="Content-based music recommender (tomigelo dataset)")
    p.add_argument("--csv", required=True, help="Path to SpotifyAudioFeaturesApril2019.csv")
    p.add_argument("--seed-id", help="Seed Spotify track_id")
    p.add_argument("--seed-query", help="Search string to find a seed (e.g., 'shape of you ed sheeran')")
    p.add_argument("--multi-seed-ids", help="Comma-separated track_ids for multi-seed centroid")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--subset", type=int, help="Optional: use only N rows (faster testing)")
    p.add_argument("--out", help="Optional: save recommendations to CSV")
    args = p.parse_args()
    df = load_dataset(args.csv, subset=args.subset)
    if args.seed_query and not args.seed_id and not args.multi_seed_ids:
        matches = search_tracks(df, args.seed_query, limit=10)
        if matches.empty:
            print("No matches found for your query.", file=sys.stderr)
            sys.exit(2)
        print("\nTop matches for your query:")
        print(matches.to_string(index=False))
        # Auto-pick the first match for convenience
        seed_id = matches.iloc[0]["track_id"]
        print(f"\nUsing first match as seed: {seed_id}")
        chosen_ids = [seed_id]
    elif args.seed_id:
        chosen_ids = [args.seed_id]
    elif args.multi_seed_ids:
        chosen_ids = [tid.strip() for tid in args.multi_seed_ids.split(",") if tid.strip()]
    else:
        seed_row = df.sample(1, random_state=7).iloc[0]
        chosen_ids = [seed_row["track_id"]]
        print(f"No seed supplied; using random seed: {seed_row['track_name']} — {seed_row['artist_name']}")
    if args.subset is not None:
        need_rows = df[df["track_id"].isin(chosen_ids)]
        if need_rows.empty:
            full_df = load_dataset(args.csv, subset=None)
            seed_rows = full_df[full_df["track_id"].isin(chosen_ids)]
            if not seed_rows.empty:
                df = pd.concat([df, seed_rows], axis=0).drop_duplicates(subset=["track_id"]).reset_index(drop=True)

    rec = ContentRecommender(k=200, metric="cosine")
    rec.fit(df)

    if len(chosen_ids) == 1:
        out = rec.recommend_by_id(chosen_ids[0], n=args.top)
    else:
        out = rec.recommend_from_multiple(chosen_ids, n=args.top)

    print("\n=== Recommendations ===")
    print(out.to_string(index=False))

    if args.out:
        out.to_csv(args.out, index=False)
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
