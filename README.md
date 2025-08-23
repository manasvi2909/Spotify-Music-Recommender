# 🎵 Spotify Audio Features Recommender

An end-to-end **Music Recommender System** built with the [Spotify Audio Features dataset (Tomigelo)](https://www.kaggle.com/datasets/tomigelo/spotify-audio-features).  
The system uses **content-based filtering** with track audio features (danceability, energy, tempo, etc.) to recommend similar songs.

---

## 📌 Features
- Loads Spotify Audio Features dataset (CSV)
- Builds a **content-based recommender** using `NearestNeighbors`
- Evaluates recommendations with **Hit-Rate@K**
- Inspect recommendations for any chosen seed track
- Clean, modular Python implementation

---

## 📂 Project Structure
```

spotify\_recommender/
│── SpotifyAudioFeaturesApril2019.csv   # dataset
│── offline_recommender.py                      # recommender class
│── README.md                           # project documentation

````

---

## ⚙️ Setup & Installation

1. Clone this repo:
   ```bash
   git clone https://github.com/yourusername/spotify-recommender.git
   cd spotify-recommender
````

````markdown
## ⚙️ Setup

2. (Optional) Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate   # Mac/Linux
   .\venv\Scripts\activate    # Windows
````

3. Install dependencies:

   ```bash
   pip install pandas scikit-learn numpy
   ```

4. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/tomigelo/spotify-audio-features)
   Place `SpotifyAudioFeaturesApril2019.csv` inside the project folder.

---

## 🚀 Usage

### A) By track name/artist query (most convenient)

```bash
python offline_recommender.py --csv SpotifyAudioFeaturesApril2019.csv --seed-query "shape of you ed sheeran" --top 15
```

The script will:

* Show top 10 matches for your query (track name + artist).
* Auto-pick the first match as the seed.
* Print the Top 15 recommended tracks with similarity.

---

### B) By exact `track_id` (if you already know it)

```bash
python offline_recommender.py --csv SpotifyAudioFeaturesApril2019.csv --seed-id 7qiZfU4dY1lWllzX7mPBI3 --top 10
```

---

### C) Multiple seeds (centroid of several songs)

```bash
python offline_recommender.py --csv SpotifyAudioFeaturesApril2019.csv --multi-seed-ids 7qiZfU4dY1lWllzX7mPBI3,0tgVpDi06FyKpA1z0VMD4v --top 20
```

---

### D) Faster testing on a subset

```bash
python offline_recommender.py --csv SpotifyAudioFeaturesApril2019.csv --seed-query "arijit singh" --top 10 --subset 50000
```

---

### E) Save recommendations to a CSV

```bash
python offline_recommender.py --csv SpotifyAudioFeaturesApril2019.csv --seed-query "dil diyan gallan" --top 25 --out recs.csv
```

---

### 🔎 Example Output

```
=== Recommendations ===
 rank    track_name      artist_name     track_id                     popularity  similarity  distance
    1    Perfect         Ed Sheeran      0tgVpDi06FyKpA1z0VMD4v       85          0.952       0.048
    2    Happier         Ed Sheeran      2RttW7RAu5nOAfq6YFvApB       83          0.948       0.052
    3    Photograph      Ed Sheeran      1HNkqx9Ahdgi1Ixy2xkKkL       82          0.941       0.059
    ...
```

* **similarity** = `1 − cosine_distance` (closer to 1.0 = more similar).
* **popularity** (0–100) shown if available.

---

## 🧠 How it Works

1. Selects key **audio features** (`danceability`, `energy`, `tempo`, etc.).
2. Scales features with `StandardScaler`.
3. Uses `NearestNeighbors` (cosine similarity) to find similar tracks.
4. Evaluates recommendations using **Hit-Rate\@K**:

   * Pick an artist with ≥2 songs
   * Use one song as the "seed"
   * Hide another song as the "target"
   * Success if target appears in the top-K recommendations

---

## 📊 Evaluation Metric

* **Hit-Rate\@K**: Fraction of test cases where at least one hidden track from the same artist is retrieved in the top-K recommendations.
* Example: `Hit-Rate@10 = 0.42` → in 42% of cases, the system found a same-artist track in the top-10.

---

## 🔮 Future Improvements

* Add **Streamlit UI** for interactive recommendations
* Use **Spotify Web API** for live playlists
* Try **matrix factorization** or **deep learning embeddings**
* Hybrid approach: combine **content-based** + **collaborative filtering**

```

