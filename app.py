# app.py
import streamlit as st
import pandas as pd

from activity_model import ActivityModel
from emotion_module import EmotionScorer
from scene_module import SceneDetector
from recommender import UnifiedRecommender


st.set_page_config(page_title="Smart Travel Recommender", page_icon="🧭", layout="wide")


@st.cache_data
def load_data(path: str):
    df = pd.read_csv(
    path,
    engine="python",  
    sep=",",
    quotechar='"',
    escapechar="\\",
    on_bad_lines="skip"   
)

    # normalize column names just in case
    df.columns = [c.strip() for c in df.columns]
    return df


@st.cache_resource
def build_system(df: pd.DataFrame):
    activity_model = ActivityModel()
    emotion = EmotionScorer()
    scene = SceneDetector()  # will be lazy-loaded only when used
    rec = UnifiedRecommender(activity_model=activity_model, emotion_scorer=emotion, scene_detector=scene)
    rec.fit(df)
    return rec


st.title("🧭 Smart Travel Preference Recommender")
st.caption("Predict → Enrich → Recommend (Activity is evaluated; Emotion/Scene are auxiliary to enhance recommendations)")

# Sidebar controls
st.sidebar.header("⚙️ Settings")
data_path = st.sidebar.text_input("Dataset path", value="cleaned_dataset.csv")
top_n = st.sidebar.slider("Number of recommendations", 3, 12, 8)
use_scene = st.sidebar.toggle("Enable Scene Detection (slower)", value=False)

try:
    df = load_data(data_path)
except Exception as e:
    st.error(f"Could not load dataset. Make sure '{data_path}' exists. Error: {e}")
    st.stop()

rec = build_system(df)

# UI inputs
colA, colB = st.columns([2, 1], gap="large")

with colA:
    st.subheader("📝 Tell us what you like")
    user_text = st.text_area(
        "Describe your trip preferences (what you like, what you want to do, vibe, etc.)",
        height=120,
        placeholder="Example: I want a calm place near nature with beautiful views, walking, and relaxing..."
    )

with colB:
    st.subheader("🌍 Optional filters")
    countries = sorted([c for c in df["country"].dropna().astype(str).unique().tolist() if c.strip()])
    country_choice = st.selectbox("Preferred country", ["Any"] + countries)

    st.markdown("---")
    run_btn = st.button("✨ Recommend", use_container_width=True)

if run_btn:
    out = rec.recommend(
        user_text=user_text,
        preferred_country=country_choice,
        top_n=top_n,
        use_scene=use_scene
    )

    st.markdown("## ✅ Your Travel Profile")
    p1, p2 = st.columns(2)
    with p1:
        st.metric("Predicted Activity", out["activity"])
    with p2:
        st.metric("Detected Mood (aux)", out["emotion"])

    st.markdown("## 📍 Recommendations")
    results = out["results"]
    if not results:
        st.warning("No results found. Try writing a bit more detail in your preference text.")
    else:
        # Show as cards
        for r in results:
            with st.container(border=True):
                left, right = st.columns([1, 2], gap="large")
                with left:
                    if r["image_url"]:
                        st.image(r["image_url"], use_container_width=True)
                    else:
                        st.info("No image available.")
                with right:
                    st.subheader(f"{r['country']} — {r['activity']}")
                    st.caption(f"Reason: {r['reason']}")
                    st.write(r["description"])
                    meta = []
                    if r["mood"]:
                        meta.append(f"mood: {r['mood']}")
                    if r["scene_tags"]:
                        meta.append(f"scene: {', '.join(r['scene_tags'])}")
                    if meta:
                        st.markdown("**Tags:** " + " | ".join(meta))
