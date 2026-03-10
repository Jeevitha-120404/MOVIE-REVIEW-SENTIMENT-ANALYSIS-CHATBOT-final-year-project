import streamlit as st
import pandas as pd
import joblib
import matplotlib
import matplotlib.pyplot as plt
import difflib
import re
import nltk
from collections import Counter

matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# Optional imports
try:
    from wordcloud import WordCloud
    WORDCLOUD_OK = True
except ImportError:
    WORDCLOUD_OK = False

try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    nltk.download("vader_lexicon", quiet=True)
    vader = SentimentIntensityAnalyzer()
    VADER_OK = True
except Exception:
    VADER_OK = False
#
st.set_page_config(
    page_title="🎬 CineBot — Sentiment Chatbot",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ── Load CSS from style.css ───────────────────────────────────
import os
css_path = os.path.join(os.path.dirname(__file__), "style.css")
with open(css_path, encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  HTML COMPONENT FUNCTIONS  (reads structure from components.html)
def verdict_card(emoji, title, sub, total):
    return f"""<div class="verdict-card">
        <span class="verdict-emoji">{emoji}</span>
        <span class="verdict-title">{title}</span><br>
        <span class="verdict-sub">{sub} &bull; {total} reviews analysed</span>
    </div>"""

def stat_row(pos, neg, neu):
    return f"""<div class="stat-row">
        <div class="stat-badge stat-pos">😄 {pos:.1f}%<span class="stat-label">Positive</span></div>
        <div class="stat-badge stat-neg">😡 {neg:.1f}%<span class="stat-label">Negative</span></div>
        <div class="stat-badge stat-neu">😐 {neu:.1f}%<span class="stat-label">Neutral</span></div>
    </div>"""

def meta_row(actor, director, genre, year):
    return f"""<div class="meta-row">
        <span class="meta-tag">🎭 <span>Actor:</span> <strong>{actor}</strong></span>
        <span class="meta-tag">🎬 <span>Director:</span> <strong>{director}</strong></span>
        <span class="meta-tag">🎭 <span>Genre:</span> <strong>{genre}</strong></span>
        <span class="meta-tag">📅 <span>Year:</span> <strong>{year}</strong></span>
    </div>"""

def review_pill(text, cls, conf=None):
    short = text[:130] + "…" if len(text) > 130 else text
    conf_str = f" <strong>({conf*100:.0f}%)</strong>" if conf else ""
    return f'<div class="review-pill {cls}">{short}{conf_str}</div>'

def highlight_card(text, cls, label, conf=None):
    conf_str = f'<span class="highlight-conf">Confidence: {conf*100:.0f}%</span>' if conf else ""
    return f"""<div class="highlight-card {cls}">
        <span class="highlight-label">{label}</span>
        <div class="highlight-text">"{text}"</div>
        {conf_str}
    </div>"""

def conf_bar(label, pct, color):
    return f"""<div class="conf-row">
        <div class="conf-label"><span>{label}</span><span>{pct:.1f}%</span></div>
        <div class="conf-bar-wrap">
            <div class="conf-bar" style="width:{pct}%;background:{color};box-shadow:0 0 10px {color}88;"></div>
        </div>
    </div>"""

def winner_banner(winner, diff):
    return f"""<div class="winner-banner">
        <span class="winner-trophy">🏆</span>
        <div class="winner-title">{winner.title()} &nbsp;WINS!</div>
        <div class="winner-sub">{diff:.1f}% more positive reviews &nbsp;🎉</div>
    </div>"""

def movie_list_card(emoji, name, meta, score):
    return f"""<div class="movie-list-card">
        <span class="movie-list-icon">{emoji}</span>
        <span class="movie-list-name">{name}</span>
        <span class="movie-list-meta">{meta}</span>
        <span class="movie-list-score">{score}</span>
    </div>"""

def sec_div(label):
    return f"""<div class="section-div">
        <div class="section-div-line"></div>{label}<div class="section-div-line"></div>
    </div>"""
# ══════════════════════════════════════════════════════════════
#  LOAD MODEL + DATA
@st.cache_resource
def load_model():
    return joblib.load("sentiment_model.pkl")

@st.cache_data
def load_data():
    df = pd.read_csv("movie_reviews_dataset.csv", encoding="utf-8")
    df.columns = df.columns.str.strip().str.lower()
    df.rename(columns={"moviename": "movie"}, inplace=True, errors="ignore")
    df["movie"] = df["movie"].str.strip().str.lower()
    for col in ["actor","director","genre","year"]:
        if col not in df.columns:
            df[col] = "Unknown"
    df["actor"]    = df["actor"].astype(str).str.strip()
    df["director"] = df["director"].astype(str).str.strip()
    df["genre"]    = df["genre"].astype(str).str.strip()
    df["year"]     = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)
    return df

model, vectorizer = load_model()
df = load_data()
ALL_MOVIES = sorted(df["movie"].unique().tolist())
ALL_ACTORS = sorted(df["actor"].dropna().unique().tolist())
ALL_DIRS   = sorted(df["director"].dropna().unique().tolist())
ALL_GENRES = sorted(df["genre"].dropna().unique().tolist())
ALL_YEARS  = sorted([y for y in df["year"].unique().tolist() if y > 0])
# ══════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
def get_verdict(pos, neg, neu):
    if pos >= 70: return "🏆","Audience Loved It!","Overwhelmingly positive"
    if neg >= 70: return "💀","Audience Hated It!","Overwhelmingly negative"
    if pos >= 50: return "😊","Mostly Positive!","Good overall reception"
    if neg >= 50: return "😤","Mostly Negative!","Poor overall reception"
    return "🤔","Divisive Film!","Mixed audience reactions"

def donut(pos, neg, neu, title):
    fig, ax = plt.subplots(figsize=(2.8, 2.8), facecolor="none")
    colors = ["#34d399","#f87171","#fb923c"]
    vals   = [pos, neg, neu]
    labels = [f"Pos {pos:.0f}%", f"Neg {neg:.0f}%", f"Neu {neu:.0f}%"]
    wedges, _, autos = ax.pie(vals, labels=None, autopct="%1.0f%%",
        startangle=90, colors=colors,
        wedgeprops={"width":0.42, "edgecolor":"none"},
        textprops={"color":"#ffffff", "fontsize":9, "fontweight":"bold"})
    # Force percentage text to be fully visible black on colored wedge
    for auto in autos:
        auto.set_color("#111111")
        auto.set_fontsize(9)
        auto.set_fontweight("bold")
    legend = ax.legend(wedges, labels, loc="lower center",
              bbox_to_anchor=(0.5,-0.25), ncol=3,
              frameon=True, fontsize=8, labelcolor="#111111")
    legend.get_frame().set_facecolor("none")
    legend.get_frame().set_edgecolor("none")
    for text in legend.get_texts():
        text.set_color("#111111")
        text.set_fontweight("bold")
    ax.set_title(title, color="white", fontsize=9, pad=8)
    fig.patch.set_alpha(0)
    plt.tight_layout(pad=0.4)
    return fig

def genre_bar(genre_data):
    """
    FIX: matplotlib does NOT support CSS rgba() strings.
    Use (R,G,B,A) tuples with values 0-1 instead.
    """
    genres   = list(genre_data.keys())
    pos_vals = [genre_data[g]["pos"] for g in genres]
    neg_vals = [genre_data[g]["neg"] for g in genres]
    x = range(len(genres))
    fig, ax = plt.subplots(figsize=(8,4), facecolor="none")
    ax.bar([i-0.2 for i in x], pos_vals, 0.35,
           label="Positive", color="#34d399", alpha=0.85)
    ax.bar([i+0.2 for i in x], neg_vals, 0.35,
           label="Negative", color="#f87171", alpha=0.85)
    ax.set_xticks(list(x))
    ax.set_xticklabels(genres, color="white", fontsize=8, rotation=25, ha="right")
    ax.set_ylabel("% Reviews", color="white")
    ax.tick_params(colors="white")
    ax.set_facecolor("none")
    # FIX: Use (R,G,B,A) tuple — NOT CSS rgba() string
    spine_color = (1.0, 1.0, 1.0, 0.15)
    for spine in ax.spines.values():
        spine.set_color(spine_color)
    ax.legend(frameon=False, labelcolor="white")
    fig.patch.set_alpha(0)
    plt.tight_layout()
    return fig

def wc_fig(text):
    wc = WordCloud(width=700,height=280,background_color=None,
                   mode="RGBA",colormap="cool",max_words=60).generate(text)
    fig, ax = plt.subplots(figsize=(7,3), facecolor="none")
    ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
    fig.patch.set_alpha(0)
    return fig

def vader_lbl(review):
    s = vader.polarity_scores(review)["compound"]
    if s >= 0.05:  return "Positive", s
    if s <= -0.05: return "Negative", s
    return "Neutral", s

def analyse(movie_key):
    data    = df[df["movie"] == movie_key]
    reviews = data["review"].tolist()
    X       = vectorizer.transform(reviews)
    preds   = model.predict(X)
    probas  = model.predict_proba(X)
    classes = list(model.classes_)
    total   = len(preds)
    pos_pct = list(preds).count("Positive") / total * 100
    neg_pct = list(preds).count("Negative") / total * 100
    neu_pct = list(preds).count("Neutral")  / total * 100
    pi = classes.index("Positive") if "Positive" in classes else None
    ni = classes.index("Negative") if "Negative" in classes else None
    best = worst = None
    bc = wc_v = 0
    for rev, pred, prob in zip(reviews, preds, probas):
        if pi is not None and pred=="Positive" and prob[pi]>bc:
            bc=prob[pi];  best=(rev, bc)
        if ni is not None and pred=="Negative" and prob[ni]>wc_v:
            wc_v=prob[ni]; worst=(rev, wc_v)
    row = data.iloc[0]
    return dict(reviews=reviews, preds=list(preds), probas=probas, classes=classes,
                total=total, pos_pct=pos_pct, neg_pct=neg_pct, neu_pct=neu_pct,
                best=best, worst=worst,
                actor=row["actor"], director=row["director"],
                genre=row["genre"], year=int(row["year"]))

def quick_pos(mv):
    r = df[df["movie"]==mv]["review"].tolist()
    p = model.predict(vectorizer.transform(r))
    return list(p).count("Positive") / len(p) * 100
# ══════════════════════════════════════════════════════════════
#  SIDEBAR
with st.sidebar:
    st.markdown("## 🎖️ Movie Leaderboard")
    st.caption("Ranked by Positive %")

    lb = sorted([(mv.title(), quick_pos(mv)) for mv in ALL_MOVIES],
                key=lambda x: x[1], reverse=True)
    medals = ["🥇","🥈","🥉"]
    for i,(name,score) in enumerate(lb):
        icon = medals[i] if i < 3 else f"{i+1}."
        st.markdown(f"""<div class="lb-entry">
            <span class="lb-rank">{icon}</span>
            <span class="lb-name">{name}</span>
            <span class="lb-score">{score:.0f}% 👍</span>
        </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("### 🔍 Filter Movies")
    sel_actor = st.selectbox("🎭 Actor",    ["All"] + ALL_ACTORS)
    sel_dir   = st.selectbox("🎬 Director", ["All"] + ALL_DIRS)
    sel_genre = st.selectbox("🎭 Genre",    ["All"] + ALL_GENRES)
    if ALL_YEARS:
        yr = st.slider("📅 Year", min_value=min(ALL_YEARS),
                       max_value=max(ALL_YEARS),
                       value=(min(ALL_YEARS), max(ALL_YEARS)))
    else:
        yr = (2015, 2024)

    filt = df.copy()
    if sel_actor != "All": filt = filt[filt["actor"]    == sel_actor]
    if sel_dir   != "All": filt = filt[filt["director"] == sel_dir]
    if sel_genre != "All": filt = filt[filt["genre"]    == sel_genre]
    filt = filt[(filt["year"] >= yr[0]) & (filt["year"] <= yr[1])]
    filt_movies = sorted(filt["movie"].unique().tolist())

    if any([sel_actor!="All", sel_dir!="All", sel_genre!="All"]):
        st.markdown(f"**{len(filt_movies)} movies found:**")
        for mv in filt_movies:
            st.markdown(f"• {mv.title()}")

    st.divider()
    st.markdown("### 🎬 All Movies")
    for mv in ALL_MOVIES:
        st.markdown(f"• {mv.title()}")
# ══════════════════════════════════════════════════════════════
#  HEADER
st.markdown("<h1>🎬 CineBot</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='font-family:DM Sans,sans-serif;opacity:0.65;margin-top:-10px;font-size:1rem;'>"
    "Tamil & Hollywood Sentiment Analysis </p>",
    unsafe_allow_html=True
)
# ══════════════════════════════════════════════════════════════
#  LIVE PREDICTOR
with st.expander("✍️  Predict YOUR own review  —  Live ML Demo", expanded=False):
    user_rev = st.text_area("Type any movie review:",
        placeholder="e.g. The movie was absolutely stunning but a bit slow in the middle...")
    if st.button("🔍  Analyse My Review") and user_rev.strip():
        Xl   = vectorizer.transform([user_rev])
        pl   = model.predict(Xl)[0]
        prbl = model.predict_proba(Xl)[0]
        clsl = list(model.classes_)
        conf = max(prbl)*100
        emap = {"Positive":"😄","Negative":"😡","Neutral":"😐"}
        cmap = {"Positive":"#34d399","Negative":"#f87171","Neutral":"#fb923c"}
        st.markdown(f"""<div class="verdict-card">
            <span class="verdict-emoji">{emap.get(pl,'🤖')}</span>
            <span class="verdict-title">ML says: <strong>{pl}</strong></span><br>
            <span class="verdict-sub">Confidence: {conf:.1f}%</span>
        </div>""", unsafe_allow_html=True)
        st.markdown(sec_div("PROBABILITY BREAKDOWN"), unsafe_allow_html=True)
        for cls, p in zip(clsl, prbl):
            st.markdown(conf_bar(cls, p*100, cmap.get(cls,"#aaa")), unsafe_allow_html=True)
        if VADER_OK:
            vl, vs = vader_lbl(user_rev)
            agree = "✅ Both agree!" if vl==pl else "⚠️ Models disagree!"
            st.markdown(f"**VADER check:** `{vl}` ({vs:.2f}) — {agree}")

st.divider()

# ══════════════════════════════════════════════════════════════
#  CHAT
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

user_input = st.chat_input(
    "Movie name  /  A vs B  /  director:Lokesh  /  actor:Vijay  /  genre:Action ✨"
)

if user_input:
    q = user_input.strip()
    st.session_state.messages.append({"role":"user","content":q})
    with st.chat_message("user"):
        st.markdown(q)

    # ── DIRECTOR SEARCH ───────────────────────────────────────
    if q.lower().startswith("director:"):
        term = q[9:].strip().lower()
        m    = df[df["director"].str.lower().str.contains(term, na=False)]
        if m.empty:
            resp = f"❌ No movies found for director: **{term}**"
            st.session_state.messages.append({"role":"assistant","content":resp})
            with st.chat_message("assistant"): st.markdown(resp)
        else:
            dname = m["director"].iloc[0]
            mvs   = sorted(m["movie"].unique().tolist())
            with st.chat_message("assistant"):
                st.markdown(f"### 🎬 Movies by {dname}")
                for mv in mvs:
                    r = analyse(mv)
                    e,vt,_ = get_verdict(r["pos_pct"],r["neg_pct"],r["neu_pct"])
                    st.markdown(movie_list_card(e, mv.title(),
                        f"{r['year']} • {r['genre']}",
                        f"😄 {r['pos_pct']:.0f}%"), unsafe_allow_html=True)
            st.session_state.messages.append({"role":"assistant","content":f"Director: {dname} ☝️"})

    # ── ACTOR SEARCH ──────────────────────────────────────────
    elif q.lower().startswith("actor:"):
        term = q[6:].strip().lower()
        m    = df[df["actor"].str.lower().str.contains(term, na=False)]
        if m.empty:
            resp = f"❌ No movies found for actor: **{term}**"
            st.session_state.messages.append({"role":"assistant","content":resp})
            with st.chat_message("assistant"): st.markdown(resp)
        else:
            aname = m["actor"].iloc[0]
            mvs   = sorted(m["movie"].unique().tolist())
            with st.chat_message("assistant"):
                st.markdown(f"### 🎭 Movies featuring {aname}")
                for mv in mvs:
                    r = analyse(mv)
                    e,vt,_ = get_verdict(r["pos_pct"],r["neg_pct"],r["neu_pct"])
                    st.markdown(movie_list_card(e, mv.title(),
                        f"{r['year']} • {r['genre']}",
                        f"😄 {r['pos_pct']:.0f}%"), unsafe_allow_html=True)
            st.session_state.messages.append({"role":"assistant","content":f"Actor: {aname} ☝️"})

    # ── GENRE SEARCH ──────────────────────────────────────────
    elif q.lower().startswith("genre:"):
        term = q[6:].strip().lower()
        m    = df[df["genre"].str.lower().str.contains(term, na=False)]
        if m.empty:
            resp = f"❌ No movies found for genre: **{term}**"
            st.session_state.messages.append({"role":"assistant","content":resp})
            with st.chat_message("assistant"): st.markdown(resp)
        else:
            mvs = sorted(m["movie"].unique().tolist())
            with st.chat_message("assistant"):
                st.markdown(f"### 🎭 {term.title()} Movies")
                for mv in mvs:
                    r = analyse(mv)
                    e,vt,_ = get_verdict(r["pos_pct"],r["neg_pct"],r["neu_pct"])
                    st.markdown(movie_list_card(e, mv.title(),
                        f"{r['actor']} • {r['year']}",
                        f"😄 {r['pos_pct']:.0f}%"), unsafe_allow_html=True)
            st.session_state.messages.append({"role":"assistant","content":f"Genre: {term} ☝️"})

    # ── COMPARE ───────────────────────────────────────────────
    elif " vs " in q.lower():
        parts = re.split(r"\s+vs\s+", q, flags=re.IGNORECASE)
        m1, m2 = parts[0].strip().lower(), parts[1].strip().lower()
        miss = []
        for mv in [m1, m2]:
            if mv not in ALL_MOVIES:
                cl = difflib.get_close_matches(mv, ALL_MOVIES, n=1, cutoff=0.5)
                if cl: miss.append(f"'{mv}' — did you mean **{cl[0].title()}**?")
                else:  miss.append(f"'{mv}' not found in dataset.")
        if miss:
            resp = "❌ " + "\n\n".join(miss)
            st.session_state.messages.append({"role":"assistant","content":resp})
            with st.chat_message("assistant"): st.markdown(resp)
        else:
            with st.chat_message("assistant"):
                st.markdown(f"### ⚔️ {m1.title()} vs {m2.title()}")
                r1, r2 = analyse(m1), analyse(m2)
                c1, c2 = st.columns(2)
                with c1:
                    st.pyplot(donut(r1["pos_pct"],r1["neg_pct"],r1["neu_pct"], m1.title()))
                    e,vt,_ = get_verdict(r1["pos_pct"],r1["neg_pct"],r1["neu_pct"])
                    st.markdown(f"<div style='text-align:center;font-family:Cinzel,serif;'>{e} {vt}</div>",
                                unsafe_allow_html=True)
                with c2:
                    st.pyplot(donut(r2["pos_pct"],r2["neg_pct"],r2["neu_pct"], m2.title()))
                    e2,vt2,_ = get_verdict(r2["pos_pct"],r2["neg_pct"],r2["neu_pct"])
                    st.markdown(f"<div style='text-align:center;font-family:Cinzel,serif;'>{e2} {vt2}</div>",
                                unsafe_allow_html=True)
                win  = m1 if r1["pos_pct"] >= r2["pos_pct"] else m2
                diff = abs(r1["pos_pct"]-r2["pos_pct"])
                st.markdown(winner_banner(win, diff), unsafe_allow_html=True)
            st.session_state.messages.append({"role":"assistant",
                "content":f"Compared {m1.title()} vs {m2.title()} ☝️"})

    # ── SINGLE MOVIE ──────────────────────────────────────────
    else:
        mk = q.lower()
        if mk not in ALL_MOVIES:
            cl = difflib.get_close_matches(mk, ALL_MOVIES, n=3, cutoff=0.4)
            if cl:
                sugg = ", ".join([f"**{c.title()}**" for c in cl])
                resp = f"❌ Not found! Did you mean: {sugg}?\n\n💡 Try: `director:Lokesh` • `actor:Vijay` • `genre:Action`"
            else:
                resp = ("❌ Movie not found.\n\nAvailable: " +
                        ", ".join([m.title() for m in ALL_MOVIES[:10]]) + "…\n\n"
                        "💡 Try: `director:name` • `actor:name` • `genre:type`")
            st.session_state.messages.append({"role":"assistant","content":resp})
            with st.chat_message("assistant"): st.markdown(resp)
        else:
            res = analyse(mk)
            with st.chat_message("assistant"):
                st.markdown(meta_row(res["actor"], res["director"],
                                     res["genre"], res["year"]), unsafe_allow_html=True)
                e,vt,vs = get_verdict(res["pos_pct"], res["neg_pct"], res["neu_pct"])
                st.markdown(verdict_card(e, vt, vs, res["total"]), unsafe_allow_html=True)
                st.markdown(stat_row(res["pos_pct"], res["neg_pct"], res["neu_pct"]),
                            unsafe_allow_html=True)

                t = st.tabs(["📊 Chart","💬 Reviews","☁️ Word Cloud","🧠 VADER"])

                with t[0]:
                    col_l, col_c, col_r = st.columns([1, 2, 1])
                    with col_c:
                        st.pyplot(donut(res["pos_pct"],res["neg_pct"],res["neu_pct"],
                                       f"Sentiment — {mk.title()}"))

                with t[1]:
                    if res["best"]:
                        rv, cf = res["best"]
                        st.markdown(highlight_card(rv,"highlight-pos","🌟 Most Positive Review",cf),
                                    unsafe_allow_html=True)
                    if res["worst"]:
                        rv2, cf2 = res["worst"]
                        st.markdown(highlight_card(rv2,"highlight-neg","💢 Most Negative Review",cf2),
                                    unsafe_allow_html=True)
                    st.markdown(sec_div("ALL REVIEWS"), unsafe_allow_html=True)
                    for rv, pred in zip(res["reviews"], res["preds"]):
                        p = {"Positive":"pos-pill","Negative":"neg-pill","Neutral":"neu-pill"}.get(pred,"neu-pill")
                        st.markdown(review_pill(rv, p), unsafe_allow_html=True)

                with t[2]:
                    if WORDCLOUD_OK:
                        st.pyplot(wc_fig(" ".join(res["reviews"])))
                    else:
                        st.info("Run: `pip install wordcloud` to enable word clouds")
                        words = " ".join(res["reviews"]).lower().split()
                        stop  = {"the","a","an","is","was","it","in","of","and","to",
                                 "this","i","for","but","with","very","not","are"}
                        freq  = Counter(w for w in words if w not in stop and len(w)>3)
                        for word,cnt in freq.most_common(15):
                            st.markdown(f"`{word}` × {cnt}")

                with t[3]:
                    if VADER_OK:
                        vp = [vader_lbl(r)[0] for r in res["reviews"]]
                        t_ = len(vp)
                        vpos = vp.count("Positive")/t_*100
                        vneg = vp.count("Negative")/t_*100
                        vneu = vp.count("Neutral") /t_*100
                        agr  = sum(1 for m,v in zip(res["preds"],vp) if m==v)/t_*100
                        c1,c2 = st.columns(2)
                        with c1:
                            st.markdown("**🤖 ML Model**")
                            st.pyplot(donut(res["pos_pct"],res["neg_pct"],res["neu_pct"],"ML"))
                        with c2:
                            st.markdown("**📖 VADER**")
                            st.pyplot(donut(vpos,vneg,vneu,"VADER"))
                        if agr >= 70: st.success(f"✅ Both agree {agr:.0f}% — High confidence!")
                        else:         st.warning(f"⚠️ Agree only {agr:.0f}% — Mixed signals.")
                    else:
                        st.info("Run: `pip install nltk` then `nltk.download('vader_lexicon')`")

            st.session_state.messages.append({
                "role":"assistant",
                "content":f"📊 Analysis for **{mk.title()}** ☝️"
            })