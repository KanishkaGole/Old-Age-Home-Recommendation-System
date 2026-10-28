from pathlib import Path
import sys
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from recommender import recommend, log_rating

MODEL_PATH = ROOT / "artifacts" / "content_model.joblib"
RATING_PATH = ROOT / "data" / "user_ratings.csv"

st.set_page_config(page_title="ElderCare Finder", page_icon="🏡", layout="wide")
st.title("🏡 ElderCare Finder")
st.caption("Content-based old-age-home discovery • India-wide directory")
st.info(
    "Recommendations are based on the supplied directory, not verified live availability. "
    "Confirm fees, vacancies, medical support, safety, and eligibility directly with each home."
)

@st.cache_resource
def load_bundle(path):
    return joblib.load(path)

if not MODEL_PATH.exists():
    st.error("The trained model artifact was not found.")
    st.code("python train_model.py")
    st.stop()

bundle = load_bundle(str(MODEL_PATH))
df = bundle["data"]

with st.sidebar:
    st.header("Your requirements")
    st.write("Selections below are **hard filters**. Results that do not match are excluded.")
    state_options = ["Any"] + sorted(df["State"].dropna().astype(str).unique().tolist())
    state = st.selectbox("Preferred state / territory", state_options)

    district_values = df["District"].dropna().astype(str)
    if state != "Any":
        district_values = df.loc[df["State"].astype(str).eq(state), "District"].dropna().astype(str)
    district_options = ["Any"] + sorted(district_values.unique().tolist())
    district = st.selectbox("Preferred district", district_options)

    facility_options = ["Any"] + sorted(df["Facility Type"].dropna().astype(str).unique().tolist())
    facility_type = st.selectbox("Facility / fee category", facility_options)

    care_options = ["Any"] + sorted(df["Care Type"].dropna().astype(str).unique().tolist())
    care_type = st.selectbox("Care type", care_options)

    gender_options = ["Any", "Men", "Women"] + sorted(
        [v for v in df["Gender"].dropna().astype(str).unique().tolist()
         if v.casefold() not in {"men", "women"}]
    )
    gender = st.selectbox("Resident eligibility", gender_options)

    management_options = ["Any"] + sorted(df["Mgmt"].dropna().astype(str).unique().tolist())
    management = st.selectbox("Management type", management_options)

    top_n = st.slider("Number of recommendations", 3, 20, 10)

st.subheader("Describe the senior citizen's preferences")
query = st.text_area(
    "Optional preferences",
    placeholder="Example: free basic care home for women, peaceful environment, accessible location",
    help="Text is matched against information present in the directory. Unsupported needs cannot be verified by this dataset."
)
prioritize_contact = st.checkbox("Prefer listings with more complete contact information", value=True)

if st.button("Find suitable homes", type="primary", use_container_width=True):
    results = recommend(
        bundle, query=query, state=state, district=district,
        facility_type=facility_type, care_type=care_type, gender=gender,
        management=management, top_n=top_n, prioritize_contact=prioritize_contact
    )
    st.session_state["results"] = results
    st.session_state["query"] = query

results = st.session_state.get("results")
if results is not None:
    st.subheader(f"Recommendations ({len(results)})")
    if results.empty:
        st.warning("No listings matched all selected hard filters. Relax one or more filters and try again.")
    else:
        st.caption("Score is a ranking signal, not a probability of suitability or a quality rating.")
        for idx, row in results.iterrows():
            title = row.get("Home Name", "Unnamed listing")
            district_text = row.get("District", "")
            state_text = row.get("State", "")
            with st.container(border=True):
                left, right = st.columns([4, 1])
                with left:
                    st.markdown(f"### {row.get('recommendation_rank', '')}. {title}")
                    st.write(" • ".join([str(x) for x in [district_text, state_text] if pd.notna(x) and str(x).strip()]))
                    st.write(f"**Facility:** {row.get('Facility Type', 'Not listed')}  |  **Care:** {row.get('Care Type', 'Not listed')}")
                    st.write(f"**Gender eligibility:** {row.get('Gender', 'Not listed')}  |  **Management:** {row.get('Mgmt', 'Not listed')}")
                    cap = row.get("Bed Capacity")
                    st.write(f"**Reported bed capacity:** {int(cap) if pd.notna(cap) and float(cap).is_integer() else cap if pd.notna(cap) else 'Not listed'}")
                    st.write(f"**Address:** {row.get('Address') if pd.notna(row.get('Address')) else 'Not listed'}")
                    phone = row.get("Telephone")
                    email = row.get("Email")
                    st.write(f"**Phone:** {phone if pd.notna(phone) else 'Not listed'}  |  **Email:** {email if pd.notna(email) else 'Not listed'}")
                    if bool(row.get("capacity_needs_verification", True)):
                        st.warning("Bed capacity is missing or zero in the source; verify directly.")
                    if pd.isna(row.get("Telephone")) and pd.isna(row.get("Email")):
                        st.warning("No phone/email is listed in this record.")
                    source = row.get("Source")
                    if pd.notna(source):
                        st.caption(f"Source field: {source}")
                with right:
                    st.metric("Match score", f"{float(row.get('recommendation_score', 0)):.3f}")
                    st.caption(f"Text similarity: {float(row.get('similarity_score', 0)):.3f}")
                    st.caption("Not a probability")

                with st.expander("Rate this recommendation"):
                    with st.form(f"rating_form_{idx}"):
                        rating = st.select_slider("How relevant is this home?", options=[1, 2, 3, 4, 5], value=3)
                        notes = st.text_input("Optional feedback (e.g., missing service, wrong contact)")
                        submitted = st.form_submit_button("Save rating")
                        if submitted:
                            log_rating(
                                RATING_PATH, idx, title, rating,
                                query=st.session_state.get("query", ""), notes=notes,
                                recommendation_rank=int(row.get("recommendation_rank", 0)),
                                recommendation_score=float(row.get("recommendation_score", 0))
                            )
                            st.success("Rating saved locally. Thank you for the feedback.")

    st.download_button(
        "Download current recommendations as CSV",
        data=results.to_csv(index=False).encode("utf-8-sig"),
        file_name="old_age_home_recommendations.csv",
        mime="text/csv"
    )

st.divider()
st.subheader("About the model")
st.markdown("""
- **Model:** TF-IDF content-based retrieval with cosine similarity.
- **Hard constraints:** state, district, facility category, care type, gender eligibility, and management type.
- **Soft ranking:** text similarity and (optionally) contact-information completeness.
- **Rating capture:** relevance ratings are saved to `data/user_ratings.csv`.
- **Not included in source data:** reliable monthly fees, real-time vacancies, verified medical services, accessibility, quality inspections, and user histories. These should be collected and verified before they are used as decision constraints.
""")
