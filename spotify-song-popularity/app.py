import joblib
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Spotify Song Popularity Prediction",
    page_icon="🎵",
    layout="centered"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    model_package = joblib.load("models/best_model.pkl")
    return model_package["model"], model_package["features"]


model, features = load_model()


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():
    return pd.read_csv("data/spotify_dataset.csv")


df = load_dataset()


# ============================================================
# HEADER
# ============================================================

st.title("🎵 Spotify Song Popularity Prediction")

st.write(
    "Enter a song name and the trained Random Forest model "
    "will predict its popularity category."
)

st.info(
    """
    **Popularity Categories**

    0–39 → Not Popular  
    40–69 → Moderately Popular  
    70–100 → Popular
    """
)


# ============================================================
# SONG NAME INPUT
# ============================================================

st.header("🔎 Search for a Song")

song_name = st.text_input(
    "Enter Song Name",
    placeholder="Example: Blinding Lights"
)


# ============================================================
# PREDICTION
# ============================================================

if st.button(
    "🔮 Predict Popularity",
    use_container_width=True
):

    if not song_name.strip():

        st.warning("Please enter a song name first.")

    else:

        search_name = song_name.strip().lower()

        # ----------------------------------------------------
        # Exact match
        # ----------------------------------------------------

        exact_matches = df[
            df["track_name"]
            .fillna("")
            .str.lower()
            .eq(search_name)
        ]

        # ----------------------------------------------------
        # Partial match if exact match not found
        # ----------------------------------------------------

        if exact_matches.empty:

            matches = df[
                df["track_name"]
                .fillna("")
                .str.lower()
                .str.contains(
                    search_name,
                    regex=False
                )
            ]

        else:

            matches = exact_matches


        # ----------------------------------------------------
        # Song not found
        # ----------------------------------------------------

        if matches.empty:

            st.error(
                f"❌ '{song_name}' was not found in the dataset."
            )

            st.write(
                "Please try another song name that exists "
                "in the Spotify dataset."
            )


        else:

            # Use the first matching song
            song = matches.iloc[0]

            # ------------------------------------------------
            # Song information
            # ------------------------------------------------

            st.success(
                f"✅ Song found: {song['track_name']}"
            )

            if pd.notna(song["artists"]):

                st.write(
                    f"**Artist:** {song['artists']}"
                )

            if len(matches) > 1:

                st.caption(
                    f"{len(matches)} matching songs were found. "
                    "The first matching record was used."
                )


            # ------------------------------------------------
            # Get required audio features
            # ------------------------------------------------

            song_features = pd.DataFrame(
                [[
                    song["danceability"],
                    song["energy"],
                    song["loudness"],
                    song["speechiness"],
                    song["acousticness"],
                    song["instrumentalness"],
                    song["liveness"],
                    song["valence"],
                    song["tempo"]
                ]],
                columns=features
            )


            # ------------------------------------------------
            # Make prediction
            # ------------------------------------------------

            prediction = model.predict(
                song_features
            )[0]

            probabilities = model.predict_proba(
                song_features
            )[0]


            probability_data = pd.DataFrame({
                "Category": model.classes_,
                "Probability": probabilities * 100
            })


            # ------------------------------------------------
            # Display prediction
            # ------------------------------------------------

            st.header("📊 Prediction Result")

            if prediction == "Popular":

                st.success(
                    f"🎉 Predicted Category: **{prediction}**"
                )

            elif prediction == "Moderately Popular":

                st.warning(
                    f"🎧 Predicted Category: **{prediction}**"
                )

            else:

                st.error(
                    f"📉 Predicted Category: **{prediction}**"
                )


            # ------------------------------------------------
            # Prediction probabilities
            # ------------------------------------------------

            st.subheader("Prediction Probabilities")

            probability_display = probability_data.copy()

            probability_display["Probability"] = (
                probability_display["Probability"].round(2)
            )

            st.dataframe(
                probability_display,
                use_container_width=True,
                hide_index=True
            )


            # ------------------------------------------------
            # Audio features used by the model
            # ------------------------------------------------

            with st.expander(
                "🔍 View Audio Features Used"
            ):

                st.dataframe(
                    song_features,
                    use_container_width=True,
                    hide_index=True
                )


# ============================================================
# PROJECT INFORMATION
# ============================================================

st.divider()

st.subheader("ℹ️ About the Project")

st.write(
    """
    This project uses supervised machine learning to classify
    Spotify songs into three popularity categories based on
    nine audio features.

    Three classification models were evaluated:

    • Logistic Regression  
    • Decision Tree  
    • Random Forest

    Random Forest was selected as the final model because it
    achieved the highest test performance.
    """
)

st.caption(
    "Random Forest Accuracy: 74.43% | "
    "Weighted F1-Score: 0.7456"
)