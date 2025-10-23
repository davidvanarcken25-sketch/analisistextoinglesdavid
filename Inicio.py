import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
from nltk.stem import SnowballStemmer
from PIL import Image

# ========================
# CONFIGURACIÓN DE PÁGINA
# ========================
st.set_page_config(
    page_title="🍽️ El Crítico Gastronómico",
    layout="centered"
)

# ========================
# CABECERA Y ESTILO
# ========================
st.markdown("""
<div style='text-align:center;'>
    <h1 style='color:#8B0000;'>🍽️ El Crítico Gastronómico</h1>
    <h3 style='color:#A0522D;'>Analizador de reseñas culinarias</h3>
    <p style='color:#444; font-size:16px;'>
        Ingresa diferentes reseñas de comidas o platos, y este crítico te dirá cuál
        responde mejor a una pregunta gastronómica según la similitud semántica TF-IDF.
        <br><br>
        Cada línea se considera una reseña distinta. ¡Averigüemos qué plato conquista tu paladar!
    </p>
</div>
""", unsafe_allow_html=True)

# Imagen decorativa
try:
    image = Image.open("chef_banner.jpg")  # Puedes agregar una imagen con este nombre
    st.image(image, width=350, caption="El Crítico Gastronómico en acción", use_container_width=False)
except Exception:
    st.info("Puedes agregar una imagen llamada 'chef_banner.jpg' para decorar la app.")

# ========================
# ENTRADA DE DATOS
# ========================
st.markdown("#### Reseñas gastronómicas (en inglés o español simple)")
text_input = st.text_area(
    "Cada línea será tratada como una reseña independiente:",
    "The pasta was delicious and creamy.\nThe soup was too salty.\nThe pizza had a perfect crust and rich flavor.",
    height=150
)

st.markdown("#### Pregunta culinaria")
question = st.text_input("Ejemplo:", "Which dish was too salty?")

# Inicializar stemmer
stemmer = SnowballStemmer("english")

def tokenize_and_stem(text: str):
    text = text.lower()
    text = re.sub(r'[^a-záéíóúñ\s]', ' ', text)
    tokens = [t for t in text.split() if len(t) > 1]
    stems = [stemmer.stem(t) for t in tokens]
    return stems

# ========================
# ANÁLISIS
# ========================
if st.button("Analizar reseñas"):
    documents = [d.strip() for d in text_input.split("\n") if d.strip()]

    if len(documents) < 1:
        st.warning("Por favor, ingresa al menos una reseña para analizar.")
    else:
        with st.spinner("El crítico está saboreando tus reseñas... 🍷"):
            vectorizer = TfidfVectorizer(
                tokenizer=tokenize_and_stem,
                stop_words="english",
                token_pattern=None
            )

            X = vectorizer.fit_transform(documents)

            # Crear DataFrame TF-IDF
            df_tfidf = pd.DataFrame(
                X.toarray(),
                columns=vectorizer.get_feature_names_out(),
                index=[f"Reseña {i+1}" for i in range(len(documents))]
            )

            st.markdown("### 🧂 Matriz TF-IDF de sabores")
            st.dataframe(df_tfidf.round(3))

            # Calcular similitud coseno
            question_vec = vectorizer.transform([question])
            similarities = cosine_similarity(question_vec, X).flatten()

            best_idx = similarities.argmax()
            best_doc = documents[best_idx]
            best_score = similarities[best_idx]

            st.markdown("---")
            st.markdown("### 🍲 Resultado del análisis")

            st.success(f"""
            **Pregunta:** {question}  
            **Reseña más relevante:** Reseña {best_idx+1}  
            **Texto:** "{best_doc}"  
            **Nivel de similitud:** {best_score:.3f}
            """)

            # Mostrar ranking de similitud
            sim_df = pd.DataFrame({
                "Reseña": [f"Reseña {i+1}" for i in range(len(documents))],
                "Texto": documents,
                "Similitud": similarities
            }).sort_values("Similitud", ascending=False)

            st.markdown("### 🧁 Ranking de similitud entre reseñas")
            st.dataframe(sim_df)

            # Palabras coincidentes (stems)
            vocab = vectorizer.get_feature_names_out()
            q_stems = tokenize_and_stem(question)
            matched = [s for s in q_stems if s in vocab and df_tfidf.iloc[best_idx].get(s, 0) > 0]

            if matched:
                st.markdown("### 🔍 Palabras base coincidentes (sabores detectados)")
                st.write(", ".join(matched))
            else:
                st.info("No se encontraron coincidencias directas entre la pregunta y las reseñas.")

# ========================
# PIE DE PÁGINA
# ========================
st.markdown("""
<hr>
<div style='text-align:center; color:gray; font-size:14px;'>
Proyecto de análisis semántico TF-IDF · 🍽️ El Crítico Gastronómico · Versión demostrativa
</div>
""", unsafe_allow_html=True)
