import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
from nltk.stem import SnowballStemmer

# ========================
# CONFIGURACIÓN DE PÁGINA
# ========================
st.set_page_config(
    page_title="🔎 Detective Semántico: Buscador Inteligente TF-IDF",
    page_icon="🕵️‍♀️",
    layout="centered"
)

# ========================
# ENCABEZADO Y ESTILO
# ========================
st.markdown("""
<div style='text-align:center;'>
    <h1 style='color:#4B0082;'>🕵️‍♀️ Detective Semántico</h1>
    <h3 style='color:#9370DB;'>Encuentra las pistas ocultas entre tus palabras con TF-IDF</h3>
</div>
""", unsafe_allow_html=True)

st.write("""
**Detective Semántico** te ayuda a encontrar el texto más relevante entre tus documentos.  
Cada línea que escribas será analizada como una pista 📜, y tu pregunta será la clave 🗝️  
para descubrir qué texto tiene la información más relacionada.

> 🗣️ *Por ahora solo funciona en inglés para aprovechar el análisis lingüístico completo.*
""")

# ========================
# ENTRADA DE DATOS
# ========================
st.markdown("### 📚 Ingresa tus documentos:")
text_input = st.text_area(
    "Cada línea es un documento independiente:",
    "The dog barks loudly.\nThe cat meows at night.\nThe dog and the cat play together.",
    height=150
)

st.markdown("### ❓ Ingresa tu pregunta:")
question = st.text_input("Ejemplo:", "Who is playing?")

# Inicializar stemmer
stemmer = SnowballStemmer("english")

def tokenize_and_stem(text: str):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = [t for t in text.split() if len(t) > 1]
    stems = [stemmer.stem(t) for t in tokens]
    return stems

# ========================
# ANÁLISIS
# ========================
if st.button("🔍 Analizar y buscar respuesta"):
    documents = [d.strip() for d in text_input.split("\n") if d.strip()]

    if len(documents) < 1:
        st.warning("⚠️ Ingresa al menos un documento para analizar.")
    else:
        with st.spinner("El detective está revisando tus documentos... 🕵️‍♀️"):
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
                index=[f"Doc {i+1}" for i in range(len(documents))]
            )

            st.markdown("### 🧮 Matriz TF-IDF")
            st.dataframe(df_tfidf.round(3))

            # Calcular similitud coseno
            question_vec = vectorizer.transform([question])
            similarities = cosine_similarity(question_vec, X).flatten()

            best_idx = similarities.argmax()
            best_doc = documents[best_idx]
            best_score = similarities[best_idx]

            st.markdown("---")
            st.markdown("### 🧠 Resultado del análisis")

            st.success(f"""
            **Pregunta:** {question}  
            **Documento más relevante:** Doc {best_idx+1}  
            **Texto:** *"{best_doc}"*  
            **Similitud:** {best_score:.3f}
            """)

            # Mostrar tabla de similitud
            sim_df = pd.DataFrame({
                "Documento": [f"Doc {i+1}" for i in range(len(documents))],
                "Texto": documents,
                "Similitud": similarities
            }).sort_values("Similitud", ascending=False)

            st.markdown("### 📊 Ranking de similitud entre documentos")
            st.dataframe(sim_df)

            # Mostrar coincidencias de stems
            vocab = vectorizer.get_feature_names_out()
            q_stems = tokenize_and_stem(question)
            matched = [s for s in q_stems if s in vocab and df_tfidf.iloc[best_idx].get(s, 0) > 0]

            if matched:
                st.markdown("### 🧩 Pistas encontradas (stems coincidentes)")
                st.write(", ".join(matched))
            else:
                st.info("No se encontraron coincidencias directas de palabras base.")

# ========================
# PIE DE PÁGINA
# ========================
st.markdown("""
<hr>
<div style='text-align:center; color:gray;'>
Hecho con 🧠 + 💜 por un curioso detective de palabras.
</div>
""", unsafe_allow_html=True)

