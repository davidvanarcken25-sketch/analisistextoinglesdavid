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
    page_title="Análisis de texto (inglés) – El Detective Semántico",
    layout="centered"
)

# ========================
# CABECERA Y ESTILO
# ========================
st.markdown("""
<div style='text-align:center;'>
    <h1 style='color:#2F2F4F;'>Análisis de texto (inglés)</h1>
    <h3 style='color:#5A5A8F;'>El Detective Semántico</h3>
    <p style='color:#666; font-size:16px;'>
        Explora la relación entre tus textos y una pregunta usando el modelo TF-IDF.
        Cada línea de texto se considera un documento, y el detective semántico investigará
        cuál contiene la respuesta más relevante según su similitud semántica.
    </p>
</div>
""", unsafe_allow_html=True)

# Imagen decorativa (más pequeña y debajo del título)
try:
    image = Image.open("detective_banner.jpg")  # Asegúrate de tener esta imagen en la carpeta del proyecto
    st.image(image, width=300, caption="El Detective Semántico en acción", use_container_width=False)
except Exception:
    st.info("Puedes agregar una imagen llamada 'detective_banner.jpg' para decorar la app.")

# ========================
# ENTRADA DE DATOS
# ========================
st.markdown("#### Documentos a analizar (en inglés)")
text_input = st.text_area(
    "Cada línea será tratada como un documento independiente:",
    "The dog barks loudly.\nThe cat meows at night.\nThe dog and the cat play together.",
    height=150
)

st.markdown("#### Pregunta a investigar (en inglés)")
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
if st.button("Iniciar análisis"):
    documents = [d.strip() for d in text_input.split("\n") if d.strip()]

    if len(documents) < 1:
        st.warning("Por favor, ingresa al menos un documento para analizar.")
    else:
        with st.spinner("El detective está revisando tus documentos..."):
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
                index=[f"Documento {i+1}" for i in range(len(documents))]
            )

            st.markdown("### Matriz TF-IDF")
            st.dataframe(df_tfidf.round(3))

            # Calcular similitud coseno
            question_vec = vectorizer.transform([question])
            similarities = cosine_similarity(question_vec, X).flatten()

            best_idx = similarities.argmax()
            best_doc = documents[best_idx]
            best_score = similarities[best_idx]

            st.markdown("---")
            st.markdown("### Resultados del análisis")

            st.success(f"""
            **Pregunta analizada:** {question}  
            **Documento más relevante:** Documento {best_idx+1}  
            **Texto:** "{best_doc}"  
            **Nivel de similitud:** {best_score:.3f}
            """)

            # Mostrar tabla de similitud
            sim_df = pd.DataFrame({
                "Documento": [f"Documento {i+1}" for i in range(len(documents))],
                "Texto": documents,
                "Similitud": similarities
            }).sort_values("Similitud", ascending=False)

            st.markdown("### Ranking de similitud entre documentos")
            st.dataframe(sim_df)

            # Mostrar coincidencias de stems
            vocab = vectorizer.get_feature_names_out()
            q_stems = tokenize_and_stem(question)
            matched = [s for s in q_stems if s in vocab and df_tfidf.iloc[best_idx].get(s, 0) > 0]

            if matched:
                st.markdown("### Palabras base coincidentes (stems encontrados)")
                st.write(", ".join(matched))
            else:
                st.info("No se encontraron coincidencias directas de palabras base.")

# ========================
# PIE DE PÁGINA
# ========================
st.markdown("""
<hr>
<div style='text-align:center; color:gray; font-size:14px;'>
Proyecto de análisis semántico TF-IDF · El Detective Semántico · Versión demostrativa
</div>
""", unsafe_allow_html=True)

