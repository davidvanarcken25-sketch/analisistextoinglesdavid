import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
from nltk.stem import SnowballStemmer
from streamlit_drawable_canvas import st_canvas

st.title("📚 + 🎨 App Demo: Texto y Dibujo")

# Pestañas
tab1, tab2 = st.tabs(["🔍 TF-IDF", "🎨 Lienzo de dibujo"])

# ---------------- TAB 1: TF-IDF ----------------
with tab1:
    # Documentos de ejemplo
    default_docs = """La lluvia caía sobre el pueblo vacío. Entre las calles empedradas, una niña avanzaba con una linterna apagada. Buscaba la casa donde, según su abuela, vivía el relojero que podía reparar el tiempo.
    Cuando finalmente lo encontró, el anciano le pidió su reloj. Ella no llevaba ninguno; en cambio, le ofreció un retrato roto de su familia. El hombre sonrió, lo colocó dentro de una esfera de cristal y lo agitó suavemente.
    Al instante, el viento cambió de dirección y, en cada ventana, la niña vio reflejadas las escenas de su pasado. Una por una, como si el tiempo hubiera sido en verdad reparado.."""

    # Stemmer en español
    stemmer = SnowballStemmer("spanish")

    def tokenize_and_stem(text):
        text = text.lower()
        text = re.sub(r'[^a-záéíóúüñ\s]', ' ', text)
        tokens = [t for t in text.split() if len(t) > 1]
        stems = [stemmer.stem(t) for t in tokens]
        return stems

    col1, col2 = st.columns([2, 1])

    with col1:
        text_input = st.text_area("📝 Documentos (uno por línea):", default_docs, height=150)
        question = st.text_input("❓ Escribe tu pregunta:", "¿Dónde juegan el perro y el gato?")

    with col2:
        st.markdown("### 💡 Preguntas sugeridas:")
        if st.button("¿Por qué la niña buscaba al relojero?", use_container_width=True):
            st.session_state.question = "¿Por qué la niña buscaba al relojero que podía reparar el tiempo?"
            st.rerun()
        if st.button("¿Qué significa la linterna apagada?", use_container_width=True):
            st.session_state.question = "¿Qué significado tiene la linterna apagada que lleva?"
            st.rerun()
        if st.button("¿Por qué sonríe el relojero?", use_container_width=True):
            st.session_state.question = "¿Por qué crees que el relojero sonríe cuando ella le da el retrato roto?"
            st.rerun()

    if 'question' in st.session_state:
        question = st.session_state.question

    if st.button("🔍 Analizar", type="primary"):
        documents = [d.strip() for d in text_input.split("\n") if d.strip()]
        if len(documents) < 1:
            st.error("⚠️ Ingresa al menos un documento.")
        elif not question.strip():
            st.error("⚠️ Escribe una pregunta.")
        else:
            vectorizer = TfidfVectorizer(tokenizer=tokenize_and_stem, min_df=1)
            X = vectorizer.fit_transform(documents)
            st.markdown("### 📊 Matriz TF-IDF")
            df_tfidf = pd.DataFrame(
                X.toarray(),
                columns=vectorizer.get_feature_names_out(),
                index=[f"Doc {i+1}" for i in range(len(documents))]
            )
            st.dataframe(df_tfidf.round(3), use_container_width=True)

            question_vec = vectorizer.transform([question])
            similarities = cosine_similarity(question_vec, X).flatten()

            best_idx = similarities.argmax()
            best_doc = documents[best_idx]
            best_score = similarities[best_idx]

            st.markdown("### 🎯 Respuesta")
            st.markdown(f"**Tu pregunta:** {question}")
            if best_score > 0.01:
                st.success(f"**Respuesta:** {best_doc}")
                st.info(f"📈 Similitud: {best_score:.3f}")
            else:
                st.warning(f"**Respuesta (baja confianza):** {best_doc}")
                st.info(f"📉 Similitud: {best_score:.3f}")

# ---------------- TAB 2: Lienzo de dibujo ----------------
with tab2:
    st.subheader("Dibuja aquí 👇")

    brush_color = st.color_picker("🎨 Elige el color del pincel", "#000000")
    stroke_width = st.slider("✏️ Grosor del pincel", 1, 25, 3)
    shapes = ["freedraw", "line", "rect", "circle", "transform"]
    shape = st.selectbox("🔲 Forma predeterminada:", shapes)

    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0.3)",
        stroke_width=stroke_width,
        stroke_color=brush_color,
        background_color="#eee",
        update_streamlit=True,
        height=400,
        width=600,
        drawing_mode=shape,
        key="canvas"
    )

    if canvas_result.image_data is not None:
        st.image(canvas_result.image_data, caption="🖼️ Tu dibujo", use_container_width=True)

