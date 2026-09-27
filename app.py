import math
import re
import unicodedata
from collections import Counter, defaultdict

import streamlit as st


# ============================================================
# AGROSEARCH - Motor de Busca Inteligente
# Laboratório Prático 04 - Desafio Integrador
# ============================================================

DOCUMENTS = {
    1: "A soja requer irrigação constante durante o período de floração para garantir a produtividade.",
    2: "O controle biológico de lagartas na soja pode ser feito com a vespa Trichogramma.",
    3: "A adubação verde com leguminosas melhora o nitrogênio no solo para o milho.",
    4: "Lagartas desfolhadoras causam grande prejuízo na cultura da soja e do algodão.",
    5: "A irrigação por gotejamento economiza água e é ideal para o cultivo orgânico.",
}

# Stopwords básicas em português. O usuário pode ligar/desligar seu uso.
STOPWORDS = {
    "a", "ao", "aos", "aquela", "aquelas", "aquele", "aqueles", "aquilo",
    "as", "até", "com", "como", "da", "das", "de", "do", "dos", "e", "é",
    "em", "entre", "essa", "essas", "esse", "esses", "esta", "estas",
    "este", "estes", "eu", "foi", "há", "isso", "isto", "já", "mas",
    "me", "na", "nas", "não", "no", "nos", "num", "numa", "o", "os",
    "ou", "para", "pela", "pelas", "pelo", "pelos", "por", "que", "se",
    "sem", "sua", "suas", "também", "um", "uma", "umas", "uns",
}

# Stemmer simples implementado no próprio projeto, sem scikit-learn.
# O objetivo é demonstrar a etapa de stemming sem depender de biblioteca
# de alto nível.
SUFFIXES = [
    "mente", "amento", "imentos", "imento", "amentos",
    "adoras", "adores", "adora", "ador",
    "ações", "ação", "ções", "ção",
    "ismos", "ismo", "istas", "ista",
    "idades", "idade",
    "ezas", "eza",
    "mente", "mente",
    "ando", "endo", "indo",
    "ados", "adas", "idos", "idas",
    "ando", "endo", "indo",
    "os", "as", "es",
    "o", "a", "e",
]

MIN_STEM_LENGTH = 3


def remove_accents(text: str) -> str:
    """Remove acentos/diacríticos e normaliza Unicode."""
    normalized = unicodedata.normalize("NFD", text)
    return "".join(
        char for char in normalized
        if unicodedata.category(char) != "Mn"
    )


def tokenize(text: str) -> list[str]:
    """Tokenização simples: mantém apenas sequências alfanuméricas."""
    text = remove_accents(text.lower())
    return re.findall(r"[a-z0-9]+", text)


def stem(word: str) -> str:
    """Stemmer simples por remoção de sufixos."""
    for suffix in sorted(SUFFIXES, key=len, reverse=True):
        if word.endswith(suffix) and len(word) - len(suffix) >= MIN_STEM_LENGTH:
            return word[:-len(suffix)]
    return word


def preprocess(
    text: str,
    use_stopwords: bool = True,
    use_stemming: bool = True,
) -> list[str]:
    """Executa tokenização, normalização, stopwords e stemming."""
    tokens = tokenize(text)

    if use_stopwords:
        tokens = [token for token in tokens if token not in STOPWORDS]

    if use_stemming:
        tokens = [stem(token) for token in tokens]

    return tokens


def build_inverted_index(processed_docs: dict[int, list[str]]) -> dict[str, list[int]]:
    """Constrói Termo -> [IDs de documentos]."""
    index = defaultdict(set)

    for doc_id, tokens in processed_docs.items():
        for token in set(tokens):
            index[token].add(doc_id)

    return {
        term: sorted(doc_ids)
        for term, doc_ids in sorted(index.items())
    }


def term_frequency(tokens: list[str]) -> dict[str, float]:
    """TF normalizado: frequência do termo / número total de tokens."""
    if not tokens:
        return {}

    counts = Counter(tokens)
    total = len(tokens)
    return {term: count / total for term, count in counts.items()}


def inverse_document_frequency(
    term: str,
    processed_docs: dict[int, list[str]],
) -> float:
    """IDF suavizado: log(N / (1 + df)) + 1."""
    n_docs = len(processed_docs)
    df = sum(1 for tokens in processed_docs.values() if term in set(tokens))
    return math.log(n_docs / (1 + df)) + 1


def tf_idf_vector(
    tokens: list[str],
    processed_docs: dict[int, list[str]],
) -> dict[str, float]:
    """Calcula o vetor TF-IDF de um documento."""
    tf = term_frequency(tokens)
    return {
        term: frequency * inverse_document_frequency(term, processed_docs)
        for term, frequency in tf.items()
    }


def query_scores(
    query_tokens: list[str],
    processed_docs: dict[int, list[str]],
) -> dict[int, dict]:
    """Calcula TF, IDF e TF-IDF acumulado por documento para a consulta."""
    results = {}

    query_tf = term_frequency(query_tokens)

    for doc_id, tokens in processed_docs.items():
        doc_tf = term_frequency(tokens)
        matched_terms = []

        score = 0.0
        for term in query_tf:
            if term in doc_tf:
                idf = inverse_document_frequency(term, processed_docs)
                # TF da consulta * TF do documento * IDF.
                contribution = query_tf[term] * doc_tf[term] * (idf ** 2)
                score += contribution
                matched_terms.append(term)

        results[doc_id] = {
            "score": score,
            "matched_terms": matched_terms,
        }

    return results


def cosine_similarity(
    query_tokens: list[str],
    doc_tokens: list[str],
    processed_docs: dict[int, list[str]],
) -> float:
    """Similaridade de cosseno entre vetor TF-IDF da query e do documento."""
    query_tf = term_frequency(query_tokens)
    doc_tf = term_frequency(doc_tokens)

    vocabulary = set(query_tf) | set(doc_tf)

    query_vector = {}
    doc_vector = {}

    for term in vocabulary:
        idf = inverse_document_frequency(term, processed_docs)
        query_vector[term] = query_tf.get(term, 0.0) * idf
        doc_vector[term] = doc_tf.get(term, 0.0) * idf

    dot = sum(query_vector[t] * doc_vector[t] for t in vocabulary)
    query_norm = math.sqrt(sum(value ** 2 for value in query_vector.values()))
    doc_norm = math.sqrt(sum(value ** 2 for value in doc_vector.values()))

    if query_norm == 0 or doc_norm == 0:
        return 0.0

    return dot / (query_norm * doc_norm)


def build_tf_idf_table(processed_docs: dict[int, list[str]]) -> list[dict]:
    """Monta dados para exibição dos vetores TF-IDF."""
    rows = []

    for doc_id, tokens in processed_docs.items():
        vector = tf_idf_vector(tokens, processed_docs)
        for term, value in sorted(vector.items()):
            rows.append({
                "Documento": f"Doc {doc_id}",
                "Termo": term,
                "TF": round(term_frequency(tokens)[term], 6),
                "IDF": round(inverse_document_frequency(term, processed_docs), 6),
                "TF-IDF": round(value, 6),
            })

    return rows


# ----------------------------- UI -----------------------------

st.set_page_config(
    page_title="AgroSearch",
    page_icon="🌱",
    layout="wide",
)

st.title("🌱 AgroSearch")
st.subheader("Motor de Busca Inteligente para Agricultura")

st.write(
    "Protótipo de recuperação de informação com pré-processamento, "
    "índice invertido, TF-IDF e similaridade de cosseno."
)

st.divider()

# Sidebar
st.sidebar.header("⚙️ Configurações")
use_stopwords = st.sidebar.checkbox("Remover Stopwords", value=True)
use_stemming = st.sidebar.checkbox("Aplicar Stemming", value=True)

st.sidebar.divider()
st.sidebar.caption(
    "O índice e os cálculos são reconstruídos conforme as opções "
    "de pré-processamento."
)

processed_docs = {
    doc_id: preprocess(
        text,
        use_stopwords=use_stopwords,
        use_stemming=use_stemming,
    )
    for doc_id, text in DOCUMENTS.items()
}

inverted_index = build_inverted_index(processed_docs)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🔎 Busca",
    "🧹 Pré-processamento",
    "📚 Índice Invertido",
    "📐 TF-IDF",
])

# ============================================================
# TAB 1 - BUSCA
# ============================================================
with tab1:
    st.header("Busca e Ranqueamento")

    query = st.text_input(
        "Digite sua consulta:",
        placeholder="Ex.: irrigação soja",
    )

    if query.strip():
        query_tokens = preprocess(
            query,
            use_stopwords=use_stopwords,
            use_stemming=use_stemming,
        )

        st.markdown("**Tokens da consulta:**")
        st.code(" | ".join(query_tokens) if query_tokens else "(nenhum token)")

        scores = query_scores(query_tokens, processed_docs)

        ranking = []
        for doc_id, data in scores.items():
            cosine = cosine_similarity(
                query_tokens,
                processed_docs[doc_id],
                processed_docs,
            )
            ranking.append({
                "Documento": f"Doc {doc_id}",
                "TF-IDF acumulado": data["score"],
                "Similaridade de Cosseno": cosine,
                "Termos encontrados": ", ".join(data["matched_terms"]) or "—",
                "Trecho": DOCUMENTS[doc_id],
                "_doc_id": doc_id,
            })

        ranking.sort(key=lambda row: row["TF-IDF acumulado"], reverse=True)

        if ranking and ranking[0]["TF-IDF acumulado"] > 0:
            winner = ranking[0]
            st.success(
                f"Documento mais relevante pelo TF-IDF acumulado: "
                f"{winner['Documento']}"
            )

        display_ranking = [
            {
                "Documento": row["Documento"],
                "TF-IDF acumulado": round(row["TF-IDF acumulado"], 6),
                "Similaridade de Cosseno": round(
                    row["Similaridade de Cosseno"], 6
                ),
                "Termos encontrados": row["Termos encontrados"],
            }
            for row in ranking
        ]

        st.dataframe(
            display_ranking,
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Trechos ranqueados")

        for position, row in enumerate(ranking, start=1):
            score = row["TF-IDF acumulado"]

            if position == 1 and score > 0:
                st.markdown(f"### 🏆 {row['Documento']}")
            else:
                st.markdown(f"### {position}. {row['Documento']}")

            st.write(row["Trecho"])
            st.caption(
                f"TF-IDF acumulado: {score:.6f} | "
                f"Cosseno: {row['Similaridade de Cosseno']:.6f}"
            )
            st.divider()

    else:
        st.info("Digite uma consulta para iniciar a busca.")

# ============================================================
# TAB 2 - PRÉ-PROCESSAMENTO
# ============================================================
with tab2:
    st.header("Pipeline de Pré-processamento")

    st.write(
        "As quatro etapas demonstradas são: tokenização, normalização "
        "(lowercase + remoção de acentos), remoção opcional de stopwords "
        "e stemming opcional."
    )

    for doc_id, original_text in DOCUMENTS.items():
        with st.expander(f"Doc {doc_id}"):
            tokens_raw = tokenize(original_text)
            tokens_clean = processed_docs[doc_id]

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Texto original**")
                st.write(original_text)

                st.markdown("**Tokens após normalização**")
                st.code(" | ".join(tokens_raw))

            with col2:
                st.markdown("**Tokens finais**")
                st.code(" | ".join(tokens_clean))

            st.caption(
                f"Quantidade de tokens finais: {len(tokens_clean)}"
            )

# ============================================================
# TAB 3 - ÍNDICE INVERTIDO
# ============================================================
with tab3:
    st.header("Índice Invertido")
    st.write(
        "Estrutura Termo → [IDs dos documentos], construída a partir "
        "dos tokens pré-processados."
    )

    index_rows = [
        {
            "Termo": term,
            "Documentos": ", ".join(f"Doc {doc_id}" for doc_id in doc_ids),
            "DF": len(doc_ids),
        }
        for term, doc_ids in inverted_index.items()
    ]

    st.dataframe(
        index_rows,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Representação JSON")

    st.json({
        term: [f"Doc {doc_id}" for doc_id in doc_ids]
        for term, doc_ids in inverted_index.items()
    })

# ============================================================
# TAB 4 - TF-IDF
# ============================================================
with tab4:
    st.header("Cálculo TF-IDF")

    st.markdown("### Fórmulas utilizadas")

    st.latex(r"TF(t,d) = \frac{\text{frequência de }t\text{ em }d}{\text{número total de termos em }d}")

    st.latex(r"IDF(t) = \log\left(\frac{N}{1+DF(t)}\right) + 1")

    st.latex(r"TF\text{-}IDF(t,d) = TF(t,d) \times IDF(t)")

    st.write(
        "A tabela abaixo mostra os valores calculados para cada termo "
        "presente nos documentos."
    )

    tfidf_rows = build_tf_idf_table(processed_docs)

    st.dataframe(
        tfidf_rows,
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "Na aba Busca, o ranking utiliza o TF-IDF dos termos da consulta "
        "encontrados em cada documento. A similaridade de cosseno também "
        "é exibida como implementação do desafio bônus."
    )

st.divider()
st.caption(
    "AgroSearch — Laboratório Prático 04 | Recuperação de Informação / PLN"
)
