# AgroSearch

Projeto do Laboratório Prático 04 - Desafio Integrador.

## Arquivos

- `app.py`: aplicação Streamlit completa.
- `requirements.txt`: dependência necessária.
- `README.md`: instruções de execução.

## Como executar

1. Abra um terminal na pasta do projeto.
2. Instale as dependências:

```bash
pip install -r requirements.txt
```

3. Execute:

```bash
streamlit run app.py
```

4. O Streamlit abrirá a aplicação no navegador.

## Funcionalidades

- Tokenização.
- Normalização: lowercase e remoção de acentos.
- Stopwords com checkbox para ligar/desligar.
- Stemming simples implementado no próprio código, sem scikit-learn.
- Índice invertido Termo -> documentos.
- TF, IDF e TF-IDF implementados manualmente.
- Ranking dos documentos por TF-IDF acumulado.
- Similaridade de cosseno como bônus.
- Interface organizada em abas.
