import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

# Função para carregar o dataset
@st.cache
def get_data():
    return pd.read_csv("train.csv")

# Função para treinar o modelo
def train_model():
    data = get_data()
    x = data.drop("fake", axis=1)
    y = data["fake"]
    rf_regressor = RandomForestClassifier()
    rf_regressor.fit(x, y)
    return rf_regressor

# Criando um dataframe
data = get_data()

# Treinando o modelo
model = train_model()

# Título
st.title("Authentiq - Previsão de Perfis Fakes no Instagram")

# Subtítulo
st.markdown("Este é um Data App utilizado para exibir a solução de Machine Learning para o problema de previsão de perfis fakes no Instagram.")

# Layout em colunas
col1, col2 = st.beta_columns(2)

# Verificando o dataset
with col1:
    st.subheader("Selecionando Atributos")
    # Atributos para serem exibidos por padrão
    defaultcols = ["profile pic", "nums/length username", "fullname words", "nums/length fullname", "name==username"]
    # Definindo atributos a partir do multiselect
    cols = st.multiselect("Selecione as Features", data.columns.tolist(), default=defaultcols)
    # Exibindo os top 10 registros do dataframe
    st.dataframe(data[cols])

# Barra Lateral
with col2:
    st.sidebar.subheader("Defina as Características do Perfil para Previsão")
    # Mapeando dados do usuário para cada atributo
    pic = st.sidebar.selectbox("Tem foto de perfil?", ("Sim", "Não"))
    pic = 1 if pic == "Sim" else 0

    private = st.sidebar.selectbox("O perfil é Privado?", ("Sim", "Não"))
    private = 1 if private == "Sim" else 0

    seguidores = st.sidebar.number_input("Número de seguidores", min_value=0, value=100)
    segue = st.sidebar.number_input("Número de pessoas que o perfil segue", min_value=0, value=100)
    posts = st.sidebar.number_input("Número de publicações", min_value=0, value=10)

    nome_do_usuario = st.sidebar.text_input('Nome do Usuário').lower()
    nome_real_cadastrado = st.sidebar.text_input('Nome Real Cadastrado').lower()
    bio = st.sidebar.text_input('Bio (Descrição)').lower()

    # Inserindo um botão na tela
    btn_load = st.sidebar.button("Fazer Previsão")

    # Verifica se o botão foi acionado
    if btn_load:
        # Calculando variáveis
        nums_lenght_username = sum(c.isdigit() for c in nome_do_usuario) / max(len(nome_do_usuario), 1)
        fulname_words = len(nome_do_usuario.split())

        numeros_nome = sum(c.isdigit() for c in nome_real_cadastrado)
        num_lenght_fullname = numeros_nome / max(len(nome_real_cadastrado.split()), 1)

        name_username = 1 if nome_do_usuario == nome_real_cadastrado else 0
        description_length = len(bio.replace(" ", ''))
        external_url = 1 if "https://" in bio or "http://" in bio else 0

        # Faz a previsão
        result = model.predict([[pic, nums_lenght_username, fulname_words, num_lenght_fullname, name_username,
                                 description_length, external_url, private, posts, seguidores, segue]])
        result = result[0]

        # Exibindo resultado
        st.subheader("Resultado da Previsão:")
        if result == 1:
            st.write("É Fake!")
        else:
            st.write("Não é Fake!")
