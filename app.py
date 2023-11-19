import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

# Definindo a cor de fundo para roxo
st.markdown(
    """
    <style>
        body {
            background-color: #800080;
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True
)

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
st.title("@uthentiq - Sua garantia de autenticidade nas redes sociais.")

# Subtítulo
st.markdown("Em um mundo digital repleto de conexões, como você pode ter certeza de que cada perfil é genuíno?.")

# Verificando o dataset
st.subheader("A realidade através de um pequeno conjunto de dados")

# Atributos para serem exibidos por padrão
defaultcols = ["profile pic", "nums/length username", "fullname words", "nums/length fullname", "name==username"]

# Definindo atributos a partir do multiselect
cols = st.multiselect("Dados", data.columns.tolist(), default=defaultcols)

# Exibindo os top 10 registros do dataframe
st.dataframe(data[cols])

## Barra Lateral
st.sidebar.subheader("Defina as características do perfil para previsão")

# Mapeando dados do usuário para cada atributo
pic = st.sidebar.selectbox("Tem foto de perfil?", ("Sim", "Não"))
# Transformando o dado de entrada em valor binário
pic = 1 if pic == "Sim" else 0

private = st.sidebar.selectbox("O perfil é Privado?", ("Sim", "Não"))
private = 1 if private == "Sim" else 0

seguidores = st.sidebar.number_input("Qual o número de seguidores que o perfil tem?", value=100)
segue = st.sidebar.number_input("Qual o número de pessoas que o perfil segue?", value=100)
posts = st.sidebar.number_input("Qual o número de publicações que o perfil já fez??", value=10)

nome_do_usuario = st.sidebar.text_input('Digite o nome do usuário:')
nome_do_usuario = nome_do_usuario.lower()

nome_real_cadastrado = st.sidebar.text_input('Digite o Nome real cadastrado:')
nome_real_cadastrado = nome_real_cadastrado.lower()

bio = st.sidebar.text_input('Digite a "Bio"(descrição) do Instagram do usuário:')
bio = bio.lower()

# Inserindo um botão na tela
btn_load = st.sidebar.button("Fazer previsão")

nums_lenght_username, fulname_words, num_lenght_fullname, name_username, description_length, external_url, result = 0, 0, 0, 0, 0, 0, 0

# Verifica se o botão foi acionado
if btn_load:
    # Calculando a variável "nums/lenght username"
    numeros = sum(c.isdigit() for c in nome_do_usuario)
    letras = sum(c.isalpha() for c in nome_do_usuario)
    espacos = sum(c.isspace() for c in nome_do_usuario)

    if numeros == 0:
        nums_lenght_username = 0
    else:
        nums_lenght_username = numeros / (letras + espacos + numeros)

    # Calculando a variável "fulname words"
    fulname_words = len(nome_do_usuario.split())

    # Calculando a variável "num/lenght fullname"
    numeros_nome = sum(c.isdigit() for c in nome_real_cadastrado)
    letras_nome = sum(c.isalpha() for c in nome_real_cadastrado)
    espacos_nome = sum(c.isspace() for c in nome_real_cadastrado)

    if numeros_nome == 0:
        num_lenght_fullname = 0
    else:
        num_lenght_fullname = numeros_nome / len(nome_real_cadastrado.split())

    # Verificando a variável "name==username"
    if nome_do_usuario == nome_real_cadastrado:
        name_username = 1
    else:
        name_username = 0

    # Calculando a variável "description_length"
    description_length = len(list(bio.replace(" ", '')))

    # Verificando a variável "external URL"
    if ("https://" in bio) or ("http://" in bio):
        external_url = 1
    else:
        external_url = 0

    result = model.predict(
        [[pic, nums_lenght_username, fulname_words, num_lenght_fullname, name_username,
          description_length, external_url, private, posts, seguidores, segue]])
    result = result[0]

    st.subheader("Este perfil...")
    if result == 1:
        st.write("É Fake!")
    else:
        st.write("Não é Fake!")
