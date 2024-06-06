# ---------------------------------------------------------------------------- #
#                              Avaliação Global 2                              #
# Nomes: Álvaro Lúcio Almeida Ribeiro         163 - GES                        #
#        Ewel Fernandes Pereira               167 - GES                        #
# ---------------------------------------------------------------------------- #

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# 2 - Lendo o CSV
df = pd.read_csv('palmerpenguins.csv', encoding="utf-8")

# 3 - Conversão
df['island'] = df['island'].replace({'Biscoe': 0, 'Dream': 1, 'Torgersen': 2})
df['sex'] = df['sex'].replace({'FEMALE': 0, 'MALE': 1})
df['species'] = df['species'].replace({'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2})

# 4 - Ordenação
df = df[['island', 'sex', 'culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g', 'species']]

# 5 - Separar em 80% e 20%
train, test = train_test_split(df, test_size=0.2, random_state=42)

# 6 - Escolha do Decision tree

# 7 - Treinar e Classificar o modelo
# Separar features e labels
X_train = train.drop('species', axis=1)
y_train = train['species']
X_test = test.drop('species', axis=1)
y_test = test['species']

# Instanciar o classificador
classifier = DecisionTreeClassifier()

# Treinar o modelo
classifier.fit(X_train, y_train)

# Classificar as amostras do conjunto de teste
predictions = classifier.predict(X_test)

# Exibir as primeiras previsões
predictions[:5]

# 8 - Exibir métricas de avaliação
print("Métricas de Avaliação:\n")
print(classification_report(y_test, predictions))

# 9 - Inserção de dados
def classify_penguin():
    print("Entre com as caracteristicas para o pinguim:")
    island = int(input("De qual ilha ele é? (0 para Biscoe, 1 para Dream, 2 para Torgersen): "))
    sex = int(input("Qual o sexo do pinguim? (0 para female, 1 para male): "))
    culmen_length_mm = float(input("Qual o comprimeinto do cúlmen? (em mm): "))
    culmen_depth_mm = float(input("Qual é a profundidade do cúlmen? (em mm): "))
    flipper_length_mm = float(input("Qual é o comprimento da nadadeira? (em mm): "))
    body_mass_g = float(input("E qual é a massa corporal (em g) do pinguim? "))
    
    input_data = pd.DataFrame([[island, sex, culmen_length_mm, culmen_depth_mm, flipper_length_mm, body_mass_g]],
                              columns=['island', 'sex', 'culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g'])
    
    prediction = classifier.predict(input_data)
    
    species_mapping = {0: 'Adelie', 1: 'Chinstrap', 2: 'Gentoo'}
    predicted_species = species_mapping[prediction[0]]
    
    print(f"A espécie do pinguim é: {predicted_species}")

classify_penguin()