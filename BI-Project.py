# Inteligencia de Negocios - PROYECTO FINAL

#Alumnos: 
#Sánchez Vázquez Alejandro Enrique
#Flores Campos Victor
#Millan Romero Gerardo


import streamlit as st
import pandas as pd               # Para la manipulación y análisis de datos
import numpy as np                # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt   # Para la generación de gráficas a partir de los datos
import seaborn as sns             # Para la visualización de datos basado en matplotlib
import io                         #Canalización de la salida de DataFrame.info() al búfer en lugar de sys.stdout - Para creación de contenido en el búfer y su respectiva escritura en un archivo de texto
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

from PIL import Image


#Biblioteca para implementar la generación de reglas de asociación mediante algoritmo A priori
from apyori import apriori

st.set_page_config(
     page_title="BI Rules",
     page_icon="increase.png",
     layout="wide",
     initial_sidebar_state="expanded",
     menu_items={
         'Get help': 'https://docs.streamlit.io',
         'Report a bug': "https://discuss.streamlit.io",
         'About': "Esta plataforma de Inteligencia de Negocios con Machine Learning fue desarrollada por alumnos de la  Facultad de Ingeniería de la UNAM (Sánchez V. Alejandro Enrique, Flores C. Victor y Millan R. Gerardo), como proyecto final de la asignatura de Inteligencia de Negocios, impartida por la profesora Ann Godelieve Wellens."
     }
 )


image = Image.open('association-rules.png')
image2 = Image.open('rules-logo.jpeg')
st.sidebar.image(image2)


@st.cache
def convert_df(df):
   return df.to_csv().encode('utf-8')

def explore(data):
    df_types = pd.DataFrame(data.dtypes, columns=['Data Type'])
    numerical_cols = df_types[~df_types['Data Type'].isin(['object','bool'])].index.values
    df_types['Count'] = data.count()
    df_types['Unique Values'] = data.nunique()
    #df_types['Min'] = data[numerical_cols].min()
    #df_types['Max'] = data[numerical_cols].max()
    #df_types['Average'] = data[numerical_cols].mean()
    #df_types['Median'] = data[numerical_cols].median()
    #df_types['St. Dev.'] = data[numerical_cols].std()
    return df_types.astype(str)
    #Solución a dataframe.dtypes
    #https://discuss.streamlit.io/t/streamlitapiexception-unable-to-convert-numpy-dtype-to-pyarrow-datatype/18253



#tab para identar hacía la derecha
#ctrl+tab para identar a la izquierda
# ctrl + } para comentar bloques de código

#FUNCIÓN PARA LIMPIAR EL DATAFRAME
def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df necesita ser un pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)



#algoritmos0=['Home']


st.sidebar.title("BI Rules")

st.sidebar.write("A través del uso de esta plataforma de Inteligencia de Negocios, usted podrá encontrar reglas de asociación entre productos/items para ofrecer promociones y recomendaciones más atractivas para sus clientes.")


st.sidebar.write("\n")

#st.sidebar.subheader("Componente #1")

#st.sidebar.subheader("Componente #2")


options=['Información del proyecto','Análisis Exploratorio de Datos','Generación de Reglas de Asociación']

componente= st.sidebar.selectbox("Selecciona la opción que deseas visualizar", options, index=0, key=1000, help="Después de seleccionar un componente, se desplegárán las opciones de los algoritmos disponibles.", on_change=None, args=None, kwargs=None)


#st.markdown("![Alt Text](https://media.giphy.com/media/pOEbLRT4SwD35IELiQ/giphy.gif)")
#st.markdown("![Alt Text](https://media.giphy.com/media/lbcLMX9B6sTsGjUmS3/giphy.gif)")


image3 = Image.open('bd2.png')

st.image(image3)







#st.title("Minería de datos - Proyecto Final")
st.title("BI Rules")
st.info("Bienvenid@ a esta plataforma creada como proyecto final de la asignatura de Inteligencia de Negocios por alumnos de la Facultad de Ingeniería de la UNAM.")


if componente == 'Información del proyecto':
 


    st.title("Conceptos fundamentales y teoría detrás del desarrollo de esta plataforma.")

    st.write("En esta sección podrás conocer más acerca de los conceptos fundamentales que forman la base de la realización y el comprendimiento de este proyecto.")

    st.markdown('# **Inteligencia de negocios**')

    image4 = Image.open('business-intelligence.png')

    st.image(image4)

    

    st.markdown('La inteligencia de negocios (BI) combina análisis de negocios, minería de datos, visualización de datos, herramientas e infraestructura de datos, y las prácticas recomendadas para ayudar a las organizaciones a tomar decisiones más basadas en los datos.')

    st.subheader('¿Por qué es importante la inteligencia de negocios?')


    st.markdown('La inteligencia de negocios muestra datos actuales e históricos dentro de su contexto empresarial para que las empresas tomen mejores decisiones. Los analistas pueden aprovechar BI para proporcionar puntos de referencia de rendimiento y de la competencia para que la organización funcione de manera más fluida y eficiente.')


    st.markdown('**Algunas formas en que la inteligencia de negocios puede ayudar a las empresas a tomar decisiones más inteligentes basadas en los datos:**')

    st.markdown('*   Identificar maneras de aumentar las ganancias')

    st.markdown('*   Comparar datos con los competidores')

    st.markdown('*   Rastrear el rendimiento')

    st.markdown('*   Optimizar operaciones')

    st.markdown('*   Predecir el éxito')

    st.markdown('*   Identificar las tendencias del mercado')

    st.markdown('*   Descubrir inconvenientes o problemas')



    image5 = Image.open('business-intelligence_2.jpg')

    st.image(image5)


    st.markdown('# **Análisis Exploratorio de Datos**')

    st.markdown("El análisis exploratorio de datos (EDA por sus siglas en inglés) implica el uso de gráficos y visualizaciones para explorar y analizar un conjunto de datos. El objetivo es explorar, investigar y aprender, no confirmar hipótesis estadísticas.")

    st.markdown("El análisis exploratorio de datos es una potente herramienta para explorar un conjunto de datos. Incluso cuando su objetivo es efectuar análisis planificados, el EDA puede utilizarse para limpiar datos, para análisis de subgrupos o simplemente para comprender mejor los datos.")

    image51 = Image.open('EDA.jpg')

    st.image(image51)


    st.markdown('# **Aprendizaje Máquina**')

    image6 = Image.open('machine-learning.jpg')

    st.image(image6)

    st.markdown("El Machine Learning (Aprendizaje automático), es una rama de la inteligencia artificial que permite que las máquinas aprendan sin ser expresamente programadas para ello. Una habilidad indispensable para hacer sistemas capaces de identificar patrones entre los datos para hacer predicciones. Esta tecnología está presente en un sinfín de aplicaciones como las recomendaciones de Netflix o Spotify, las respuestas inteligentes de Gmail o el habla de Siri y Alexa.")

    st.markdown("Los algoritmos del aprendizaje automático se clasifican a menudo como supervisados ​​o no supervisados. Los algoritmos supervisados ​​pueden aplicar lo que se ha aprendido en el pasado a nuevos datos. Los algoritmos no supervisados ​​pueden extraer inferencias de conjuntos de datos.")

    st.markdown("A continuación se presenta un esuqema de las diferentes áreas que presenta esta rama del conocimiento, perteneciente a la Inteligencia Artificial.")

    st.write("")

    image7 = Image.open('ML.PNG')

    st.image(image7)



    st.markdown('# **Sistemas de recomendación**')

    image8 = Image.open('src4.jpg')

    st.image(image8)

    st.markdown("Los sistemas de recomendación son herramientas importantes que ayudan a los usuarios a conocer opciones o elementos de interés para personalizar la experiencia del usuario. Tenemos contacto con estos poderosos sistemas de recomendación a diario.")

    st.markdown("Sin duda, los casos más conocidos de uso de esta tecnología son Netflix acertando en recomendar series y películas, Spotify sugiriendo canciones y artistas ó Amazon ofreciendo productos de venta cruzada muy tentadores para cada usuario.Pero también Google nos sugiere búsquedas relacionadas, Android aplicaciones en su tienda y Facebook amistades. O las típicas “lecturas relacionadas” en los blogs y periódicos.")

    st.markdown("Los sistemas de machine learning están revolucionando la forma en la que la recomendación de productos automatizada funciona. Antes, los resultados de los motores de búsqueda se ordenaban en base a dónde y cuán frecuentemente un texto se encontraba en la metadata del producto. Hoy, los algoritmos de búsqueda son capaces de utilizar más datos e información más detallada. Los resultados de la recomendación son sin duda alguna mucho más útiles y atractivos para el usuario a través del uso del aprendizaje Máquina.")

    image9 = Image.open('recommender_systems.png')

    st.image(image9)



    st.markdown('# **Reglas de Asociación**')

    image10 = Image.open('rules.png')

    st.image(image10)

    st.markdown("Los algoritmos de reglas de asociación tienen como objetivo encontrar relaciones dentro un conjunto de transacciones, en concreto, items o atributos que tienden a ocurrir de forma conjunta. En este contexto, el término transacción hace referencia a cada grupo de eventos que están asociados de alguna forma, por ejemplo:")

    st.markdown('*   La cesta de la compra en un supermercado.')

    st.markdown('*   Los libros que compra un cliente en una librería.')

    st.markdown('*   Las páginas web visitadas por un usuarios')

    st.markdown('*   Las características que aparecen de forma conjunta.')

    st.markdown('A cada uno de los eventos o elementos que forman parte de una transacción se le conoce como item y a un conjunto de ellos itemset. Una transacción puede estar formada por uno o varios items, en el caso de ser varios, cada posible subconjunto de ellos es un itemset distinto. Por ejemplo, la transacción T = {A,B,C} está formada por 3 items (A, B y C) y sus posibles itemsets son: {A,B,C}, {A,B}, {B,C}, {A,C}, {A}, {B} y {C}.')

    st.markdown("Una regla de asociación se define como una implicación del tipo “si X entonces Y” (X⇒Y), donde X e Y son itemsets o items individuales. El lado izquierdo de la regla recibe el nombre de antecedente o lenft-hand-side (LHS) y el lado derecho el nombre de consecuente o right-hand-side (RHS). Por ejemplo, la regla {A,B} => {C} significa que, cuando ocurren A y B, también ocurre C.")


    st.markdown("Existen varios algoritmos diseñados para identificar itemsets frecuentes y reglas de asociación.")



    st.markdown('# **Algoritmo APriori**')


    image11 = Image.open('apriori0.jpeg')

    st.image(image11)

    

    st.markdown("El algoritmo a priori es un algoritmo utilizado en minería de datos, sobre bases de datos transaccionales, que permite encontrar de forma eficiente conjuntos de ítems frecuentes, los cuales sirven de base para generar reglas de asociación. Procede identificando los ítems individuales frecuentes en la base y extendiéndolos a conjuntos de mayor tamaño siempre y cuando esos conjuntos de datos aparezcan suficientemente seguidos en dicha base de datos. Este algoritmo se ha aplicado grandemente en el análisis de transacciones comerciales y en problemas de predicción.")

    image12 = Image.open('apriori.jpg')

    st.image(image12)

    st.markdown("El funcionamiento de este algoritmo radica en comenzar con todos los elementos de la lista del conjunto de elementos. Luego, los candidatos se generan por autounión. Extendemos la longitud de los conjuntos de elementos un elemento a la vez. La prueba de subconjunto se realiza en cada etapa y se eliminan los conjuntos de elementos que contienen subconjuntos poco frecuentes. Repetimos el proceso hasta que no se puedan derivar más conjuntos de elementos exitosos de los datos.")

    image13 = Image.open('apriori2.png')

    st.image(image13)




    st.markdown("Para finalizar, se hace la definición de los parámetros que este algoritmo necesita para generar las reglas de asociación:")

    st.markdown('*   Soporte (Support): El soporte del ítem o itemset X es el número de transacciones que contienen X dividido entre el total de transacciones.')

    st.markdown('*   Confianza (Confidence): Es la probabilidad condicional de que una transacción que contenga un ítem {X}, también contenga un ítem {Y}.')

    st.markdown('*   Levantamiento (Lift): El indicador lift expresa cuál es la proporción del soporte observado de un conjunto de productos respecto del soporte teórico de ese conjunto dado el supuesto de independencia.')


    image14 = Image.open('Lift_Confidence_Support.png')

    st.image(image14)


    st.header("Propósito del proyecto")

    st.subheader("El propósito de este proyecto es aplicar los conocimientos teóricos vistos en la clase de Inteligencia de Negocios, para que a través de esta plataforma el usuario pueda generar reglas de asociación; con el fin de apoyar a cualquier cliente a potencializar su negocio a través del incremento de sus ventas o usuarios mediante relaciones importantes entre ítems que permitirán ofrecer recomendaciones y promociones a los usuarios.")

    st.subheader("Además de ello, esta plataforma también brindará al usuario la posibilidad de generar un Análisis Exploratorio De Datos, el cual le será de utilidad para conocer de una mejor manera los datos con los cuales generará las reglas de asociación.")

    st.header("Arquitectura del proyecto")

    image14 = Image.open('arquitectura.jpg')

    st.image(image14)


    st.header("Algoritmos y funcionalidades")


    image15 = Image.open('eda_1.jpg')

    st.image(image15)

    image16 = Image.open('algoritmo_apriori.jpg')

    st.image(image16)




elif componente == 'Análisis Exploratorio de Datos':


    st.title("Análisis Exploratorio de Datos")


    dataset_EDA_Avanzado=st.file_uploader("Introduce tu archivo CSV para realizar el Análisis Exploratorio De Datos Avanzado", type='csv', accept_multiple_files=False, key=2, help="Sólo se permite la carga de un sólo archivo en formato CSV",on_change = None, args = None, kwargs = None)
    
    if dataset_EDA_Avanzado is not None:
        st.write("")
        dataframe_Advanced = pd.read_csv(dataset_EDA_Avanzado)
        st.subheader("Visualización del dataframe generado con el archivo cargado")
        st.dataframe(dataframe_Advanced)

        st.subheader("Análisis del perfil del conjunto de datos")

        profile2 = ProfileReport(dataframe_Advanced, explorative=True)

        with st.spinner(text="En progreso (esto puede tardar algunos segundos)..."):
            

            st_profile_report(profile2)

        st.success('¡Análisis Exploratorio de Datos generado exitosamente!')





elif componente ==  "Generación de Reglas de Asociación":

    st.title("Generación de Reglas de Asociación a través del uso del algoritmo APriori")




    dataset=st.file_uploader("Introduce tu archivo CSV para hacer uso del algoritmo A Priori", type='csv', accept_multiple_files=False, key=7, help="Sólo se permite la carga de un sólo archivo en formato CSV",on_change = None, args = None, kwargs = None)
    
    if dataset is not None:
        st.write("")
        
        st.subheader("Visualización del dataframe generado con el archivo cargado")
        

        df_options=['El dataframe cargado tiene headers','El dataframe cargado no tiene headers']
        st.write("Para una mejor lectura del conjuto de datos, indíca si tu dataframe tiene headers o no a través de la siguiente sección.")
        df_option_selected = st.radio("Selecciona por favor la configuración que tiene el dataframe", (df_options), index=0)
            
        if df_option_selected == 'El dataframe cargado tiene headers':
            dataframe = pd.read_csv(dataset)

        elif df_option_selected == 'El dataframe cargado no tiene headers':
            dataframe = pd.read_csv(dataset, header=None)
            

        st.dataframe(dataframe)

        st.subheader("Número de filas y columnas")
        #Obtención del número de filas del dataframe
        filas=dataframe.index
        num_filas=(len(filas))

        #Obtención del número de columnas del dataframe
        colDF=dataframe.columns
        num_columnas=len(colDF)


        col1, col2 = st.columns(2)
        col1.metric('Número de filas del dataset cargado', num_filas)
        col2.metric('Número de columnas del dataset cargado', num_columnas)


        st.header("Exploración del conjunto de datos")
        st.write("Antes de ejecutar el algoritmo es ampliamente recomendable observar la distribución de la frecuencia de los elementos.")
        st.write('Para ello, se generará una gráfica de frecuencia para visualizar los items para importantes del dataset, por lo cual se procede a generar una tabla de frecuencia respecto a cada item.')
        

        Transacciones = dataframe.values.reshape(-1).tolist() #-1 significa 'dimensión no conocida'
        
        apartado_trasacciones_df=Transacciones

        #Se crea una matriz (dataframe) usando la lista y se incluye una columna 'Frecuencia'
        Lista = pd.DataFrame(Transacciones)
        Lista['Frecuencia'] = 1

        #Se agrupa los elementos
        Lista = Lista.groupby(by=[0], as_index=False).count().sort_values(by=['Frecuencia'], ascending=False) #Conteo
        Lista['Porcentaje'] = (Lista['Frecuencia'] / Lista['Frecuencia'].sum()) #Porcentaje
        Lista = Lista.rename(columns={0 : 'Item'})

        #Se muestra la lista
        st.dataframe(Lista)


        if st.button("Si deseas visualizar la lista de transacciones, presiona este botón.", key=601, help='Esta sección puede ser de utilidad si deseas conocer el número total de transacciones', on_click=None, args=None, kwargs=None):
            st.dataframe(apartado_trasacciones_df)    #visualización de lista de transacciones generada


            filas_lista_transacciones=apartado_trasacciones_df.index
            num_filas_transacciones=(len(apartado_trasacciones_df))
            st.write("Número de transacciones:")
            st.subheader(num_filas_transacciones)


        else:

            st.write("")


        st.header("Gráfica de barras de frecuencia por cada item del conjunto de datos")

        graph_601 ,ax= plt.subplots(figsize=(16, 20), dpi=1200)
        plt.ylabel('Item')
        plt.xlabel('Frecuencia')
        plt.barh(Lista['Item'], width=Lista['Frecuencia'], color='blue')

        st.pyplot(graph_601)


        st.header("Algoritmo A Priori")
        st.subheader("Preparación de la data")
        st.write("La función Apriori de Python requiere que el conjunto de datos tenga la forma de una lista de listas, donde cada transacción es una lista interna dentro de una gran lista. Los datos actuales están en un dataframe de Pandas, por lo que, se requiere convertir en una lista.")


        TransaccionesLista = dataframe.stack().groupby(level=0).apply(list).tolist()


        NuevaLista = []

        for item in TransaccionesLista:
          if str(item) != 'nan':
            NuevaLista.append(item)
        #print(NuevaLista)
        #st.write(NuevaLista)



        if st.button("Si deseas visualizar la lista de listas generada en esta preparación de la data, presiona este botón.", key=602, help='Esta sección puede ser de utilidad si deseas conocer el número total de transacciones', on_click=None, args=None, kwargs=None):
            st.write(NuevaLista)


        else:
            st.write("")




        #Visualización de una lista en específico
        st.subheader("Visualización de una lista en específico")

        num_lista_RA = st.number_input('Introduce el número de la lista N que deseas visualizar', min_value=0, max_value=num_filas-1, step=1, format=None, key='num_lista_apartadoextraRA', help='Sólo se admiten valores numéricos enteros positivos', on_change=None, args=None, kwargs=None)
        st.write("El máximo número que puedes ingresar es:", num_filas-1)
        num_lista_RA= int(num_lista_RA)    


        st.write(NuevaLista[num_lista_RA])    








        st.header("Aplicación del algoritmo")
        st.write("En esta sección, deberás de introducir los valores requeridos para llevar a cabo la generación de reglas de asociación mediante el algoritmo A priori, los cuales son el soporte, la confianza y la elevación.")

        st.write("Recuerda que el soporte mínimo definirá el grado de importancia que tendrá la regla. Como recomnendación y a manera de ejemplo, pueden calcularse a través del cálculo de aquellos productos que se hayan transaccionado al menos n veces por día o semana respecto a las operaciones llevadas a cabo por cada cliente")

        st.write("La confianza mínima es la fiabilidad de la regla, la cual de acuerdo al valor n establecido establecerá que el n porciento de las transacciones tendrán esa característica.")

        st.write("Finalmente, deberás seleccionar también el grado de elevación que deseas tengan las reglas de asociación. Este valor n hará que la probabilidad de transaccionar ese item aumente n veces, y siempre debe ser mayor a uno.")





        st.subheader("Parámetros de generación de reglas de asociación")
        st.write("Una vez aclarado lo anteror, introduce los parámetros con los cuales quieres ejecutar el algoritmo para generar las reglas.")
        
        valor_soporte_min = st.number_input('Introduce el valor del soporte mínimo que deseas utilizar', min_value=0.0001, max_value=1.0, format="%.4f", key='valor_RA_SOPORTE', help='Sólo se admiten valores numéricos positivos entre 0.0001 y 1', on_change=None, args=None, kwargs=None)
        #st.write(valor_soporte)


        valor_confianza = st.number_input('Introduce el valor del porcentaje que deses utilizar para el parámetro de la confianza', min_value=0.0001, max_value=1.0, format="%.4f", key='valor_RA_confianza', help='Sólo se admiten valores numéricos positivos entre 0.0001 y 1', on_change=None, args=None, kwargs=None)
        #st.write(valor_confianza)

        valor_elevacion = st.number_input('Introduce el valor de elevación que deseas utilizar. Recuerda que debe ser mayor a uno', min_value=1.001, max_value=None, format="%.3f", key='valor_RA_elevacion', help='Sólo se admiten valores numéricos mayores a 1.001', on_change=None, args=None, kwargs=None)
        #st.write(valor_soporte)

        ReglasC1 = apriori(TransaccionesLista, 
               min_support = valor_soporte_min, 
               min_confidence = valor_confianza, 
               min_lif = valor_elevacion)





        if st.button("Presiona este botón para generar las reglas de asociación", key=605, help=None, on_click=None, args=None, kwargs=None):

                    
            with st.spinner("Esto podría tardar algunos segundos, por favor espera."):


                ResultadosC1 = list(ReglasC1)

                for item in ResultadosC1:

                    #El primer índice de la lista
                    Emparejar = item[0]
                    items = [x for x in Emparejar]
                    st.write("Regla: " + str(item[0]))

                    #El segundo índice de la lista
                    st.write("Soporte: " + str(item[1]))

                    #El tercer índice de la lista
                    st.write("Confianza: " + str(item[2][0][2]))
                    st.write("Li: " + str(item[2][0][3])) 
                    st.write("=====================================") 
                        
        

            st.success('Reglas de asociación generadas exitosamente')


        else:

            st.write("")









