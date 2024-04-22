import os
import streamlit as st
import pandas as pd
from helpers import gpt_df_paralell, table_sample, estimate_costs

# streamlit run app.py --server.enableXsrfProtection false


# STYLES
with open( "style.css" ) as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)


# VARIABLES GLOBALES
default_prompt_context_value ="""La siguiente es una lista de respuestas a la pregunta:
¿Cuál es la primera marca en la que piensas cuando hablamos del Día de la Mujer?

Tu trabajo es normalizar cada registro en un formato correcto, y que valores que refieran a un mismo concepto tengan el mismo valor.
Usa mayúscula cuando corresponda."""
prompt_format = """
Debes responder en formato JSON con el siguiente formato:
results[
    {
    "id": xxx,
    "answer": "xxx"
    },
    {
    "id": xxx,
    "answer": "xxx"
    },
    {
    "id": xxx,
    "answer": "xxx"
    },
]"""
language = None
prompt_context = None
prompt_language = None
column = None
data_incomplete = True
model = None


# MAIN

# TITLE & LOGO
st.image("nodus_logo.png", caption=None, width=200, use_column_width=False, clamp=False, channels="RGB", output_format="auto")
st.title("CSV AI Processor")

# UPLOAD CSV
st.subheader("Your CSV file", divider=False)
user_csv = st.file_uploader(label="Upload CSV", type=['csv'], label_visibility="hidden")

if user_csv is not None:
    user_csv.seek(0)
    df_original = pd.read_csv(user_csv, low_memory=False)
    with st.expander("See your data table"):
        st.text(f"Numero de registros: {df_original.shape[0]}")
        st.dataframe(df_original, use_container_width=True)
    st.divider()
    

# Define Processing Parameters
    st.subheader("Define processing parameters")
    # Crear un dropdown para seleccionar una columna a procesar y el idioma
    col1, col2 = st.columns([2,1], gap="medium")
    with col1:
        prompt_context = st.text_area("Provide specifics about the task and context about the dataset.", 
                                  value=default_prompt_context_value, max_chars=None,
                                  height=207, 
                                  disabled=False, label_visibility="visible") 
    with col2:
        column_to_norm = st.selectbox('Select a field for processing:', df_original.columns)
        language = st.selectbox("Specify register's entries language", ['Español', 'Portugués', 'Inglés'])
        model = st.selectbox('Choose an AI model', ['gpt-3.5-turbo','gpt-4-turbo'])
    with st.expander("Additional settings"):
        output_column_name = st.text_input("Output column name", value="answer", max_chars=200, key=None, type='default')
        col3, col4 = st.columns([1,1], gap="medium")      
        with col3:
            step_size = st.number_input("Step size (If not sure, leave default value)", min_value=1, max_value=100, value=30, step=1, format=None, key=None)
        with col4:
            max_workers = st.number_input("Max workers (If not sure, leave default value)", min_value=1, max_value=20, value=10, step=1, format=None, key=None)
        is_sample = st.checkbox("Sample the data (for testing purposes)", value=False, key=None)
        if is_sample:
            col5, col6 = st.columns([1,1], gap="medium")
            with col5:
                sample_type = st.selectbox('Select a sample type:', ['head', 'tail', 'random'])
            with col6:
                sample_size = st.number_input("Sample size", min_value=1, value=75, step=1, format=None, key=None)
            
            df = table_sample(df = df_original, n = sample_size, type = sample_type)
        else:
            df = df_original
                
        # create a new column with the original id
        df['original_id'] = df.index
        # convert the column to string
        df[column_to_norm] = df[column_to_norm].astype(str)
        # create a new column with all values as lower case
        df['value'] = df[column_to_norm].str.lower()
        # create a df with the unique values of the column
        unique_values = df['value'].unique()
        unique_values = pd.DataFrame(unique_values, columns = ['value'])
        # order ascending
        # unique_values.sort_values(by='value', inplace=True)
        # create a new id column
        unique_values['id'] = unique_values.index
        # merge to df the ids from unique_values
        df = df.merge(unique_values, on = 'value', how = 'left')
        # run the normalization script using gpt

        
    if language is not None:
        if language == "Español":
            prompt_language = f"""Observación: Los registros deben ser procesados en: {language}."""
        elif language == "Portugués":
            prompt_language = f"""Observação: Os registros devem ser processados em: {language}."""
        elif language == "Inglés":
            prompt_language = f"""Note: Records must be processed in: {language}."""    
        with st.expander("See final prompt"):
            st.write(f"""{prompt_context}\n\n{prompt_language}\n""")
            st.code(prompt_format, language='python')
        data_incomplete = False

    prompt = f"""
    {prompt_context}\n 
    {prompt_language}
    {prompt_format}\n
    """
    
    total_cost = round(estimate_costs(prompt, unique_values, "id", "value", model, step_size), 5)

# START PROCESSING
st.divider()
st.subheader("Start your processing")

if 'total_cost' in globals():
    st.info(f"Estimated costs for processing your data: **U$D {total_cost}**", icon = "ℹ️")
    st.write("\n")

start_normalization = st.button("Start processing job", type='primary', disabled=data_incomplete)

if start_normalization:
    with st.spinner('Wait for it...'):
        results = gpt_df_paralell(prompt = prompt, # Elige el prompt
                        df = unique_values, # Elige df
                        text_col = "value", # Elige la columna de texto
                        id_col="id", # Elige la columna de id
                        model= model, # Elige el modelo
                        step=step_size,
                        max_workers=max_workers)
        # Final manipulation
        final = df.merge(results[['id','answer']], on='id', how='left')
        drop_cols = ['value', 'id', 'original_id']
        # Rename the answer column
        final.rename(columns={'answer': output_column_name}, inplace=True)
        final.drop(columns=drop_cols, inplace=True)

        # Show the final dataframe
        st.dataframe(final, use_container_width=True)