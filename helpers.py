import json
import os
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
from concurrent.futures import ThreadPoolExecutor


# Cargar variables de entorno desde el archivo .env
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Función para obtener la respuesta de GPT
def gpt_json(prompt,data, model="gpt-3.5-turbo"):
  response = client.chat.completions.create(
    model=model,
    max_tokens=3900,
    temperature=0,
    response_format={ "type": "json_object" },
    messages=[
    {"role": "system", "content": prompt},
    {"role": "user", "content": data}
    ]
  )
  return response.choices[0].message.content

def process_chunk(prompt, chunk, model="gpt-3.5-turbo"):
    """Funcion auxiliar para procesar un chunk de un dataframe en paralelo."""
    chunk_json = chunk.to_json(orient="records")#, index=False) #ACA INDEX FALSE ME TRAE UN PROBLEMA
    response = gpt_json(prompt=prompt+"\nANSWER EACH ID", data=chunk_json, model=model)
    response = json.loads(response)
    return pd.DataFrame(response['results'])


def gpt_df_paralell(prompt, df, text_col, id_col="id", model="gpt-3.5-turbo", step=30, max_workers=10):
    relevant_cols = df[[id_col, text_col]]
    chunks = [relevant_cols[i:i+step] for i in range(0, relevant_cols.shape[0], step)]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(lambda chunk: process_chunk(prompt, chunk, model), chunks))

    output = pd.concat(results, ignore_index=True)
    deliver = pd.merge(df, output, on=id_col)
    return deliver

def table_sample(df, n=75, type='head'):
    if type == 'head':
        return df.head(n)
    elif type == 'tail':
        return df.tail(n)
    else:
        return df.sample(n)