import json
import os
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
import tiktoken as tk
import math
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

def num_tokens_from_messages(messages:list, model:str = "gpt-3.5-turbo"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tk.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tk.get_encoding("cl100k_base")

    tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
    tokens_per_name = -1  # if there's a name, the role is omitted
    
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
                
    return num_tokens

def calculate_prompt_tokens(prompt:str, df:pd.DataFrame, id_col:str, text_col:str, model:str = "gpt-3.5-turbo", chunk_size:int = 30):
    """
    Calculates the total number of tokens used for a given prompt and DataFrame.

    Args:
        prompt (str): The prompt to be used in the conversation.
        df (pd.DataFrame): The DataFrame containing the data.
        id_col (str): The name of the column containing the ID.
        text_col (str): The name of the column containing the text.
        model (str, optional): The name of the language model to use. Defaults to "gpt-3.5-turbo".
        chunk_size (int, optional): The number of rows to process in each chunk. Defaults to 30.

    Returns:
        int: The total number of tokens required for the conversation.
    """
    
    total_tokens = 0
    for i in range(0, df[[id_col, text_col]].shape[0], chunk_size):
        chunk = df[[id_col, text_col]][i:i+chunk_size]
        messages = [{"role": "system", "content": prompt},
                    {"role": "user", "content": chunk.to_json(orient="records")}
            ]
    
        total_tokens += num_tokens_from_messages(messages)
    
    return math.ceil(total_tokens * 1.05)  # Add a 5% margin

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tk.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def estimate_response_tokens(df:pd.DataFrame, id_col:str, text_col:str, chunk_size=30):
    """
    Estimates the number of response tokens based on the given dataframe, id column, text column, and chunk size.

    Parameters:
    df (pandas.DataFrame): The dataframe containing the data.
    id_col (str): The name of the column containing the IDs.
    text_col (str): The name of the column containing the text.
    chunk_size (int, optional): The number of rows to process in each chunk. Defaults to 30.

    Returns:
    int: The estimated number of response tokens.
    """
    response_tokens = 0
    for i in range(0, df[[id_col, text_col]].shape[0], chunk_size):
        chunk_dict = df[i:i+chunk_size].to_dict(orient="records")
        json_string = '{{\n  "results": {}\n}}'.format(
            ',\n    '.join([f'{{ "id": {item[id_col]}, "answer": "{item[text_col]}" }}' for item in chunk_dict])
        )
        response_tokens += num_tokens_from_string(json_string)

    response_tokens *= 1.2  # Add a 20% margin

    return math.ceil(response_tokens)

def estimate_costs(prompt:str, df:pd.DataFrame, id_col:str, text_col:str, model:str = "gpt-3.5-turbo", chunk_size = 30):
    """
    Estimates the costs of using a language model to process a table.
    
    Args:
        prompt (str): The prompt to be used for generating responses.
        df (pd.DataFrame): The table data as a pandas DataFrame.
        id_col (str): The name of the column containing unique identifiers for each row.
        text_col (str): The name of the column containing the text data to be processed.
        model (str, optional): The language model to be used. Defaults to "gpt-3.5-turbo".
        chunk_size (int, optional): The number of rows to process in each chunk. Defaults to 30.
    
    Returns:
        float: The estimated total cost of processing the table.
    """
    
    costs_dict = {
        "gpt-3.5-turbo":{
            "prompt": 0.0005,
            "response": 0.0015
        },
        "gpt-4-turbo":{
            "prompt": 0.01,
            "response": 0.03
        }
    }
    
    prompt_tokens = calculate_prompt_tokens(prompt, df, id_col, text_col, model, chunk_size)
    response_tokens = estimate_response_tokens(df, id_col, text_col, chunk_size)
    
    prompt_cost = (costs_dict[model]["prompt"] * prompt_tokens) / 1000
    response_cost = (costs_dict[model]["response"] * response_tokens) / 1000
    
    total_cost = prompt_cost + response_cost
    
    return total_cost