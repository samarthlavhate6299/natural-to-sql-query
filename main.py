import psycopg2
import re
import os
import google.generativeai as genai
from huggingface_hub import InferenceClient
import sounddevice as sd
import numpy as np


def get_db_connection():
    """
    Establishes a connection to the PostgreSQL database.
    """
    try:
        conn = psycopg2.connect(
            dbname="temp_db",
            user="postgres",
            password="your_new_password", 
            host="localhost",
            port="5432"
        )
        print("Database connection successful.")
        return conn
    except psycopg2.OperationalError as e:
        print(f"Database connection failed: {e}")
        return None


def get_database_schema(conn):
    """
    Fetches and returns the database schema.
    """
    if conn is None:
        print("Cannot get schema: No database connection.")
        return

    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    cols.table_name,
                    cols.column_name,
                    cols.data_type,
                    STRING_AGG(
                        CASE
                            WHEN kcu.table_name IS NOT NULL
                            THEN CONCAT(kcu.constraint_name, ' (', tc.constraint_type, ' -> ', ccu.table_name, '.', ccu.column_name, ')')
                            ELSE NULL
                        END,
                        ', '
                    ) AS "key_details"
                FROM 
                    information_schema.columns AS cols
                LEFT JOIN 
                    information_schema.key_column_usage AS kcu
                ON 
                    cols.table_name = kcu.table_name AND cols.column_name = kcu.column_name
                LEFT JOIN 
                    information_schema.table_constraints AS tc
                ON 
                    kcu.constraint_name = tc.constraint_name AND kcu.table_schema = tc.table_schema
                LEFT JOIN 
                    information_schema.constraint_column_usage AS ccu
                ON 
                    tc.constraint_name = ccu.constraint_name AND tc.table_schema = ccu.table_schema
                WHERE 
                    cols.table_schema = 'public'
                GROUP BY 
                    cols.table_name,
                    cols.column_name,
                    cols.data_type
                ORDER BY 
                    cols.table_name,
                    cols.column_name;
            """)
            return cur.fetchall()
    except psycopg2.Error as e:
        print(f"An error occurred while fetching the schema: {e}")


def configure_gemini(api_key: str):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.0-flash")


def generate_sql_query(model, user_input, schema):
    """
    Generate SQL query from natural language input.
    """
    user_message = f"""You are a world-class SQL generation bot. Your purpose is to convert a natural language question into a highly efficient 

          and accurate SQL query for a PostgreSQL database.

            **Rules:**

            - **Primary Goal:** You will write a SQL query that answers the user's question.

            - **No Explanations:** You will not provide any explanation, context, or conversational text.

            - **No Markdown:** You will not wrap the SQL query in markdown formatting like ```sql ... ```.

            - **Schema Adherence:** You MUST strictly use ONLY the tables and columns defined in the provided schema. Do not invent or assu.me the existence of any columns or tables not listed.

            - **Output Format:** Your response must be the raw SQL query and ONLY the raw SQL query. do not add any text like "Here is your query:{user_input}".

            - **Read-Only:** You are strictly forbidden from generating any SQL that modifies data. This includes `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, etc. You will ONLY generate `SELECT` statements.

            - **Clarity:** If a user's question is ambiguous, make the most logical assumption based on the schema to formulate the query.
           
            IMPORTANT: Your final output must be ONLY raw SQL text with no markdown, no code fences, and no additional words. 

            **Database Schema:**  
            {schema}
    """
    response = model.generate_content(user_message)
    return re.sub(r'```sql\n|```', '', response.text)


def analyze_sql_output(model, user_input, query_result):
    """
    Generate natural language analysis of SQL result.
    """
    systemprompt = f"""
        You are an expert data analyst. 

        Your task is to carefully analyze the SQL query result and provide a clear, accurate, and concise answer to the user's question.

        User Question:
        {user_input}

        SQL Query Result:
        {query_result}

        Instructions and Rule for Output:
        - Base your answer only on the SQL query result.
        - Do not mention SQL, queries, or database internals in the response.
        - Respond in plain natural language, directly answering the user's question.
        - If the result is empty or does not provide enough information, say so politely.
    """
    return model.generate_content(systemprompt).text


def speak_text(text, provider="replicate", api_key=""):
    """
    Convert text to speech and play audio.
    """
    client = InferenceClient(provider=provider, api_key=api_key)
    audio = client.text_to_speech(text, model="hexgrad/Kokoro-82M")
    audio_np = np.frombuffer(audio, dtype=np.int16)
    sd.play(audio_np, 19000)
    sd.wait()

def main():
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    REPLICATE_API_KEY = os.getenv('REPLICATE_API_KEY') 

    model = configure_gemini(GEMINI_API_KEY)
    conn = get_db_connection()

    if not conn:
        return

    schema = get_database_schema(conn)

    while True:
        user_input = input("Please enter your query: ")
        if user_input.lower() == "exit":
            break

        sql_query = generate_sql_query(model, user_input, schema)

        if sql_query == "Failed":
            print("Not able to generate SQL query")
            continue

        try:
            with conn.cursor() as cur:
                cur.execute(sql_query)
                result = cur.fetchall()
        except Exception as e:
            print(f"Error executing query: {e}")
            continue

        answer = analyze_sql_output(model, user_input, result)
        speak_text(answer, api_key=REPLICATE_API_KEY)

        print("Output:", answer)

    conn.close()
    print("Database connection closed.")


if __name__ == "__main__":
    main()
