
from src.db import get_connection
from psycopg2.extras import RealDictCursor

def search_recipes(embedding_vector, top_k=5):
    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    query = """
    SELECT id, title, ingredients, directions, embedding <#> %s::vector AS distance
    FROM recipes
    ORDER BY embedding <#> %s::vector
    LIMIT %s;
"""
    cur.execute(query, (embedding_vector, embedding_vector, top_k))

    results = cur.fetchall()

    cur.close()
    conn.close()
    return results
