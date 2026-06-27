import psycopg2
from chefai.core.config import get_settings

def get_connection():
    settings = get_settings()
    return psycopg2.connect(
        dbname=settings.db_name,
        user=settings.db_user,
        password=settings.db_password,
        host=settings.db_host,
        port=settings.db_port
    )

