import asyncpg

# Database connection settings
DATABASE_URL = "postgresql://username:password@localhost/dbname"


async def get_db_connection():
    return await asyncpg.connect(DATABASE_URL)


async def save_pdf_to_db(note_id: int, file_name: str, file_url: str):
    """
    Save the MinIO file URL to the database.
    """
    conn = await get_db_connection()
    try:
        await conn.execute(
            """
            INSERT INTO documents (note_id, file_name, file_url)
            VALUES ($1, $2, $3)
            """,
            note_id,
            file_name,
            file_url,
        )
    finally:
        await conn.close()
