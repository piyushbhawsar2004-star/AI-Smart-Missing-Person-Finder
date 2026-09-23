import getpass
import os

from werkzeug.security import generate_password_hash

from db import Database
from dotenv import load_dotenv

load_dotenv()
username = input('Admin username: ').strip()
password = getpass.getpass('Admin password: ')
if len(password) < 12:
    raise SystemExit('Use a password with at least 12 characters.')
database = Database(os.environ.get('MONGO_URI', 'mongodb://localhost:27017/missing_person_finder'))
database.users.update_one({'username': username}, {'$set': {'username': username, 'passwordHash': generate_password_hash(password), 'role': 'admin', 'active': True}}, upsert=True)
print(f'Admin user {username!r} created or updated.')
