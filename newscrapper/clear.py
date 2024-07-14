import redis

# Connect to KeyDB
r = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Flush all databases
r.flushall()

print("All data cleared from KeyDB.")