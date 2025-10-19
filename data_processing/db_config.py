# Production database (will no longer function once I remove paid plan)
# config = {
#     'MONGO_USERNAME': 'samlearner',
#     'MONGO_PASSWORD': 'e2oetyiluYFjkA0J',
#     'MONGO_CLUSTER_ID': 'wc7hc',
#     'MONGO_DB': 'letterboxd'
# }

# Local database
# config = {
#     'MONGO_DB': 'letterboxd',
#     'CONNECTION_URL': 'mongodb://localhost:27017/'
# }

# Remote database
config = {
    'MONGO_DB': 'letterboxd',
    'CONNECTION_URL': "mongodb+srv://samlearner:LKezA5hWxdaReAYM@serverlessinstance1.wc7hc.mongodb.net/letterboxd?retryWrites=true&w=majority&authSource=admin"
}

tmdb_key = "547d1e38e1c78598070786f3a16681bc"