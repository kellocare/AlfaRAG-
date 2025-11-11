# Для маленьких датасетов (< 100K)
small_config = {
    'M': 16,
    'ef_construction': 100,
    'ef_search': 50
}

# Для средних датасетов (100K - 1M)
medium_config = {
    'M': 24, 
    'ef_construction': 200,
    'ef_search': 128
}

# Для больших датасетов (> 1M)
large_config = {
    'M': 32,
    'ef_construction': 400, 
    'ef_search': 256
}