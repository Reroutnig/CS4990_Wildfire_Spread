import earthaccess

earthaccess.login()

results = earthaccess.search_data(
    short_name="MOD13C2",
    temporal=("2020-01-01", "2024-12-31"),
    bounding_box=(-125, 25, -100, 50) # North america
)

# getting a few files for testing 
files = earthaccess.download(results[:5])